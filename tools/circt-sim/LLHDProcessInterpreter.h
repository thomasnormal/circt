//===- LLHDProcessInterpreter.h - LLHD process interpretation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLHDProcessInterpreter class, which interprets LLHD
// process bodies during simulation. It handles:
// - Signal registration from llhd.sig ops
// - Time conversion from llhd.time to SimTime
// - Core operation handlers: llhd.prb, llhd.drv, llhd.wait, llhd.halt
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETER_H
#define CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETER_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/EventQueue.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <chrono>
#include <deque>
#include <map>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

// Forward declarations for SCF, Func, and LLVM dialects
namespace mlir {
namespace scf {
class IfOp;
class ForOp;
class WhileOp;
} // namespace scf
namespace func {
class CallOp;
class FuncOp;
} // namespace func
namespace LLVM {
class AllocaOp;
class LoadOp;
class StoreOp;
class GEPOp;
class CallOp;
class LLVMFuncOp;
class AddressOfOp;
class GlobalOp;
} // namespace LLVM
} // namespace mlir

namespace circt {
namespace sim {

class JITBlockCompiler;
struct JITBlockSpec;
class JITCompileManager;
struct ProcessThunkExecutionState;

//===----------------------------------------------------------------------===//
// InterpretedValue - Runtime value representation
//===----------------------------------------------------------------------===//

/// Represents a runtime value during interpretation.
/// Supports integer values of arbitrary width and unknown (X) values.
class InterpretedValue {
public:
  /// Default constructor creates an unknown (X) value.
  InterpretedValue() : isUnknown(true), width(1) {}

  /// Construct from an APInt value.
  explicit InterpretedValue(llvm::APInt value)
      : value(std::move(value)), isUnknown(false), width(this->value.getBitWidth()) {}

  /// Construct from a uint64_t value with specified width.
  InterpretedValue(uint64_t val, unsigned w)
      : value(w, val), isUnknown(false), width(w) {}

  /// Create an unknown (X) value of the specified width.
  static InterpretedValue makeX(unsigned w) {
    InterpretedValue v;
    v.width = w;
    v.isUnknown = true;
    return v;
  }

  /// Check if this is an unknown (X) value.
  bool isX() const { return isUnknown; }

  /// Get the bit width.
  unsigned getWidth() const { return width; }

  /// Get the APInt value (only valid if not unknown).
  const llvm::APInt &getAPInt() const { return value; }

  /// Get as uint64_t. For values wider than 64 bits, returns the lower 64 bits.
  uint64_t getUInt64() const {
    if (isUnknown)
      return 0;
    // For values wider than 64 bits, extract the lower 64 bits
    if (value.getBitWidth() > 64)
      return value.trunc(64).getZExtValue();
    return value.getZExtValue();
  }

  /// Convert to SignalValue for use with ProcessScheduler.
  /// SignalValue now supports arbitrary widths using APInt.
  SignalValue toSignalValue() const {
    if (isUnknown)
      return SignalValue::makeX(width);
    return SignalValue(value);
  }

  /// Create from a SignalValue.
  static InterpretedValue fromSignalValue(const SignalValue &sv) {
    if (sv.isUnknown())
      return makeX(sv.getWidth());
    return InterpretedValue(sv.getAPInt());
  }

private:
  llvm::APInt value;
  bool isUnknown;
  unsigned width;
};

//===----------------------------------------------------------------------===//
// CallStackFrame - Saved function call context for suspend/resume
//===----------------------------------------------------------------------===//

/// Represents a saved function call frame for resuming execution after a wait.
/// When a wait occurs inside a nested function call, we need to save the
/// function's execution context so we can resume from the correct point.
struct CallStackFrame {
  /// The MLIR function being executed (mutually exclusive with llvmFuncOp).
  mlir::func::FuncOp funcOp;

  /// The LLVM function being executed (mutually exclusive with funcOp).
  mlir::LLVM::LLVMFuncOp llvmFuncOp;

  /// The block within the function where execution should resume.
  mlir::Block *resumeBlock = nullptr;

  /// The operation iterator within the block where execution should resume.
  mlir::Block::iterator resumeOp;

  /// The call operation that invoked this function (for setting results).
  mlir::Operation *callOp = nullptr;

  /// Arguments passed to the function (for re-entry if needed).
  llvm::SmallVector<InterpretedValue, 4> args;

  /// Call operands for signal mapping (LLVM functions only).
  llvm::SmallVector<mlir::Value, 4> callOperands;

  /// Whether this is an LLVM function frame.
  bool isLLVM() const { return llvmFuncOp != nullptr; }

  CallStackFrame() = default;
  CallStackFrame(mlir::func::FuncOp func, mlir::Block *block,
                 mlir::Block::iterator op, mlir::Operation *call)
      : funcOp(func), resumeBlock(block), resumeOp(op), callOp(call) {}
  CallStackFrame(mlir::LLVM::LLVMFuncOp func, mlir::Block *block,
                 mlir::Block::iterator op, mlir::Operation *call)
      : llvmFuncOp(func), resumeBlock(block), resumeOp(op), callOp(call) {}
};

//===----------------------------------------------------------------------===//
// ProcessExecutionState - Per-process execution state
//===----------------------------------------------------------------------===//

/// Simple memory block for LLVM alloca operations.
/// Stores a contiguous block of bytes that can be read/written.
struct MemoryBlock {
  /// The raw memory storage.
  std::vector<uint8_t> data;

  /// The size of the allocated memory in bytes.
  size_t size = 0;

  /// The element type width in bits (for type tracking).
  unsigned elementBitWidth = 0;

  /// Whether the memory has been initialized.
  bool initialized = false;

  /// If non-zero, the signal ID that this alloca backs (for fork-child
  /// write propagation).  Populated lazily by the drive interpreter.
  SignalId backingSignalId = 0;

  MemoryBlock() = default;
  MemoryBlock(size_t sz, unsigned elemBits)
      : data(sz, 0), size(sz), elementBitWidth(elemBits), initialized(false) {}
};

using InstanceId = uint32_t;

struct InstanceInputMappingEntry {
  mlir::BlockArgument arg;
  mlir::Value value;
  InstanceId instanceId = 0;
};

using InstanceInputMapping = llvm::SmallVector<InstanceInputMappingEntry, 8>;

/// Execution state for an LLHD process or seq.initial block being interpreted.
struct ProcessExecutionState {
  /// The process operation being executed (either llhd.process or seq.initial).
  mlir::Operation *processOrInitialOp = nullptr;

  /// Instance context for this process.
  InstanceId instanceId = 0;

  /// Per-instance input mapping for module block arguments.
  InstanceInputMapping inputMap;

  /// Current basic block being executed.
  mlir::Block *currentBlock = nullptr;

  /// Iterator to the current operation within the block.
  mlir::Block::iterator currentOp;

  /// SSA value map: maps MLIR Values to their interpreted runtime values.
  llvm::DenseMap<mlir::Value, InterpretedValue> valueMap;

  /// For ref-typed block arguments, track the concrete incoming SSA value
  /// selected by the most recent branch transfer. This preserves reference
  /// provenance (e.g., llhd.sig.array_get) across block argument boundaries.
  llvm::DenseMap<mlir::BlockArgument, mlir::Value> refBlockArgSources;

  /// Memory model for LLVM dialect operations.
  /// Maps pointer values to their allocated memory blocks.
  llvm::DenseMap<mlir::Value, MemoryBlock> memoryBlocks;

  /// Counter for generating unique memory addresses.
  uint64_t nextMemoryAddress = 0x1000;

  /// Flag indicating whether the process has halted.
  bool halted = false;

  /// Flag indicating whether the process was killed via process::kill().
  bool killed = false;

  /// Per-process random generator for process::srandom/get_randstate.
  std::mt19937 randomGenerator;

  /// Flag indicating whether the process is waiting.
  bool waiting = false;

  /// Flag indicating this is a seq.initial block (runs once at time 0).
  bool isInitialBlock = false;

  /// The next block to branch to after a wait.
  mlir::Block *destBlock = nullptr;

  /// If true, resume at the current operation (currentOp) instead of at the
  /// beginning of destBlock. Used for deferred llhd.halt and sim.terminate.
  bool resumeAtCurrentOp = false;

  /// Operands to pass to the destination block after a wait.
  llvm::SmallVector<InterpretedValue, 4> destOperands;

  /// Current call depth for tracking recursive function calls.
  /// Used to prevent stack overflow from unbounded recursion.
  size_t callDepth = 0;

  /// Recursion depth counter for function calls. Tracks (funcOp, arg0) pairs
  /// to prevent exponential blowup from DFS over diamond patterns in graphs
  /// (e.g., UVM m_find_successor over the phase DAG). Uses a depth counter
  /// instead of a set to allow legitimate nesting (e.g., UVM factory calls
  /// where constructing one object triggers more factory calls via the same
  /// singleton factory pointer).
  llvm::DenseMap<mlir::Operation *, llvm::DenseMap<uint64_t, unsigned>>
      recursionVisited;

  /// Function result cache for pure functions called repeatedly with the same
  /// arguments. Used to speed up UVM phase graph traversal where functions like
  /// get_schedule, get_domain, get_phase_type, find, m_find_successor, and
  /// m_find_predecessor are called thousands of times with the same args.
  /// Key: (funcOp pointer, hash of argument values).
  /// Value: cached return values.
  /// Invalidated when uvm_phase::add is called (modifies the phase graph).
  llvm::DenseMap<mlir::Operation *,
                 llvm::DenseMap<uint64_t,
                                llvm::SmallVector<InterpretedValue, 2>>>
      funcResultCache;

  /// Number of function result cache hits (for diagnostics).
  size_t funcCacheHits = 0;

  /// Last operation executed by this process (for diagnostics).
  mlir::Operation *lastOp = nullptr;

  /// Total operations executed by this process.
  size_t totalSteps = 0;

  /// Total operations executed inside function bodies (subset of totalSteps).
  size_t funcBodySteps = 0;

  /// Name of the function currently being executed (empty if at process level).
  std::string currentFuncName;

  /// Total operations in this process/initial body (used for step budget).
  size_t opCount = 0;

  /// Cached sensitivity list entries for combinational wait reuse.
  llvm::SmallVector<SensitivityEntry, 4> lastSensitivityEntries;

  /// Cached values for the last sensitivity list.
  llvm::SmallVector<SignalValue, 4> lastSensitivityValues;

  /// Cache of derived wait sensitivities keyed by wait operations.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<SensitivityEntry, 4>>
      waitSensitivityCache;

  /// Process-level fallback sensitivity derived from all ProbeOps in the
  /// process/initial body. Reused across wait ops that need probe fallback.
  llvm::SmallVector<SensitivityEntry, 8> probeSensitivityFallbackCache;
  bool probeSensitivityFallbackValid = false;

  /// Indicates whether the cached sensitivity data is valid.
  bool lastSensitivityValid = false;

  /// True if the last wait included a delay.
  bool lastWaitHadDelay = false;

  /// True if the last wait used edge-specific sensitivity.
  bool lastWaitHasEdge = false;

  /// True if this process is eligible for cache-based skipping.
  bool cacheable = false;

  /// Number of times this process skipped execution via caching.
  uint64_t cacheSkips = 0;

  /// Number of times derived wait sensitivities were reused from cache.
  uint64_t waitSensitivityCacheHits = 0;

  /// Accumulated delay from __moore_delay calls (in femtoseconds).
  /// Multiple sequential __moore_delay calls within a function accumulate here,
  /// and the total is used when the process actually suspends.
  int64_t pendingDelayFs = 0;

  /// Operation to restart from when resuming after a wait_condition with false.
  /// When a wait(condition) is called with a false condition, we need to
  /// restart from a point that allows the condition to be re-evaluated.
  /// This stores the iterator to the first operation in the condition
  /// computation chain.
  mlir::Block::iterator waitConditionRestartOp;

  /// Block containing the restart operation.
  mlir::Block *waitConditionRestartBlock = nullptr;

  /// Values to invalidate when restarting for wait_condition re-evaluation.
  /// These are SSA values that feed into the condition and need to be
  /// recomputed when the condition is re-checked.
  llvm::SmallVector<mlir::Value, 8> waitConditionValuesToInvalidate;

  /// Saved process body position for wait_condition inside function calls.
  /// When wait_condition is inside a function (e.g., phase_hopper::get()),
  /// the process body's currentBlock/currentOp are overwritten with the
  /// function body's restart point. These fields save the process body
  /// position so it can be restored after the function completes.
  mlir::Block *waitConditionSavedBlock = nullptr;
  mlir::Block::iterator waitConditionSavedOp;

  /// Queue object currently associated with a wait(condition) on queue size.
  /// Non-zero when wait_condition observed a __moore_queue_size dependency and
  /// registered queue-backed wakeup state for this process.
  uint64_t waitConditionQueueAddr = 0;

  /// Monotonic token for wait_condition polling callbacks.
  /// Incremented each time a new poll callback is scheduled so stale callbacks
  /// from previous waits do not wake newer wait states.
  uint64_t waitConditionPollToken = 0;

  /// Call stack for resuming execution after a wait inside a function.
  /// When a wait (e.g., sim.fork with blocking join) occurs inside a nested
  /// function call, we push the function's context onto this stack so that
  /// when the process resumes, we can continue from the correct point inside
  /// the function rather than skipping to the next process-level operation.
  llvm::SmallVector<CallStackFrame, 4> callStack;

  /// Optional override for the outermost caller call site used when restoring
  /// process-level position after call-stack resume.
  /// This is populated when pure tail-wrapper frames are elided at suspension
  /// time so resume can still jump back to the original caller correctly.
  mlir::Operation *callStackOutermostCallOp = nullptr;

  /// When a seq_item_pull_port::get interceptor finds an empty FIFO, it saves
  /// the call_indirect operation here so that on resume the innermost call
  /// stack frame can be overridden to re-execute the get call (instead of
  /// skipping past it).
  mlir::Operation *sequencerGetRetryCallOp = nullptr;

  /// Per-callsite state for native fast-path execution of
  /// `*::BaudClkGenerator` helper loops.
  struct BaudClkGeneratorFastPathState {
    bool initialized = false;
    bool primed = false;
    uint64_t countAddr = 0;
    bool countLocalOnly = false;
    int32_t localCount = 0;
    uint64_t clockFieldOffset = 0;
    uint64_t outputFieldOffset = 0;
    SignalId clockSignalId = 0;
    SignalId outputSignalId = 0;
    int32_t divider = 1;
    bool useDelayBatching = false;
    uint64_t delayBatchEdges = 0;
    uint64_t pendingDelayedEdges = 0;
    uint64_t stableEdgeIntervals = 0;
    int64_t edgeIntervalFs = 0;
    int64_t lastEdgeTimeFs = -1;
    bool clockSampleValid = false;
    bool lastClockSample = false;
  };
  llvm::DenseMap<mlir::Operation *, BaudClkGeneratorFastPathState>
      baudClkGeneratorFastPathByCall;

  /// Address to write the result of a pending mailbox get operation.
  /// When a blocking mailbox get is waiting for a message, this stores where
  /// to write the message value when it becomes available.
  uint64_t pendingMailboxGetResultAddr = 0;

  /// The mailbox ID for a pending blocking get operation.
  /// Non-zero when this process is waiting for a mailbox message.
  uint64_t pendingMailboxGetId = 0;

  /// Address to write the result of a pending mailbox peek operation.
  /// When a blocking mailbox peek is waiting for a message, this stores where
  /// to write the message value when it becomes available.
  uint64_t pendingMailboxPeekResultAddr = 0;

  /// The mailbox ID for a pending blocking peek operation.
  /// Non-zero when this process is waiting to peek a mailbox.
  uint64_t pendingMailboxPeekId = 0;

  /// The semaphore ID for a pending blocking get operation.
  /// Non-zero when this process is waiting for semaphore keys.
  uint64_t pendingSemaphoreGetId = 0;

  /// Parent process ID for shared memory in fork/join.
  /// When a child process is created by sim.fork, parent-scope allocas are
  /// accessed through this chain rather than deep-copied.  Only allocas
  /// defined within the fork body are local to the child.
  ProcessId parentProcessId = 0;

  /// Stable region anchor used by native thunk classification/execution.
  /// For fork children this is the fork-branch region, not the owning
  /// llhd.process body. Keeping this stable avoids drifting to transient
  /// function-body regions when currentBlock changes during call-stack resume.
  mlir::Region *nativeThunkRootRegion = nullptr;

  /// Native-thunk resume token used for resumable compiled process patterns.
  uint64_t jitThunkResumeToken = 0;

  ProcessExecutionState() = default;
  explicit ProcessExecutionState(llhd::ProcessOp op)
      : processOrInitialOp(op.getOperation()), currentBlock(nullptr),
        halted(false), waiting(false), isInitialBlock(false) {}
  explicit ProcessExecutionState(llhd::CombinationalOp op)
      : processOrInitialOp(op.getOperation()), currentBlock(nullptr),
        halted(false), waiting(false), isInitialBlock(false) {}
  explicit ProcessExecutionState(seq::InitialOp op)
      : processOrInitialOp(op.getOperation()), currentBlock(nullptr),
        halted(false), waiting(false), isInitialBlock(true) {}

  /// Helper to get the process op (returns null if this is a seq.initial).
  llhd::ProcessOp getProcessOp() const {
    return mlir::dyn_cast_or_null<llhd::ProcessOp>(processOrInitialOp);
  }

  /// Helper to get the initial op (returns null if this is an llhd.process).
  seq::InitialOp getInitialOp() const {
    return mlir::dyn_cast_or_null<seq::InitialOp>(processOrInitialOp);
  }

  /// Helper to get the combinational op (returns null otherwise).
  llhd::CombinationalOp getCombinationalOp() const {
    return mlir::dyn_cast_or_null<llhd::CombinationalOp>(processOrInitialOp);
  }
};

//===----------------------------------------------------------------------===//
// DiscoveredOps - Collected operations from iterative traversal
//===----------------------------------------------------------------------===//

/// Structure to hold operations discovered during a single iterative traversal.
/// This replaces multiple recursive walk() calls with one pass over the IR,
/// avoiding stack overflow on large designs (e.g., 165k+ line UVM testbenches).
struct DiscoveredOps {
  /// hw.instance operations found in the module.
  llvm::SmallVector<hw::InstanceOp, 16> instances;

  /// llhd.sig operations found in the module.
  llvm::SmallVector<llhd::SignalOp, 64> signals;

  /// llhd.output operations found in the module.
  llvm::SmallVector<llhd::OutputOp, 16> outputs;

  /// llhd.process operations found in the module.
  llvm::SmallVector<llhd::ProcessOp, 32> processes;

  /// llhd.combinational operations found in the module.
  llvm::SmallVector<llhd::CombinationalOp, 16> combinationals;

  /// seq.initial operations found in the module.
  llvm::SmallVector<seq::InitialOp, 16> initials;

  /// llhd.drv operations at module level (not inside processes/initials).
  llvm::SmallVector<llhd::DriveOp, 32> moduleDrives;

  /// seq.firreg operations found in the module.
  llvm::SmallVector<seq::FirRegOp, 32> firRegs;

  /// verif.clocked_assert operations found at module level.
  llvm::SmallVector<verif::ClockedAssertOp, 4> clockedAsserts;

  /// Clear all collected operations.
  void clear() {
    instances.clear();
    signals.clear();
    outputs.clear();
    processes.clear();
    combinationals.clear();
    initials.clear();
    moduleDrives.clear();
    firRegs.clear();
    clockedAsserts.clear();
  }
};

/// Structure to hold operations discovered during global initialization.
struct DiscoveredGlobalOps {
  /// LLVM global operations.
  llvm::SmallVector<mlir::LLVM::GlobalOp, 64> globals;

  /// LLVM global constructor operations.
  llvm::SmallVector<mlir::LLVM::GlobalCtorsOp, 4> ctors;

  /// Clear all collected operations.
  void clear() {
    globals.clear();
    ctors.clear();
  }
};

//===----------------------------------------------------------------------===//
// LLHDProcessInterpreter - Main interpreter class
//===----------------------------------------------------------------------===//

/// Interprets LLHD process bodies during simulation.
///
/// This class bridges the gap between the ProcessScheduler infrastructure
/// and LLHD IR by:
/// 1. Registering signals from llhd.sig operations
/// 2. Creating simulation processes for llhd.process operations
/// 3. Interpreting the process body operations step by step
/// 4. Handling llhd.wait by suspending processes
/// 5. Handling llhd.drv by scheduling signal updates
class LLHDProcessInterpreter {
public:
  LLHDProcessInterpreter(ProcessScheduler &scheduler);
  ~LLHDProcessInterpreter();

  /// Initialize the interpreter for a hardware module.
  /// This walks the module to find all signals and processes,
  /// including those in child module instances.
  mlir::LogicalResult initialize(hw::HWModuleOp hwModule);

  /// Finalize initialization by executing global constructors (UVM init).
  /// Must be called AFTER all top modules have been initialized via
  /// initialize(), so that all modules' executeModuleLevelLLVMOps() have
  /// completed (including hdl_top initial blocks that call config_db::set).
  mlir::LogicalResult finalizeInit();

  /// Drive all interface field shadow signals with their current memory values.
  /// Initialize child module instances using pre-discovered operations.
  /// This uses the pre-discovered hw.instance operations and registers
  /// signals/processes from the referenced modules iteratively.
  mlir::LogicalResult initializeChildInstances(const DiscoveredOps &ops,
                                               InstanceId parentInstanceId);

  /// Get the signal ID for an MLIR value (signal reference).
  SignalId getSignalId(mlir::Value signalRef) const;

  /// Get the signal ID for an MLIR value within an explicit instance context.
  SignalId getSignalIdInInstance(mlir::Value signalRef,
                                 InstanceId instanceId) const;

  /// Get the signal name for a signal ID.
  llvm::StringRef getSignalName(SignalId id) const;

  /// Forward-propagate a parent interface signal change to all child copies.
  void forwardPropagateOnSignalChange(SignalId signal,
                                       const SignalValue &value);

  /// Re-evaluate interface tri-state rules for this signal trigger.
  void reevaluateInterfaceTriState(SignalId signal);

  /// Re-evaluate module drives that depend on the given signal.
  /// Called from the signal-change callback so that combinational logic
  /// (continuous assignments) is re-computed when VPI or other external
  /// writes change an input signal.
  void executeModuleDrivesForSignal(SignalId sigId);

  /// Register a top-level hw.output update. When the associated llhd.process
  /// yields new values via interpretWait, executeInstanceOutputUpdates will
  /// immediately re-evaluate and update the output signal. This ensures
  /// correct output values even when inline combinational evaluation fails
  /// for multi-block process CFGs.
  void registerTopLevelOutputUpdate(SignalId signalId, mlir::Value outputValue);

  /// Get the number of registered signals.
  size_t getNumSignals() const { return valueToSignal.size(); }

  /// Get the number of registered processes.
  size_t getNumProcesses() const { return processStates.size(); }

  /// Get total UVM fast-path hits observed in this run.
  uint64_t getUvmFastPathHitsTotal() const {
    uint64_t total = 0;
    for (const auto &entry : uvmFastPathHitCount)
      total += entry.getValue();
    return total;
  }

  /// Get the number of distinct UVM fast-path action keys seen.
  size_t getUvmFastPathActionKeyCount() const {
    return uvmFastPathHitCount.size();
  }

  /// Get the number of UVM fast-path actions promoted by hotness hooks.
  size_t getUvmJitPromotedActionCount() const {
    return uvmJitPromotedFastPaths.size();
  }

  /// Get the configured UVM JIT hotness threshold.
  uint64_t getUvmJitHotThreshold() const { return uvmJitHotThreshold; }

  /// Get the remaining UVM JIT promotion budget.
  int64_t getUvmJitPromotionBudgetRemaining() const {
    return uvmJitPromotionBudget;
  }

  /// Per-process first deopt reason observed by compile-mode dispatch.
  const llvm::DenseMap<uint64_t, std::string> &getJitDeoptReasonByProcess()
      const {
    return jitDeoptReasonByProcess;
  }

  /// Per-process first deopt detail observed by compile-mode dispatch.
  const llvm::DenseMap<uint64_t, std::string> &getJitDeoptDetailByProcess()
      const {
    return jitDeoptDetailByProcess;
  }

  struct JitRuntimeIndirectTargetEntry {
    std::string targetName;
    uint64_t calls = 0;
  };

  struct JitRuntimeIndirectSiteProfile {
    uint64_t siteId = 0;
    std::string owner;
    std::string location;
    uint64_t callsTotal = 0;
    uint64_t unresolvedCalls = 0;
    uint64_t targetSetVersion = 0;
    uint64_t targetSetHash = 0;
    std::vector<JitRuntimeIndirectTargetEntry> targets;
  };

  struct JitRuntimeIndirectSiteGuardSpec {
    mlir::Operation *siteOp = nullptr;
    uint64_t expectedTargetSetVersion = 0;
    uint64_t expectedTargetSetHash = 0;
    uint64_t expectedUnresolvedCalls = 0;
  };

  /// Per-site runtime target-set profile for func.call_indirect dispatch.
  /// Used for guarded JIT specialization triage in compile mode.
  std::vector<JitRuntimeIndirectSiteProfile>
  getJitRuntimeIndirectSiteProfiles() const;

  /// Lookup runtime profile details for a specific call_indirect site.
  std::optional<JitRuntimeIndirectSiteProfile>
  lookupJitRuntimeIndirectSiteProfile(
      mlir::func::CallIndirectOp callOp) const;

  /// Resolve scheduler-registered process name for a process ID.
  std::string getJitDeoptProcessName(ProcessId procId) const;

  /// Compile-mode JIT hot threshold, clamped to at least 1.
  uint64_t getJitCompileHotThreshold() const;

  friend class LLHDProcessInterpreterTest;
  friend struct ScopedInstanceContext;
  friend struct ScopedInputValueMap;

  /// Dump process execution state for diagnostics.
  void dumpProcessStates(llvm::raw_ostream &os) const;

  /// Dump operation execution statistics.
  void dumpOpStats(llvm::raw_ostream &os, size_t topN) const;

  /// Dump per-process execution statistics.
  void dumpProcessStats(llvm::raw_ostream &os, size_t topN) const;

  /// Set the maximum number of operations a process can execute per activation.
  void setMaxProcessSteps(size_t maxSteps) { maxProcessSteps = maxSteps; }

  /// Provide pre-registered port signal mappings (name → SignalId) from the
  /// simulation context.  During initialize(), these are used to populate
  /// valueToSignal for module block arguments that are non-ref-type ports
  /// (e.g., four-state struct-encoded ports).
  void setPortSignalMap(const llvm::StringMap<SignalId> &map) {
    externalPortSignals = &map;
  }

  /// Enable collection of operation execution statistics.
  void setCollectOpStats(bool enable) { collectOpStats = enable; }

  /// Enable or disable compile-mode execution behavior.
  /// When enabled, also initializes the block-level JIT compiler.
  void setCompileModeEnabled(bool enable);

  /// Enable or disable block-level JIT (true native codegen).
  /// Called by setCompileModeEnabled, or directly for testing.
  void setBlockJITEnabled(bool enable);

  /// Enable or disable runtime call_indirect target-set profiling.
  void setJitRuntimeIndirectProfileEnabled(bool enable) {
    jitRuntimeIndirectProfileEnabled = enable;
  }

  /// Provide JIT compile manager for compile-mode thunk/deopt accounting.
  void setJITCompileManager(JITCompileManager *manager) {
    jitCompileManager = manager;
  }

  /// Set a callback to be called when sim.terminate is executed.
  /// The callback receives (success, verbose) parameters.
  void setTerminateCallback(std::function<void(bool, bool)> callback) {
    terminateCallback = std::move(callback);
  }

  /// Check if termination has been requested.
  bool isTerminationRequested() const { return terminationRequested; }

  /// Check if the $finish grace period has expired. Call this from the main
  /// simulation loop periodically. Returns true if the grace period is active
  /// and has expired, meaning we should force termination.
  bool checkFinishGracePeriod() {
    if (!finishGracePeriodActive)
      return false;
    auto elapsed = std::chrono::steady_clock::now() - finishGracePeriodStart;
    if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >=
        kFinishGracePeriodSecs) {
      terminationRequested = true;
      return true;
    }
    return false;
  }

  /// Poll the active $monitor registration and emit output on value changes.
  void pollRegisteredMonitor();

  /// Set a callback to check if abort has been requested (e.g., by timeout).
  void setShouldAbortCallback(std::function<bool()> callback) {
    shouldAbortCallback = std::move(callback);
  }

  /// Set a callback to invoke when abort is triggered.
  void setAbortCallback(std::function<void()> callback) {
    abortCallback = std::move(callback);
  }

  /// Set a callback for $dumpfile to create a VCD file at runtime.
  void setDumpfileCallback(std::function<void(llvm::StringRef)> callback) {
    dumpfileCallback = std::move(callback);
  }

  /// Set a callback for $dumpvars to start VCD tracing at runtime.
  void setDumpvarsCallback(std::function<void()> callback) {
    dumpvarsCallback = std::move(callback);
  }

  /// Set a callback for $dumpoff to pause VCD value-change recording.
  void setDumpoffCallback(std::function<void()> callback) {
    dumpoffCallback = std::move(callback);
  }

  /// Set a callback for $dumpon to resume VCD value-change recording.
  void setDumponCallback(std::function<void()> callback) {
    dumponCallback = std::move(callback);
  }

  /// Apply a DPI/VPI force to a signal and synchronize interpreter force state.
  void applyDpiForce(SignalId sigId, const SignalValue &forcedValue);

  /// Release a DPI/VPI force and restore pre-force value when available.
  void applyDpiRelease(SignalId sigId);

  /// Resolve a hierarchical HDL path (e.g., "top.sig") to a SignalId.
  /// Tries exact match, then leaf-name match, then suffix matches.
  /// Returns 0 if not found.
  SignalId resolveHdlPathToSignalId(llvm::StringRef path) const;

  /// Check if abort has been requested (calls the shouldAbortCallback).
  bool isAbortRequested() const {
    return shouldAbortCallback && shouldAbortCallback();
  }

  /// Get the number of clocked assertion failures observed during simulation.
  size_t getClockedAssertionFailures() const {
    return clockedAssertionFailures;
  }

  /// Get the bit width of a type. Made public for use by helper functions.
  /// Uses a cache for composite types (struct/array) to avoid repeated recursion.
  static unsigned getTypeWidth(mlir::Type type);
  static unsigned getTypeWidthUncached(mlir::Type type);
  /// Determine signal encoding based on the type.
  static SignalEncoding getSignalEncoding(mlir::Type type);

  /// Get the logical (VPI-visible) width of a type, stripping 4-state overhead.
  /// For a hw::StructType with value/unknown fields, returns the value width.
  /// For nested structs (e.g., packed SV structs with 4-state leaf fields),
  /// recursively strips the 4-state encoding from each leaf field.
  static unsigned getLogicalWidth(mlir::Type type);

  /// Evaluate a value for continuous assignments by reading from signal state.
  /// Made public so circt-sim can re-evaluate hw.output operands when VPI
  /// changes input port signals.
  InterpretedValue evaluateContinuousValue(mlir::Value value);

private:
  struct InstanceOutputInfo {
    mlir::Value outputValue;
    InstanceInputMapping inputMap;
    InstanceId instanceId = 0;
  };

  //===--------------------------------------------------------------------===//
  // Iterative Operation Discovery (Stack Overflow Prevention)
  //===--------------------------------------------------------------------===//

  /// Discover all operations in a module using an iterative worklist algorithm.
  /// This replaces multiple recursive walk() calls with a single pass,
  /// preventing stack overflow on large designs (165k+ lines).
  void discoverOpsIteratively(hw::HWModuleOp hwModule, DiscoveredOps &ops);

  /// Discover all global operations using an iterative worklist algorithm.
  /// This replaces recursive walk() calls for global initialization.
  void discoverGlobalOpsIteratively(DiscoveredGlobalOps &ops);

  //===--------------------------------------------------------------------===//
  // Signal Registration
  //===--------------------------------------------------------------------===//

  /// Register all signals from the module using pre-discovered operations.
  mlir::LogicalResult registerSignals(hw::HWModuleOp hwModule,
                                      const DiscoveredOps &ops);

  /// Register a single signal from an llhd.sig operation.
  SignalId registerSignal(llhd::SignalOp sigOp);

  //===--------------------------------------------------------------------===//
  // Process Registration
  //===--------------------------------------------------------------------===//

  /// Register all processes from the module using pre-discovered operations.
  mlir::LogicalResult registerProcesses(const DiscoveredOps &ops);

  /// Register a single process from an llhd.process operation.
  ProcessId registerProcess(llhd::ProcessOp processOp);

  /// Register a single combinational process from an llhd.combinational op.
  ProcessId registerCombinational(llhd::CombinationalOp combinationalOp);

  /// Register a single initial block from a seq.initial operation.
  ProcessId registerInitialBlock(seq::InitialOp initialOp);

  /// Register a module-level drive operation.
  void registerModuleDrive(llhd::DriveOp driveOp, InstanceId instanceId,
                           const InstanceInputMapping &inputMap);

  /// Build the allocaBackingSignal mapping by scanning process yields and
  /// their corresponding module-level drives.
  void buildAllocaToSignalMapping(hw::HWModuleOp hwModule);

  /// Analyze NBA yield chains to detect loop accumulation patterns where
  /// a stale probe is read instead of the accumulated block argument.
  /// Populates nbaYieldBlockArgMap and nbaProbeValues.
  void analyzeNBAYieldChains();

  /// After a memory-backed struct field write, propagate the updated alloca
  /// contents to the backing signal so probes see the new value.
  void propagateAllocaWriteToBackingSignal(
      MemoryBlock *block, uint64_t blockOffset, unsigned parentWidth,
      mlir::Value parentSignal,
      llvm::ArrayRef<llhd::SigStructExtractOp> extractChain,
      ProcessId procId);

  /// Execute module-level drives for a process after it yields.
  void executeModuleDrives(ProcessId procId);

  /// Execute instance output updates that depend on a process result.
  void executeInstanceOutputUpdates(ProcessId procId);

  /// Register combinational processes for static module-level drives.
  /// These drives need to re-execute when their input signals change.
  void registerContinuousAssignments(hw::HWModuleOp hwModule,
                                     InstanceId instanceId,
                                     const InstanceInputMapping &inputMap);

  /// Register seq.firreg operations using pre-discovered operations.
  void registerFirRegs(const DiscoveredOps &ops, InstanceId instanceId,
                       const InstanceInputMapping &inputMap);

  /// Execute a single seq.firreg register update.
  void executeFirReg(seq::FirRegOp regOp, InstanceId instanceId);

  /// Register verif.clocked_assert operations as reactive processes.
  void registerClockedAssertions(const DiscoveredOps &ops,
                                 InstanceId instanceId,
                                 const InstanceInputMapping &inputMap);

  /// Execute a single clocked assertion check (called on clock edge).
  void executeClockedAssertion(verif::ClockedAssertOp assertOp,
                               InstanceId instanceId);

  /// Resolve a signal ID from an arbitrary value.
  SignalId resolveSignalId(mlir::Value value) const;

  /// Get the stored signal value type for a signal ID (nested type).
  mlir::Type getSignalValueType(SignalId sigId) const;


  /// Convert aggregate values between LLVM and HW layout conventions.
  llvm::APInt convertLLVMToHWLayout(llvm::APInt value, mlir::Type llvmType,
                                    mlir::Type hwType) const;
  llvm::APInt convertHWToLLVMLayout(llvm::APInt value, mlir::Type hwType,
                                    mlir::Type llvmType) const;
  /// Fallback aggregate layout conversion when only the HW type is known.
  /// This assumes a packed LLVM-style memory layout (fields/elements laid out
  /// low-to-high in declaration order).
  llvm::APInt convertLLVMToHWLayoutByHWType(llvm::APInt value,
                                            mlir::Type hwType) const;
  llvm::APInt convertHWToLLVMLayoutByHWType(llvm::APInt value,
                                            mlir::Type hwType) const;
  /// Build an encoded unknown value for 4-state types, or return nullopt
  /// if the type cannot represent X/Z via explicit unknown bits.
  std::optional<llvm::APInt> getEncodedUnknownForType(mlir::Type type) const;

  /// Collect signal IDs referenced by a value expression.
  void collectSignalIds(mlir::Value value,
                        llvm::SmallVectorImpl<SignalId> &signals) const;

  /// Look up a mapped input value and its instance context, if present.
  bool lookupInputMapping(mlir::BlockArgument arg, mlir::Value &mappedValue,
                          InstanceId &mappedInstance) const;

  /// Collect process IDs that a value depends on (via llhd.process results).
  void collectProcessIds(mlir::Value value,
                         llvm::SmallVectorImpl<ProcessId> &processIds) const;

  // NOTE: collectSignalIdsFromCombinational was removed and inlined into
  // collectSignalIds to prevent stack overflow on large designs.

  /// Normalize implicit four-state Z drives to high-impedance strengths.
  /// This preserves pullup/open-drain resolution for drive values encoded as Z
  /// ({value=1, unknown=1} in each logical bit).
  void normalizeImplicitZDriveStrength(SignalId signalId,
                                       const InterpretedValue &driveVal,
                                       DriveStrength &strength0,
                                       DriveStrength &strength1) const;

  /// Resolve the interface field signal loaded by a tri-state-backed drive
  /// value, if any. Cached per (drive op, instance) to avoid re-matching hot
  /// drive paths.
  SignalId resolveTriStateDriveSourceFieldSignal(llhd::DriveOp driveOp,
                                                 InstanceId instanceId);

  /// Evaluate tri-state-backed drive values from cond/src/else rule intent
  /// instead of reading potentially bus-mirrored destination field storage.
  /// Returns true when the drive value was derived from a tri-state rule.
  bool tryEvaluateTriStateDestDriveValue(llhd::DriveOp driveOp,
                                         SignalId targetSigId,
                                         InterpretedValue &driveVal);

  /// Execute a single continuous assignment (static module-level drive).
  void executeContinuousAssignment(llhd::DriveOp driveOp);

  /// Schedule a combinational update of an instance output signal.
  void scheduleInstanceOutputUpdate(
      SignalId signalId, mlir::Value outputValue, InstanceId instanceId,
      const InstanceInputMapping *inputMap);

  /// Helper for iterative continuous-value evaluation with cycle detection.
  InterpretedValue evaluateContinuousValueImpl(mlir::Value value);

  /// Evaluate an llhd.combinational op and return its yielded values.
  /// When \p traceThrough is true, probes inside the body use
  /// combSignalDriveMap to trace through combinational expressions rather
  /// than reading potentially stale scheduler signal values.
  bool evaluateCombinationalOp(llhd::CombinationalOp combOp,
                               llvm::SmallVectorImpl<InterpretedValue> &results,
                               bool traceThrough = false);

  //===--------------------------------------------------------------------===//
  // Process Execution
  //===--------------------------------------------------------------------===//

  /// Execute one step of a process (interpret one operation).
  /// Returns true if the process should continue, false if suspended/halted.
  bool executeStep(ProcessId procId);

  /// Return the active step limit for \p procId in the current phase.
  size_t getEffectiveMaxProcessSteps(ProcessId procId) const;

  /// Execute a process until it suspends or halts.
  void executeProcess(ProcessId procId);

  /// Resume a process after a wait condition is satisfied.
  void resumeProcess(ProcessId procId);

  /// Initialize the stable native-thunk region anchor for a process state.
  void initializeNativeThunkRootRegion(ProcessExecutionState &state);

  /// Resolve the stable region used by native thunk policy/execution.
  const mlir::Region *
  resolveNativeThunkProcessRegion(const ProcessExecutionState &state) const;
  mlir::Region *resolveNativeThunkProcessRegion(ProcessExecutionState &state);

  /// Compile-mode process thunk installation outcome.
  enum class ProcessThunkInstallResult {
    Installed,
    MissingThunk,
    UnsupportedOperation,
  };

  struct PeriodicToggleClockThunkSpec {
    bool hasInitialDrive = false;
    llhd::DriveOp initialDriveOp;
    mlir::Value intToTimeResult;
    llhd::WaitOp waitOp;
    mlir::Block *waitDestBlock = nullptr;
    llhd::ProbeOp probeOp;
    mlir::Operation *toggleOp = nullptr;
    llhd::DriveOp toggleDriveOp;
    uint64_t delayFs = 0;

    /// Pre-resolved signal ID for direct memory access (bypasses interpreter).
    SignalId signalId = 0;
    /// Bit width of the signal.
    uint32_t signalWidth = 0;
    /// XOR constant for the toggle operation.
    uint64_t toggleXorConstant = 0;
    /// Drive delay components (from the drive op, NOT the wait op).
    uint64_t driveRealFs = 0;
    uint32_t driveDelta = 0;
    uint32_t driveEpsilon = 0;
    /// Whether the native fast path is available (all fields resolved).
    bool nativePathAvailable = false;
    /// Cached Process pointer for O(1) scheduleProcessDirect calls.
    Process *cachedProcess = nullptr;
  };

  enum class DirectProcessFastPathKind : uint8_t {
    None = 0,
    PeriodicToggleClock = 1u << 0,
    ResumableWaitSelfLoop = 1u << 1,
    JITCompiledBlock = 1u << 2,
  };

  /// Try a cached direct fast path for common hot process loop shapes.
  /// This bypasses compile-budgeted thunk installation for known-safe loops.
  bool tryExecuteDirectProcessFastPath(ProcessId procId,
                                       ProcessExecutionState &state);

  /// Attempt to compile/install a native thunk for this process.
  ProcessThunkInstallResult tryInstallProcessThunk(ProcessId procId,
                                                   ProcessExecutionState &state,
                                                   std::string *deoptDetail);

  /// Return true when this process is eligible for the initial native thunk.
  bool isTrivialNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process executes a one-block straight-line body
  /// ending in `llhd.halt` or `sim.fork.terminator`.
  bool isSingleBlockTerminatingNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process executes a forward-only multiblock body
  /// where each block has safe preludes and terminates with either
  /// `cf.br`/`cf.cond_br` or `llhd.halt`/`sim.fork.terminator`.
  bool isMultiBlockTerminatingNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process matches one-block combinational
  /// `... -> llhd.yield` execution that can be thunk-dispatched.
  bool isCombinationalNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process matches a resumable self-looping wait body:
  /// optional entry `cf.br` into a loop block with safe preludes ending in
  /// `llhd.wait` whose destination is the same loop block.
  bool isResumableWaitSelfLoopNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process/fork region is a resumable multiblock wait
  /// state machine: all terminators are branch/cond_br/wait/(optional)
  /// terminal halt, with at least one suspend source (`llhd.wait`,
  /// `__moore_wait_condition`, or `__moore_delay`).
  bool isResumableMultiblockWaitNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  /// Return true when the process matches
  /// wait(delay|observed)->(optional print)->halt.
  bool isResumableWaitThenHaltNativeThunkCandidate(
      ProcessId procId, const ProcessExecutionState &state,
      llvm::SmallVectorImpl<JitRuntimeIndirectSiteGuardSpec>
          *profileGuardSpecs = nullptr) const;

  enum class CallStackResumeResult {
    NoFrames,
    Completed,
    Suspended,
    Failed,
  };

  /// Resume saved function call frames for processes that suspended inside
  /// nested calls. Returns the resume outcome for caller-side dispatch.
  CallStackResumeResult resumeSavedCallStackFrames(
      ProcessId procId, ProcessExecutionState &state);

  /// Execute the initial native thunk body for trivial terminating processes.
  void executeTrivialNativeThunk(ProcessId procId,
                                 ProcessThunkExecutionState &thunkState);

  bool executeResumableWaitThenHaltNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool executeResumableWaitSelfLoopNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool executeResumableMultiblockWaitNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool executeSingleBlockTerminatingNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool executeMultiBlockTerminatingNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool executeCombinationalNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  bool tryBuildPeriodicToggleClockThunkSpec(
      const ProcessExecutionState &state,
      PeriodicToggleClockThunkSpec &spec) const;

  bool executePeriodicToggleClockNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  /// Execute a JIT-compiled hot block for a process.
  /// Returns true if the JIT block handled execution (even on deopt).
  bool executeJITCompiledBlockNativeThunk(
      ProcessId procId, ProcessExecutionState &state,
      ProcessThunkExecutionState &thunkState);

  /// Emit a concise unsupported-shape detail for deopt telemetry.
  std::string
  getUnsupportedThunkDeoptDetail(const ProcessExecutionState &state) const;

  struct JITDeoptStateSnapshot {
    mlir::Block *currentBlock = nullptr;
    mlir::Block::iterator currentOp;
    bool halted = false;
    bool waiting = false;
    mlir::Block *destBlock = nullptr;
    bool resumeAtCurrentOp = false;
    llvm::SmallVector<InterpretedValue, 4> destOperands;
    llvm::SmallVector<CallStackFrame, 4> callStack;
    mlir::Operation *callStackOutermostCallOp = nullptr;
    uint64_t jitThunkResumeToken = 0;
  };

  bool snapshotJITDeoptState(ProcessId procId, JITDeoptStateSnapshot &snapshot);
  bool restoreJITDeoptState(ProcessId procId,
                            const JITDeoptStateSnapshot &snapshot);

  //===--------------------------------------------------------------------===//
  // Time Conversion
  //===--------------------------------------------------------------------===//

  /// Convert an LLHD TimeAttr to SimTime.
  static SimTime convertTime(llhd::TimeAttr timeAttr);

  /// Convert an LLHD time value (SSA value) to SimTime.
  SimTime convertTimeValue(ProcessId procId, mlir::Value timeValue);

  //===--------------------------------------------------------------------===//
  // Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret an llhd.prb (probe) operation.
  mlir::LogicalResult interpretProbe(ProcessId procId, llhd::ProbeOp probeOp);

  /// Interpret an llhd.drv (drive) operation.
  mlir::LogicalResult interpretDrive(ProcessId procId, llhd::DriveOp driveOp);

  /// Interpret an llhd.wait operation.
  mlir::LogicalResult interpretWait(ProcessId procId, llhd::WaitOp waitOp);

  /// Interpret an llhd.halt operation.
  mlir::LogicalResult interpretHalt(ProcessId procId, llhd::HaltOp haltOp);

  /// Interpret an llhd.constant_time operation.
  mlir::LogicalResult interpretConstantTime(ProcessId procId,
                                            llhd::ConstantTimeOp timeOp);

  /// Interpret a general operation (dispatches to specific handlers).
  mlir::LogicalResult interpretOperation(ProcessId procId,
                                         mlir::Operation *op);

  //===--------------------------------------------------------------------===//
  // SCF Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret an scf.if operation.
  mlir::LogicalResult interpretSCFIf(ProcessId procId, mlir::scf::IfOp ifOp);

  /// Interpret an scf.for operation.
  mlir::LogicalResult interpretSCFFor(ProcessId procId, mlir::scf::ForOp forOp);

  /// Interpret an scf.while operation.
  mlir::LogicalResult interpretSCFWhile(ProcessId procId,
                                         mlir::scf::WhileOp whileOp);

  /// Interpret the condition region of an scf.while operation.
  mlir::LogicalResult
  interpretWhileCondition(ProcessId procId, mlir::Region &region,
                          llvm::ArrayRef<InterpretedValue> args,
                          llvm::SmallVectorImpl<InterpretedValue> &results);

  /// Interpret a region (used for scf.if/for/while bodies).
  mlir::LogicalResult
  interpretRegion(ProcessId procId, mlir::Region &region,
                  llvm::ArrayRef<InterpretedValue> args,
                  llvm::SmallVectorImpl<InterpretedValue> &results);

  //===--------------------------------------------------------------------===//
  // Func Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret a func.call operation.
  mlir::LogicalResult interpretFuncCall(ProcessId procId,
                                         mlir::func::CallOp callOp);

  /// Interpret a func.call_indirect operation.
  mlir::LogicalResult
  interpretFuncCallIndirect(ProcessId procId,
                            mlir::func::CallIndirectOp callIndirectOp);

  /// Handle UVM-focused fast-paths for func.call sites.
  /// Returns true when handled and results (if any) are already set.
  bool handleUvmFuncCallFastPath(ProcessId procId, mlir::func::CallOp callOp,
                                 llvm::StringRef calleeName);

  /// Handle native fast-path execution for `*::BaudClkGenerator` helper loops.
  /// Returns true when handled (including suspended wait/retry flow).
  bool handleBaudClkGeneratorFastPath(
      ProcessId procId, mlir::func::CallOp callOp, mlir::func::FuncOp funcOp,
      llvm::ArrayRef<InterpretedValue> args, llvm::StringRef calleeName);

  /// Handle UVM-focused fast-paths for func.call_indirect sites.
  /// Returns true when handled and results (if any) are already set.
  bool handleUvmCallIndirectFastPath(ProcessId procId,
                                     mlir::func::CallIndirectOp callIndirectOp,
                                     llvm::StringRef calleeName);

  /// Shared helper for wait_for_self_and_siblings_to_drop fast-path handling.
  /// Returns true when the call has been handled (including suspended polls).
  bool handleUvmWaitForSelfAndSiblingsToDrop(ProcessId procId,
                                             uint64_t phaseAddr,
                                             mlir::Operation *callOp);

  /// Raise/drop helpers used by UVM objection fast-paths.
  /// `dropPhaseObjection` also wakes any process waiters that are blocked on
  /// the handle reaching zero.
  void raisePhaseObjection(int64_t handle, int64_t count);
  void dropPhaseObjection(int64_t handle, int64_t count);

  /// Register a process as waiting for an objection handle to reach zero.
  void enqueueObjectionZeroWaiter(int64_t handle, ProcessId procId,
                                  mlir::Operation *retryOp);
  /// Remove a process from any objection-zero wait list.
  void removeObjectionZeroWaiter(ProcessId procId);
  /// Wake zero-waiters if the objection count has reached zero.
  void wakeObjectionZeroWaitersIfReady(int64_t handle);

  /// Register/remove wait(condition) queue waiters keyed by queue object.
  void enqueueQueueNotEmptyWaiter(uint64_t queueAddr, ProcessId procId,
                                  mlir::Operation *retryOp);
  void removeQueueNotEmptyWaiter(ProcessId procId);
  /// Wake queue-backed wait(condition) waiters for a queue that became non-empty.
  void wakeQueueNotEmptyWaitersIfReady(uint64_t queueAddr);

  /// Handle UVM-focused fast-paths at func body entry.
  /// Returns true when handled; caller should skip normal function execution.
  bool handleUvmFuncBodyFastPath(
      ProcessId procId, mlir::func::FuncOp funcOp,
      llvm::ArrayRef<InterpretedValue> args,
      llvm::SmallVectorImpl<InterpretedValue> &results,
      mlir::Operation *callOp);

  /// Interpret a function body.
  /// @param procId The process ID executing this function.
  /// @param funcOp The function to execute.
  /// @param args Arguments to pass to the function.
  /// @param results Output: return values from the function.
  /// @param callOp The call operation that invoked this function (for saving
  ///               in call stack frames when waiting).
  /// @param resumeBlock If not null, resume execution from this block instead
  ///                    of the entry block.
  /// @param resumeOp If resumeBlock is set, resume from this operation within
  ///                 the block.
  mlir::LogicalResult
  interpretFuncBody(ProcessId procId, mlir::func::FuncOp funcOp,
                    llvm::ArrayRef<InterpretedValue> args,
                    llvm::SmallVectorImpl<InterpretedValue> &results,
                    mlir::Operation *callOp = nullptr,
                    mlir::Block *resumeBlock = nullptr,
                    mlir::Block::iterator resumeOp = {});

  /// Guard accidental re-entry to UVM run_test call paths.
  /// Returns failure when the single-entry policy is enabled and violated.
  mlir::LogicalResult checkUvmRunTestEntry(ProcessId procId,
                                           llvm::StringRef calleeName);

  //===--------------------------------------------------------------------===//
  // Sim Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret a sim.proc.print operation.
  mlir::LogicalResult interpretProcPrint(ProcessId procId,
                                          sim::PrintFormattedProcOp printOp);

  /// Interpret a sim.terminate operation.
  mlir::LogicalResult interpretTerminate(ProcessId procId,
                                          sim::TerminateOp terminateOp);

  /// Poll execute_phase monitor-fork completion without re-executing sim.fork.
  void pollExecutePhaseMonitorFork(ProcessId procId, uint64_t phaseAddr,
                                   uint64_t pollToken);
  void scheduleExecutePhaseMonitorForkPoll(ProcessId procId,
                                           uint64_t phaseAddr,
                                           uint64_t pollToken,
                                           int64_t objectionCount);
  void pollJoinNoneDisableForkResume(ProcessId procId, ForkId forkId,
                                     uint64_t token);
  void maybeTraceFilteredCall(ProcessId procId, llvm::StringRef callKind,
                              llvm::StringRef calleeName, int64_t nowFs,
                              uint64_t deltaStep);
  void maybeTraceCallIndirectSiteCacheHit(int64_t methodIndex) const;
  void maybeTraceCallIndirectSiteCacheStore(bool hasStaticMethodIndex,
                                            int64_t methodIndex) const;
  void maybeTraceInterfaceSensitivityBegin(ProcessId procId,
                                           llvm::StringRef processName,
                                           size_t sourceCount) const;
  void maybeTraceInterfaceSensitivitySource(SignalId sourceSignalId,
                                            size_t fieldCount) const;
  void maybeTraceInterfaceSensitivityAddedField(SignalId fieldSignalId) const;
  void maybeTraceInterfaceFieldShadowScanSummary(
      size_t topLevelSignalCount, size_t instanceSignalMapCount,
      size_t childCopyPairCount) const;
  void maybeTraceInterfaceParentGepStructName(
      llvm::StringRef structName) const;
  void maybeTraceInterfaceParentScanResult(
      SignalId signalId, unsigned userCount, unsigned probeCount,
      unsigned gepCount, bool foundInterfaceStruct,
      uint64_t mallocAddress) const;
  void maybeTraceChildInstanceFound(llvm::StringRef instanceName,
                                    llvm::StringRef moduleName) const;
  void maybeTraceChildInstanceMissingRootModule() const;
  void maybeTraceChildInstanceMissingModule(llvm::StringRef moduleName) const;
  void maybeTraceRegisteredChildSignal(llvm::StringRef hierarchicalName,
                                       SignalId signalId) const;
  void maybeTraceRegisteredChildProcess(llvm::StringRef processName,
                                        ProcessId processId) const;
  void maybeTraceRegisteredChildCombinational(
      llvm::StringRef combinationalName, ProcessId processId,
      size_t sensitivityCount) const;
  void maybeTraceRegisteredChildInitialBlock(llvm::StringRef initialName,
                                             ProcessId processId) const;
  void maybeTraceChildInputMapped(llvm::StringRef portName,
                                  SignalId signalId) const;
  void maybeTraceChildInstanceOutputCountMismatch(
      llvm::StringRef instanceName, unsigned resultCount,
      unsigned outputCount) const;
  void maybeTraceInitializationRegistrationSummary(
      size_t signalCount, size_t processCount) const;
  void maybeTraceIterativeDiscoverySummary(
      size_t instanceCount, size_t signalCount, size_t outputCount,
      size_t processCount, size_t combinationalCount, size_t initialCount,
      size_t moduleDriveCount, size_t firRegCount,
      size_t clockedAssertCount) const;
  void maybeTraceRegisteredPortSignal(llvm::StringRef portName,
                                      SignalId signalId,
                                      unsigned width) const;
  void maybeTraceMappedExternalPortSignal(llvm::StringRef portName,
                                          SignalId signalId) const;
  void maybeTraceRegisteredOutputSignal(llvm::StringRef outputName,
                                        SignalId signalId,
                                        unsigned width) const;
  void maybeTraceRegisteredSignalInitialValue(
      const llvm::APInt &initialValue) const;
  void maybeTraceRegisteredSignal(llvm::StringRef signalName,
                                  SignalId signalId,
                                  unsigned width) const;
  void maybeTraceExportSignalsBegin(size_t signalCount) const;
  void maybeTraceExportedSignal(llvm::StringRef hierarchicalPath,
                                SignalId signalId,
                                uint32_t width) const;
  void maybeTraceSignalRegistryEntryCount(size_t entryCount) const;
  void maybeTraceRegistryAccessorsConfigured(bool connected) const;
  void maybeTraceFoundProcessOp(unsigned resultCount) const;
  void maybeTraceRegisteredTopProcess(llvm::StringRef processName,
                                      ProcessId processId) const;
  void maybeTraceRegisteredTopCombinationalProcess(
      llvm::StringRef processName, ProcessId processId,
      size_t sensitivityCount) const;
  void maybeTraceRegisteredTopInitialBlock(llvm::StringRef processName,
                                           ProcessId processId) const;
  void traceInterfaceSignalOverwrite(uint64_t fieldAddress,
                                     SignalId oldSignalId,
                                     SignalId newSignalId,
                                     llvm::StringRef newSignalName) const;
  void maybeTraceMultiDriverPostNbaConditionalSignals(
      size_t pendingFirRegCount) const;
  void maybeTraceMultiDriverPostNbaApply(
      SignalId signalId, const SignalValue &driveValue) const;
  void maybeTraceMultiDriverPostNbaSkipFirReg(SignalId signalId) const;
  void maybeTraceMultiDriverSuppressedUnconditional(SignalId signalId) const;
  void maybeTraceMultiDriverConditionalEnable(
      SignalId signalId, const InterpretedValue &enableValue) const;
  void maybeTraceMultiDriverStoredConditional(SignalId signalId,
                                              const SignalValue &driveValue) const;
  void maybeTraceExecuteModuleDrives(ProcessId procId) const;
  void maybeTraceModuleDriveTrigger(SignalId triggerSignalId,
                                    SignalId destinationSignalId,
                                    ProcessId processId) const;
  void maybeTraceInterfacePropagationSource(
      SignalId sourceSignalId, const InterpretedValue &sourceValue,
      size_t fanoutCount) const;
  void maybeTraceInterfacePropagationChild(
      SignalId childSignalId, const InterpretedValue &childValue) const;
  void maybeTraceInterfaceCopyPairLink(SignalId parentSignalId,
                                       uint64_t parentAddress,
                                       SignalId childSignalId,
                                       uint64_t childAddress) const;
  void maybeTraceInterfaceDeferredSameInterfaceLink(
      SignalId sourceSignalId, SignalId destinationSignalId) const;
  void maybeTraceInterfaceCopyPairSummary(size_t totalPairs,
                                          unsigned resolvedPairs,
                                          unsigned unresolvedSourceCount,
                                          unsigned unresolvedDestCount,
                                          unsigned resolvedDeferredCount,
                                          size_t childToParentCount,
                                          size_t propagationCount) const;
  void maybeTraceInterfaceSignalCopyLink(SignalId sourceSignalId,
                                         SignalId destinationSignalId,
                                         uint64_t destinationAddress) const;
  void maybeTraceInterfaceSignalCopySummary(size_t totalPairs,
                                            unsigned resolvedPairs,
                                            unsigned unresolvedDestCount) const;
  void maybeTraceInterfaceFieldSignalDumpHeader(size_t totalEntries) const;
  void maybeTraceInterfaceFieldSignalDumpEntry(uint64_t address,
                                               SignalId signalId) const;
  void maybeTraceInterfaceAutoLinkSignalDumpHeader(
      size_t totalInterfaces) const;
  void maybeTraceInterfaceAutoLinkSignalDumpEntry(
      SignalId interfaceSignalId,
      llvm::ArrayRef<SignalId> fieldSignalIds) const;
  void maybeTraceInterfaceParentSignals(
      llvm::ArrayRef<SignalId> parentSignalIds) const;
  void maybeTraceInterfaceInstanceAwareLink(
      SignalId childSignalId, SignalId parentSignalId) const;
  void maybeTraceInterfaceFieldPropagationMap() const;
  void maybeTraceInterfaceAutoLinkMatch(unsigned bestMatchCount,
                                        SignalId childInterfaceSignalId,
                                        SignalId parentInterfaceSignalId) const;
  void maybeTraceInterfaceAutoLinkTotal(unsigned autoLinkedCount) const;
  void maybeTraceInterfaceIntraLinkDetection(size_t reverseTargetCount,
                                             size_t interfaceGroupCount) const;
  void maybeTraceInterfaceIntraLinkReverseTarget(SignalId signalId,
                                                 size_t childCount) const;
  void maybeTraceInterfaceIntraLinkBlock(size_t fieldCount,
                                         size_t publicCount,
                                         size_t danglingCount) const;
  void maybeTraceInterfaceIntraLinkChildBlocks(size_t childBlockCount) const;
  void maybeTraceInterfaceIntraLinkMatch(SignalId danglingSignalId,
                                         SignalId publicSignalId,
                                         size_t publicChildCount) const;
  void maybeTraceInterfaceIntraLinkTotal(unsigned intraLinkCount) const;
  void maybeTraceInterfaceTriStateCandidateInstall(
      SignalId conditionSignalId, SignalId sourceSignalId,
      SignalId destinationSignalId, unsigned conditionBitIndex,
      const InterpretedValue &elseValue) const;
  void maybeTraceInterfaceTriStateCandidateSummary(
      size_t candidateCount, unsigned installedCount, unsigned unresolvedCount,
      size_t totalRuleCount) const;
  void maybeTraceInterfaceTriStateTrigger(SignalId sourceSignalId,
                                          size_t ruleCount) const;
  void maybeTraceInterfaceTriStateRule(
      unsigned ruleIndex, SignalId conditionSignalId, unsigned conditionBitIndex,
      const InterpretedValue &conditionValue, bool conditionTrue,
      SignalId sourceSignalId, const InterpretedValue &sourceValue,
      SignalId destinationSignalId, const InterpretedValue &destinationBefore,
      const InterpretedValue &selectedValue, bool selectedIsExplicitHighZ) const;
  void maybeTraceCondBranch(ProcessId procId,
                            const InterpretedValue &conditionValue,
                            bool tookTrueBranch, mlir::Value conditionSsaValue);
  void maybeTraceInstanceOutput(SignalId signalId,
                                llvm::ArrayRef<SignalId> sourceSignals,
                                size_t processCount) const;
  void maybeTraceInstanceOutputDependencySignals(
      SignalId signalId, llvm::ArrayRef<SignalId> sourceSignals) const;
  void maybeTraceInstanceOutputUpdate(SignalId signalId,
                                      const InterpretedValue &value) const;
  void maybeTraceCombTraceThroughHit(SignalId signalId,
                                     const InterpretedValue &driveValue) const;
  void maybeTraceCombTraceThroughMiss(SignalId signalId, bool inCombMap,
                                      bool visited) const;
  void maybeTraceContinuousFallback(mlir::Value value,
                                    unsigned invalidatedCount) const;
  void maybeTraceDriveSchedule(SignalId signalId,
                               const InterpretedValue &driveValue,
                               const SimTime &currentTime,
                               const SimTime &targetTime,
                               const SimTime &delay) const;
  void maybeTraceDriveFailure(ProcessId procId, llhd::DriveOp driveOp,
                              llvm::StringRef reason,
                              mlir::Value signalValue) const;
  void maybeTraceRefArgResolveFailure(ProcessId procId,
                                      mlir::Value unresolvedValue,
                                      bool hasBlockArgSource,
                                      mlir::Value blockArgSourceValue,
                                      SignalId blockArgSourceSignal) const;
  void maybeTraceI3CFieldDriveSignalStruct(
      ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef fieldName,
      unsigned bitOffset, unsigned fieldWidth, SignalId parentSignalId,
      const InterpretedValue &driveValue) const;
  void maybeTraceI3CFieldDriveMemStruct(
      ProcessId procId, llhd::DriveOp driveOp, llvm::StringRef fieldName,
      int64_t index, unsigned bitOffset, int64_t fieldWidth,
      const InterpretedValue &driveValue) const;
  void maybeTraceI3CFieldDriveMem(ProcessId procId, llhd::DriveOp driveOp,
                                  uint64_t index, uint64_t elementOffset,
                                  const InterpretedValue &driveValue) const;
  void maybeTraceI3CFieldDrive(ProcessId procId, llhd::DriveOp driveOp,
                               llvm::StringRef fieldName, uint64_t index,
                               unsigned bitOffset,
                               const InterpretedValue &driveValue) const;
  void maybeTraceArrayDriveRemap(ProcessId procId, mlir::Value originalSignal,
                                 mlir::Value remappedSignal);
  void maybeTraceArrayDriveSchedule(SignalId parentSignalId, uint64_t index,
                                    unsigned bitOffset, unsigned elementWidth,
                                    const llvm::APInt &elementValue,
                                    bool driveValueIsX, unsigned parentWidth,
                                    unsigned resultWidth,
                                    const SimTime &delay) const;
  void maybeTraceI3CAddressBitDrive(llhd::DriveOp driveOp,
                                    unsigned bitOffset,
                                    const InterpretedValue &driveValue) const;
  void maybeTraceI3CRefCast(ProcessId procId, bool resolved, uint64_t address,
                            unsigned bitOffset, mlir::Value inputValue,
                            mlir::Value outputValue) const;
  void maybeTraceI3CCastLayout(ProcessId procId, mlir::Type outputType,
                               const llvm::APInt &inputBits,
                               const llvm::APInt &convertedBits) const;
  void maybeTraceI3CConfigHandleGet(llvm::StringRef callee, llvm::StringRef key,
                                    uint64_t ptrPayload, uint64_t outRefAddr,
                                    llvm::StringRef fieldName) const;
  void maybeTraceI3CConfigHandleSet(llvm::StringRef callee, llvm::StringRef key,
                                    uint64_t ptrPayload,
                                    llvm::StringRef fieldName) const;
  void maybeTraceI3CHandleCall(ProcessId procId, llvm::StringRef calleeName,
                               llvm::ArrayRef<InterpretedValue> args) const;
  void maybeTraceI3CCallStackSave(ProcessId procId, llvm::StringRef funcName,
                                  size_t blockOpCount,
                                  size_t savedFrameCount) const;
  void maybeTraceI3CToClassArgs(ProcessId procId, llvm::StringRef calleeName,
                                llvm::ArrayRef<InterpretedValue> args) const;
  void maybeTraceConfigDbFuncCallGetBegin(llvm::StringRef callee,
                                          llvm::StringRef key,
                                          llvm::StringRef instName,
                                          llvm::StringRef fieldName,
                                          size_t entryCount) const;
  void maybeTraceConfigDbFuncCallGetHit(llvm::StringRef key,
                                        size_t byteCount) const;
  void maybeTraceConfigDbFuncCallGetMiss(llvm::StringRef key) const;
  void maybeTraceConfigDbFuncCallSet(llvm::StringRef callee,
                                     llvm::StringRef key, unsigned valueBytes,
                                     size_t entryCount) const;
  void maybeTraceSequencerFuncCallSize(llvm::StringRef calleeName,
                                       uint64_t selfAddr, int32_t count,
                                       size_t totalConnections) const;
  void maybeTraceSequencerFuncCallGet(llvm::StringRef calleeName,
                                      uint64_t portAddr,
                                      size_t fifoMapCount) const;
  void maybeTraceSequencerFuncCallPop(uint64_t itemAddr, uint64_t portAddr,
                                      uint64_t queueAddr,
                                      bool fromFallbackSearch) const;
  void maybeTraceSequencerFuncCallTryMiss(uint64_t portAddr,
                                          uint64_t refAddr) const;
  void maybeTraceSequencerFuncCallWait(uint64_t portAddr,
                                       uint64_t queueAddr) const;
  void maybeTraceSequencerFuncCallItemDoneMiss(uint64_t callerAddr) const;
  void maybeTraceAnalysisWriteFuncCallBegin(llvm::StringRef calleeName,
                                            uint64_t portAddr,
                                            bool inConnectionMap) const;
  void maybeTraceAnalysisWriteFuncCallTerminals(size_t terminalCount) const;
  void maybeTraceAnalysisWriteFuncCallMissingVtableHeader(
      uint64_t impAddr) const;
  void maybeTraceAnalysisWriteFuncCallMissingAddressToGlobal(
      uint64_t vtableAddr) const;
  void maybeTraceAnalysisWriteFuncCallMissingAddressToFunction(
      uint64_t writeFuncAddr) const;
  void maybeTraceAnalysisWriteFuncCallMissingModuleFunction(
      llvm::StringRef functionName) const;
  void maybeTraceAnalysisWriteFuncCallDispatch(llvm::StringRef functionName) const;
  void maybeTraceFuncCacheSharedHit(llvm::StringRef functionName,
                                    uint64_t argHash) const;
  void maybeTraceFuncCacheSharedStore(llvm::StringRef functionName,
                                      uint64_t argHash) const;
  void maybeTracePhaseOrderProcessPhase(uint64_t phaseAddr, uint64_t impAddr,
                                        std::optional<int> order) const;
  void maybeTracePhaseOrderProcessPhaseWaitPred(uint64_t phaseAddr, int order,
                                                uint64_t predImpAddr) const;
  void maybeTracePhaseOrderProcessPhaseWaitUnknownImp(uint64_t phaseAddr,
                                                      uint64_t impAddr) const;
  void maybeTracePhaseOrderFinishPhase(uint64_t phaseAddr, uint64_t impAddr,
                                       int order, size_t waiterCount) const;
  void maybeTracePhaseOrderWakeWaiter(ProcessId waiterProcId,
                                      uint64_t impAddr) const;
  void maybeTracePhaseOrderExecutePhase(uint64_t phaseAddr,
                                        llvm::StringRef phaseName) const;
  void maybeTraceMailboxTryPut(ProcessId procId, uint64_t mboxId,
                               uint64_t message, bool success) const;
  void maybeTraceMailboxWakeGetByTryPut(ProcessId procId, ProcessId waiterProcId,
                                        uint64_t mboxId, uint64_t outAddr,
                                        uint64_t message) const;
  void maybeTraceMailboxGet(ProcessId procId, uint64_t mboxId,
                            llvm::StringRef mode, uint64_t outAddr,
                            std::optional<uint64_t> message) const;
  void maybeTraceRandClassSrandom(uint64_t objAddr, uint32_t seed) const;
  void maybeTraceRandBasic(uint64_t objAddr, uint64_t size, int32_t rc) const;
  void maybeTraceRandBytes(uint64_t dataAddr, int64_t size, int32_t rc,
                           int firstByte) const;
  void maybeTraceRandRange(uint64_t objAddr, int64_t minVal, int64_t maxVal,
                           int64_t result) const;
  void maybeTraceTailWrapperSuspendElide(ProcessId procId,
                                         llvm::StringRef wrapperName,
                                         llvm::StringRef calleeName,
                                         bool monitorDeserializer,
                                         bool driveSample) const;
  void maybeTraceOnDemandLoadSignal(uint64_t addr, SignalId signalId,
                                    llvm::StringRef signalName) const;
  void maybeTraceOnDemandLoadNoSignal(uint64_t addr) const;
  void maybeTraceStructInjectX(llvm::StringRef fieldName, bool structIsX,
                               bool newValueIsX, unsigned totalWidth) const;
  void maybeTraceAhbTxnFieldWrite(ProcessId procId, llvm::StringRef funcName,
                                  uint64_t txnAddr, llvm::StringRef fieldName,
                                  std::optional<uint64_t> index,
                                  unsigned bitOffset, unsigned width,
                                  const InterpretedValue &value,
                                  const SimTime &now) const;
  void maybeTraceFreadSignalPath(SignalId signalId, int32_t elemWidth,
                                 int32_t elemStorageBytes, int32_t numElements,
                                 uint64_t totalBytes) const;
  void maybeTraceFreadSignalWidth(uint64_t bitWidth, size_t rawBytes,
                                  int32_t result) const;
  void maybeTraceFreadPointerPath(int32_t elemWidth, int32_t elemStorageBytes,
                                  int32_t numElements, uint64_t totalBytes) const;
  void maybeTraceFuncProgress(ProcessId procId, size_t funcBodySteps,
                              size_t totalSteps, llvm::StringRef funcName,
                              size_t callDepth,
                              llvm::StringRef opName) const;
  void maybeTraceProcessStepOverflowInFunc(ProcessId procId,
                                           size_t effectiveMaxProcessSteps,
                                           size_t totalSteps,
                                           llvm::StringRef funcName,
                                           bool isLLVMFunction) const;
  void maybeTraceProcessActivationStepLimitExceeded(
      ProcessId procId, size_t maxStepsPerActivation) const;
  void maybeTraceProcessStepOverflow(ProcessId procId,
                                     size_t effectiveMaxProcessSteps,
                                     size_t totalSteps, size_t funcBodySteps,
                                     llvm::StringRef currentFuncName,
                                     llvm::StringRef lastOpName) const;
  void maybeTraceSvaAssertionFailed(llvm::StringRef label, int64_t timeFs,
                                    mlir::Location loc) const;
  void maybeTraceImmediateAssertionFailed(llvm::StringRef label,
                                          mlir::Location loc) const;
  void maybeTraceUvmRunTestEntry(ProcessId procId, llvm::StringRef calleeName,
                                 uint64_t entryCount) const;
  void maybeTraceUvmRunTestReentryError(ProcessId procId,
                                        llvm::StringRef calleeName,
                                        uint64_t entryCount) const;
  void maybeTraceFuncCallInternalFailureWarning(
      llvm::StringRef calleeName) const;
  void maybeTraceSimTerminateTriggered(ProcessId procId,
                                       mlir::Location loc) const;
  void maybeTraceUvmJitPromotionCandidate(llvm::StringRef stableKey,
                                          uint64_t hitCount,
                                          uint64_t hotThreshold,
                                          size_t budgetRemaining) const;
  void maybeTraceGetNameLoop(ProcessId procId, uint64_t seq,
                             const SimTime &now, size_t callDepth,
                             size_t callStackSize, bool waiting,
                             uint64_t sameSiteStreak, uint64_t getNameCount,
                             uint64_t getFullNameCount,
                             uint64_t inGetNameBodyCount,
                             uint64_t randEnabledCount,
                             llvm::StringRef calleeName,
                             llvm::StringRef currentFuncName) const;
  void maybeTraceGetNameLoopLLVM(ProcessId procId, uint64_t seq,
                                 const SimTime &now, size_t callDepth,
                                 size_t callStackSize, bool waiting,
                                 size_t dynamicStringCount,
                                 size_t internStringCount,
                                 uint64_t sameSiteStreak,
                                 uint64_t randEnabledCount,
                                 uint64_t getNameCount,
                                 uint64_t inGetNameBodyCount,
                                 llvm::StringRef calleeName,
                                 llvm::StringRef currentFuncName) const;
  void maybeTraceBaudFastPathReject(ProcessId procId,
                                    llvm::StringRef calleeName,
                                    llvm::StringRef reason) const;
  void maybeTraceBaudFastPathNullSelfStall(ProcessId procId,
                                           llvm::StringRef calleeName) const;
  void maybeTraceBaudFastPathGep(llvm::StringRef calleeName, bool baseMatches,
                                 size_t dynamicIndexCount,
                                 llvm::ArrayRef<int32_t> rawIndices) const;
  void maybeTraceBaudFastPathMissingFields(llvm::StringRef calleeName,
                                           unsigned gepSeen, bool sawClockField,
                                           bool sawOutputField) const;
  void maybeTraceBaudFastPathBatchParityAdjust(ProcessId procId,
                                               llvm::StringRef calleeName,
                                               uint64_t adjustedEdges) const;
  void maybeTraceBaudFastPathBatchMismatch(ProcessId procId,
                                           llvm::StringRef calleeName,
                                           uint64_t pendingEdges,
                                           bool haveClockSample,
                                           bool clockSample,
                                           bool clockSampleValid,
                                           bool lastClockSample,
                                           uint64_t nowFs) const;
  void maybeTraceBaudFastPathBatchSchedule(ProcessId procId,
                                           llvm::StringRef calleeName,
                                           uint64_t batchEdges,
                                           uint64_t edgeInterval,
                                           uint64_t stableIntervals,
                                           int32_t count,
                                           int32_t divider) const;
  void maybeTraceBaudFastPathHit(ProcessId procId,
                                 llvm::StringRef calleeName, int32_t divider,
                                 bool primed, bool countLocalOnly,
                                 SignalId clockSignalId,
                                 SignalId outputSignalId) const;
  void maybeTraceFirRegUpdate(SignalId signalId,
                              const InterpretedValue &newValue,
                              bool posedge) const;
  void traceI3CForkRuntimeEvent(llvm::StringRef tag, ProcessId parentProcId,
                                ProcessId childProcId, ForkId forkId,
                                llvm::StringRef mode = "");
  void maybeTraceJoinNoneCheck(ProcessId procId, ForkId forkId,
                               ProcessId childProcId,
                               unsigned pollCount) const;
  void maybeTraceJoinNoneWait(ProcessId procId, ForkId forkId,
                              ProcessId childProcId,
                              unsigned pollCount) const;
  void maybeTraceJoinNoneResume(ProcessId procId, ForkId forkId,
                                unsigned pollCount,
                                bool waitingForChildReady) const;
  void maybeTraceForkIntercept(ProcessId procId, llvm::StringRef joinKind,
                               size_t branchCount, uint64_t phaseAddr,
                               bool hadExplicitBlockingMap,
                               bool shapeMatch) const;
  void maybeTraceForkInterceptObjectionWait(ProcessId procId,
                                            int64_t objectionHandle,
                                            int64_t objectionCount,
                                            uint64_t phaseAddr) const;
  void maybeTraceForkCreate(ProcessId procId, ForkId forkId,
                            llvm::StringRef joinKind, size_t branchCount,
                            uint64_t phaseAddr, bool shapeExec,
                            llvm::StringRef parentFunc,
                            llvm::StringRef forkName) const;
  void maybeTraceJoinNoneYield(ProcessId procId, ForkId forkId) const;
  void maybeTraceProcessFinalize(ProcessId procId, bool killed) const;
  void maybeTraceWaitSensitivityList(ProcessId procId, llvm::StringRef tag,
                                     const SensitivityList &waitList) const;
  void maybeTraceWaitEventCache(ProcessId procId, llvm::StringRef action,
                                mlir::Operation *waitEventOp,
                                const SensitivityList &waitList) const;
  void maybeTraceWaitEventNoop(ProcessId procId, llvm::StringRef funcName,
                               mlir::Operation *waitEventOp) const;
  void maybeTraceForkTerminator(ProcessId procId) const;
  void maybeTraceJoinAnyImmediate(ProcessId procId, ForkId forkId) const;
  void maybeTraceDisableForkBegin(ProcessId procId, size_t forkCount,
                                  bool deferredFire) const;
  void maybeTraceDisableForkDeferredPoll(ProcessId procId, ForkId forkId,
                                         ProcessId childProcId,
                                         unsigned pollCount) const;
  void maybeTraceDisableForkDeferredArm(ProcessId procId, ForkId forkId,
                                        ProcessId childProcId,
                                        ProcessState childSchedState,
                                        uint64_t token) const;
  void maybeTraceDisableForkChild(ProcessId procId, ForkId forkId,
                                  ProcessId childProcId,
                                  llvm::StringRef mode) const;
  bool shouldDeferDisableFork(ProcessId procId, ProcessId &deferChildId,
                              ForkId &deferForkId,
                              ProcessState &deferChildSchedState);
  void fireDeferredDisableFork(ProcessId procId, uint64_t deferToken);

  /// Interpret a sim.fork operation.
  /// Creates child processes for each branch region and schedules them.
  mlir::LogicalResult interpretSimFork(ProcessId procId, sim::SimForkOp forkOp);

  /// Interpret a sim.fork.terminator operation.
  /// Marks the forked child process as complete.
  mlir::LogicalResult interpretSimForkTerminator(ProcessId procId,
                                                  sim::SimForkTerminatorOp termOp);

  /// Interpret a sim.join operation.
  /// Waits for all processes in the fork to complete.
  mlir::LogicalResult interpretSimJoin(ProcessId procId, sim::SimJoinOp joinOp);

  /// Interpret a sim.join_any operation.
  /// Waits for any one process in the fork to complete.
  mlir::LogicalResult interpretSimJoinAny(ProcessId procId,
                                           sim::SimJoinAnyOp joinAnyOp);

  /// Interpret a sim.wait_fork operation.
  /// Waits for all child processes spawned by the current process.
  mlir::LogicalResult interpretSimWaitFork(ProcessId procId,
                                            sim::SimWaitForkOp waitForkOp);

  /// Interpret a sim.disable_fork operation.
  /// Terminates all processes in the specified fork group.
  mlir::LogicalResult interpretSimDisableFork(ProcessId procId,
                                               sim::SimDisableForkOp disableForkOp);

  /// Evaluate a format string operation to produce output string.
  std::string evaluateFormatString(ProcessId procId, mlir::Value fmtValue);

  /// Drop cached SSA values reachable from a format expression so subsequent
  /// evaluateFormatString calls observe fresh signal/probe values.
  void invalidateFormatValueCache(ProcessId procId, mlir::Value fmtValue);

  //===--------------------------------------------------------------------===//
  // Seq Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret an llhd.yield operation for llhd.combinational processes.
  mlir::LogicalResult interpretCombinationalYield(ProcessId procId,
                                                  llhd::YieldOp yieldOp);

  /// Interpret a seq.yield operation (terminator for seq.initial).
  mlir::LogicalResult interpretSeqYield(ProcessId procId, seq::YieldOp yieldOp);

  //===--------------------------------------------------------------------===//
  // Moore Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret a moore.wait_event operation.
  /// This suspends the process until a signal change is detected, similar to
  /// llhd.wait but using Moore dialect edge detection semantics.
  mlir::LogicalResult interpretMooreWaitEvent(ProcessId procId,
                                               moore::WaitEventOp waitEventOp);

  //===--------------------------------------------------------------------===//
  // LLVM Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret an llvm.alloca operation.
  mlir::LogicalResult interpretLLVMAlloca(ProcessId procId,
                                           mlir::LLVM::AllocaOp allocaOp);

  /// Interpret an llvm.load operation.
  mlir::LogicalResult interpretLLVMLoad(ProcessId procId,
                                         mlir::LLVM::LoadOp loadOp);

  /// Interpret an llvm.store operation.
  mlir::LogicalResult interpretLLVMStore(ProcessId procId,
                                          mlir::LLVM::StoreOp storeOp);

  /// Re-evaluate synthetic tri-state interface rules that depend on the
  /// updated source field and drive affected destination fields.
  void applyInterfaceTriStateRules(SignalId triggerSigId);

  /// Interpret an llvm.getelementptr operation.
  mlir::LogicalResult interpretLLVMGEP(ProcessId procId,
                                        mlir::LLVM::GEPOp gepOp);

  /// Interpret an llvm.call operation.
  mlir::LogicalResult interpretLLVMCall(ProcessId procId,
                                         mlir::LLVM::CallOp callOp);

  /// Interpret the runtime helper for SystemVerilog wait(condition).
  mlir::LogicalResult interpretMooreWaitConditionCall(
      ProcessId procId, mlir::LLVM::CallOp callOp);

  /// Intercept a DPI function call (func.func with no body).
  /// Returns success if the function was intercepted, failure otherwise.
  mlir::LogicalResult interceptDPIFunc(ProcessId procId,
                                        llvm::StringRef calleeName,
                                        mlir::LLVM::CallOp callOp);
  mlir::LogicalResult interceptDPIFunc(ProcessId procId,
                                        llvm::StringRef calleeName,
                                        mlir::func::CallOp callOp);

  /// Interpret an LLVM function body.
  /// If callOperands is provided, signal mappings are created for BlockArguments
  /// when the corresponding call operand resolves to a signal ID.
  /// @param callerOp The call operation in the caller (for call stack frames).
  /// @param resumeBlock If not null, resume from this block instead of entry.
  /// @param resumeOp If resumeBlock is set, resume from this operation.
  mlir::LogicalResult
  interpretLLVMFuncBody(ProcessId procId, mlir::LLVM::LLVMFuncOp funcOp,
                        llvm::ArrayRef<InterpretedValue> args,
                        llvm::SmallVectorImpl<InterpretedValue> &results,
                        llvm::ArrayRef<mlir::Value> callOperands = {},
                        mlir::Operation *callerOp = nullptr,
                        mlir::Block *resumeBlock = nullptr,
                        mlir::Block::iterator resumeOp = {});

  /// Get the size in bytes for an LLVM type (sum of field sizes, no alignment
  /// padding, matching MooreToCore's sizeof computation).
  unsigned getLLVMTypeSize(mlir::Type type) const;

  /// Get the size in bytes for an LLVM type, matching GEP offset computation
  /// (byte-addressable, no sub-byte packing).
  unsigned getLLVMTypeSizeForGEP(mlir::Type type) const;

  /// Get the bit width for memory-backed aggregate layout used by ref-addressed
  /// loads/stores (GEP-style byte-addressable sizing).
  unsigned getMemoryLayoutBitWidth(mlir::Type type) const;

  /// Get the natural alignment in bytes for an LLVM type.
  unsigned getLLVMTypeAlignment(mlir::Type type) const;

  /// Get the byte offset of a field within an LLVM struct type (unaligned,
  /// matching MooreToCore's layout).
  unsigned getLLVMStructFieldOffset(mlir::LLVM::LLVMStructType structType,
                                   unsigned fieldIndex) const;

  /// Find the memory block for a pointer value.
  MemoryBlock *findMemoryBlock(ProcessId procId, mlir::Value ptr);

  /// Find a memory block by address (searches mallocBlocks).
  MemoryBlock *findMemoryBlockByAddress(uint64_t addr,
                                        ProcessId procId = static_cast<ProcessId>(-1),
                                        uint64_t *outOffset = nullptr);

  /// Find a native memory block by address (e.g., assoc array element refs).
  bool findNativeMemoryBlockByAddress(uint64_t addr, uint64_t *outOffset,
                                      size_t *outSize) const;

  /// Read a Moore string payload from interpreter/native memory.
  bool tryReadStringKey(ProcessId procId, uint64_t strPtrVal, int64_t strLen,
                        std::string &out);

  /// Decode a Moore packed string value ({ptr, len}) to std::string.
  std::string readMooreStringStruct(ProcessId procId,
                                    InterpretedValue packedValue);
  std::string readMooreStringStruct(ProcessId procId, mlir::Value operand);

  /// Intercept config_db implementation methods invoked via call_indirect.
  bool tryInterceptConfigDbCallIndirect(
      ProcessId procId, mlir::func::CallIndirectOp callIndirectOp,
      llvm::StringRef calleeName,
      llvm::ArrayRef<InterpretedValue> args);

  /// Intercept native UVM port helper calls in LLVM::CallOp path.
  bool tryInterceptUvmPortCall(ProcessId procId, llvm::StringRef calleeName,
                               mlir::LLVM::CallOp callOp);

  //===--------------------------------------------------------------------===//
  // Value Management
  //===--------------------------------------------------------------------===//

  /// Get the interpreted value for an SSA value.
  InterpretedValue getValue(ProcessId procId, mlir::Value value);

  /// Set the interpreted value for an SSA value.
  void setValue(ProcessId procId, mlir::Value value, InterpretedValue val);

  /// Resolve a process handle to a process ID.
  ProcessId resolveProcessHandle(uint64_t handle);

  /// Register a process execution state and its stable handle mapping.
  void registerProcessState(ProcessId procId, ProcessExecutionState &&state,
                            std::optional<uint32_t> initialSeed = std::nullopt);

  /// Resume any processes awaiting completion of the target process.
  void notifyProcessAwaiters(ProcessId procId);

  /// Finalize a process (finished or killed) and wake awaiters.
  void finalizeProcess(ProcessId procId, bool killed);

  /// Recursively kill a process and all its fork descendants.
  /// Used when a UVM phase ends to clean up forever-loop monitors, etc.
  void killProcessTree(ProcessId procId);

  /// Returns true if `rootProcId` or any non-halted descendant is still alive.
  bool isProcessSubtreeAlive(ProcessId rootProcId) const;

  //===--------------------------------------------------------------------===//
  // Signal Registry Bridge
  //===--------------------------------------------------------------------===//

  /// Export all registered signals to the MooreRuntime signal registry.
  /// This enables DPI/VPI functions like uvm_hdl_read() to access signals.
  void exportSignalsToRegistry();

  /// Set up accessor callbacks for the signal registry.
  void setupRegistryAccessors();

  //===--------------------------------------------------------------------===//
  // Member Variables
  //===--------------------------------------------------------------------===//

  /// Reference to the process scheduler.
  ProcessScheduler &scheduler;

  /// Fork/join manager for concurrent process spawning.
  ForkJoinManager forkJoinManager;

  /// Synchronization primitives manager for semaphores and mailboxes.
  SyncPrimitivesManager syncPrimitivesManager;

  /// Maximum number of ops a process may execute before forcing a stop.
  size_t maxProcessSteps = 0;

  /// Optional per-function operation cap (0 = unlimited).
  /// Guarded by CIRCT_SIM_MAX_FUNC_OPS for debugging pathological loops.
  size_t maxFunctionOps = 0;

  /// Name of the top module (for hierarchical path construction).
  std::string moduleName;

  /// Active instance context for signal/process resolution.
  InstanceId activeInstanceId = 0;

  /// Recursion depth counter for evaluateContinuousValue to prevent stack
  /// overflow when instance hierarchies create deep recursive chains.
  unsigned continuousEvalDepth = 0;

  /// Set of signal IDs currently being evaluated through combSignalDriveMap
  /// in the recursive evaluateContinuousValue chain. Used to detect cycles
  /// (e.g., signal A drives signal B, signal B drives signal A) and break
  /// them by falling back to reading the signal value from the scheduler.
  llvm::DenseSet<SignalId> continuousEvalVisitedSignals;

  /// Cache of process → yield WaitOp for inline combinational evaluation.
  /// Maps ProcessOp to its yield WaitOp (or nullptr if not applicable).
  llvm::DenseMap<mlir::Operation *, mlir::Operation *>
      processInlineYieldCache;

  /// Set of process operations currently being inline-evaluated in the
  /// recursive evaluateContinuousValue chain. Used to detect cycles.
  llvm::DenseSet<mlir::Operation *> continuousEvalVisitedProcesses;

  /// Set of (procId, waitOp) pairs that have already been through the
  /// empty-sensitivity "always @(*)" fallback delta-resume path at least once.
  /// Used to prevent infinite delta cycles for processes that have no
  /// detectable LLHD signal dependencies (e.g., string-type always blocks).
  llvm::DenseSet<std::pair<ProcessId, mlir::Operation *>>
      emptySensitivityFallbackExecuted;

  /// Next instance ID to allocate for a hw.instance.
  InstanceId nextInstanceId = 1;

  /// Root MLIR module (for symbol table lookups).
  mlir::ModuleOp rootModule;

  /// Set of module names that have already been processed (to avoid
  /// processing the same module multiple times for different instances).
  llvm::StringSet<> processedModules;

  /// Cache of discovered operations per module to avoid repeated walks.
  llvm::StringMap<DiscoveredOps> discoveredOpsCache;

  /// Map from MLIR signal values to signal IDs.
  llvm::DenseMap<mlir::Value, SignalId> valueToSignal;

  /// Map from alloca Values to the signal IDs they back.
  /// Built during initialization by scanning process yield/drive patterns.
  /// Used to propagate fork-child alloca writes to the backing signal.
  llvm::DenseMap<mlir::Value, SignalId> allocaBackingSignal;

  /// Map from instance IDs to per-instance signal maps.
  llvm::DenseMap<InstanceId, llvm::DenseMap<mlir::Value, SignalId>>
      instanceValueToSignal;

  /// Map from instance IDs to input mappings (block args -> parent values).
  llvm::DenseMap<InstanceId, InstanceInputMapping> instanceInputMaps;

  /// Map from instance IDs to per-instance process maps.
  llvm::DenseMap<InstanceId, llvm::DenseMap<mlir::Operation *, ProcessId>>
      instanceOpToProcessId;

  struct FirRegState {
    SignalId signalId = 0;
    InterpretedValue prevClock;
    bool hasPrevClock = false;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
  };

  /// State for a clocked assertion checker (edge detection + LTL evaluation).
  struct ClockedAssertionState {
    InterpretedValue prevClock;
    bool hasPrevClock = false;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
    /// Per-operation history buffer for LTL temporal evaluation.
    /// Key: the implication op; Value: deque of antecedent match booleans.
    llvm::DenseMap<mlir::Operation *, std::deque<bool>> anteHistory;
  };

  /// Evaluate an LTL property tree recursively, handling temporal operators.
  /// Returns true if the property holds at the current evaluation point.
  bool evaluateLTLProperty(mlir::Value val, ClockedAssertionState &state);

  /// Map from instance IDs to per-instance firreg state maps.
  llvm::DenseMap<InstanceId, llvm::DenseMap<mlir::Operation *, FirRegState>>
      instanceFirRegStates;

  /// Map from instance IDs to instance result maps (result value -> output info).
  llvm::DenseMap<InstanceId, llvm::DenseMap<mlir::Value, InstanceOutputInfo>>
      instanceOutputMap;

  struct InstanceOutputUpdate {
    SignalId signalId = 0;
    mlir::Value outputValue;
    InstanceId instanceId = 0;
    llvm::SmallVector<ProcessId, 4> processIds;
    InstanceInputMapping inputMap;
  };
  llvm::SmallVector<InstanceOutputUpdate, 16> instanceOutputUpdates;

  /// Map from child module input block arguments to instance operand values.
  mutable llvm::DenseMap<mlir::Value, mlir::Value> inputValueMap;
  /// Map from child module input block arguments to instance IDs.
  mutable llvm::DenseMap<mlir::Value, InstanceId> inputValueInstanceMap;

  /// Map from signal IDs to signal names.
  llvm::DenseMap<SignalId, std::string> signalIdToName;

  /// Map from signal IDs to their nested value types.
  llvm::DenseMap<SignalId, mlir::Type> signalIdToType;

  /// Pending epsilon drives - for immediate blocking assignment semantics.
  /// When a signal is driven with epsilon delay within the same process,
  /// this map holds the value so subsequent probes can see it immediately.
  /// Key is SignalId, value is the pending value.
  llvm::DenseMap<SignalId, InterpretedValue> pendingEpsilonDrives;

  /// NBA yield block argument chain tracking. When NBA (non-blocking) bit-select
  /// assignments occur inside for-loops, the MLIR structure reads a stale probe
  /// from before the loop rather than the accumulated block argument. This map
  /// records (Block*, argIndex) pairs that carry accumulated values destined for
  /// NBA yield operands. The cf.br handler uses this to update stale probes.
  llvm::DenseMap<std::pair<mlir::Block *, unsigned>, SignalId>
      nbaYieldBlockArgMap;

  /// Maps (ProcessId, SignalId) to ProbeOp result Values inside the process.
  /// When a cf.br updates a block arg in nbaYieldBlockArgMap, we also update
  /// the valueMap for these probe results so loop iterations see accumulated
  /// values rather than the stale pre-loop probe.
  llvm::DenseMap<std::pair<ProcessId, SignalId>,
                 llvm::SmallVector<mlir::Value, 2>>
      nbaProbeValues;

  /// Maps SignalId to ALL block argument Values in the NBA yield chain.
  /// When any block arg in the chain is updated, ALL other block args for the
  /// same signal must also be updated so that inner-loop struct_extract ops
  /// that read outer-loop block args see accumulated values.
  llvm::DenseMap<SignalId, llvm::SmallVector<mlir::Value, 4>>
      nbaChainBlockArgs;

  /// Map from signal IDs to pre-force saved values. When a signal is forced,
  /// we save the current value here so we can restore it on release.
  /// The saved value is updated by procedural drives during force, so that
  /// release restores the LAST driven value (IEEE 1800-2017 §10.6.2).
  llvm::DenseMap<SignalId, InterpretedValue> forcedSignalSavedValues;

  /// Set of signal IDs currently under force. Normal drives (continuous and
  /// procedural) are suppressed while a signal is in this set. Procedural
  /// drives update forcedSignalSavedValues instead. Cleared on release.
  llvm::DenseSet<SignalId> forcedSignals;

  /// Map from signal IDs to {procId, Value key} for backing memory blocks.
  /// Allocated when unrealized_conversion_cast converts !llhd.ref<T> to
  /// !llvm.ptr so that llvm.store writes become visible to later llhd.prb reads.
  llvm::DenseMap<SignalId, std::pair<ProcessId, mlir::Value>>
      signalBackingMemory;

  /// Map from process IDs to execution states.
  /// NOTE: This uses std::map rather than DenseMap to guarantee reference
  /// stability.  evaluateCombinationalOp inserts and erases temporary entries
  /// during getValue calls; with DenseMap this can trigger a rehash and
  /// invalidate references held by callers (e.g., interpretWait's
  /// `auto &state = processStates[procId]`).  std::map provides stable
  /// references across inserts and erases.
  std::map<ProcessId, ProcessExecutionState> processStates;

  /// Shared function result cache across all processes for pure, high-frequency
  /// UVM/domain getters. This complements per-process funcResultCache to avoid
  /// repeating identical graph/singleton queries in sibling processes.
  llvm::DenseMap<mlir::Operation *,
                 llvm::DenseMap<uint64_t,
                                llvm::SmallVector<InterpretedValue, 2>>>
      sharedFuncResultCache;

  /// Number of shared function result cache hits (for diagnostics).
  uint64_t sharedFuncCacheHits = 0;

  /// Cached active process state for the currently-executing process.
  /// Set by executeProcess() to avoid repeated std::map lookups in
  /// getValue()/setValue()/executeStep() on every operation.
  ProcessId activeProcessId = 0;
  ProcessExecutionState *activeProcessState = nullptr;

  /// Map from process handle values to process IDs.
  llvm::DenseMap<uint64_t, ProcessId> processHandleToId;

  /// Map from process IDs to lists of awaiters.
  llvm::DenseMap<ProcessId, llvm::SmallVector<ProcessId, 4>> processAwaiters;

  /// Next temporary process ID used for combinational evaluation.
  static constexpr ProcessId kTempProcessIdBase = 1ull << 60;
  ProcessId nextTempProcId = kTempProcessIdBase;

  /// Map from llhd.process ops to process IDs.
  llvm::DenseMap<mlir::Operation *, ProcessId> opToProcessId;

  /// Operation execution statistics (by op name).
  llvm::StringMap<uint64_t> opStats;

  bool collectOpStats = false;

  /// Module-level drives connected to process results.
  struct ModuleDrive {
    llhd::DriveOp driveOp;
    ProcessId procId = InvalidProcessId;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
  };
  llvm::SmallVector<ModuleDrive, 4> moduleDrives;

  /// Map from signal ID to indices in moduleDrives that depend on that signal
  /// (via llhd.prb in the combinational chain). Used to re-evaluate module
  /// drives when dependent signals change.
  llvm::DenseMap<SignalId, llvm::SmallVector<size_t, 2>>
      signalDependentModuleDrives;

  /// Static module-level drives (not connected to process results).
  struct StaticModuleDrive {
    llhd::DriveOp driveOp;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
  };
  llvm::SmallVector<StaticModuleDrive, 4> staticModuleDrives;

  /// Map from signal ID to its combinational drive expression.
  /// Used by evaluateContinuousValue to trace through intermediate signals
  /// whose values may be stale (epsilon delays not yet fired) instead of
  /// reading stale signal values. Only populated for static drives with no
  /// enable condition (pure combinational wires).
  struct CombSignalDriveInfo {
    mlir::Value driveValue;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
  };
  llvm::DenseMap<SignalId, CombSignalDriveInfo> combSignalDriveMap;

  /// Signals that have multiple drives and should NOT be in combSignalDriveMap.
  /// For read-modify-write patterns (e.g., Brent-Kung adder bit-by-bit drives),
  /// the combSignalDriveMap shortcut would only record the last drive.
  llvm::DenseSet<SignalId> multiDrivenSignals;

  /// Signals that require distinct continuous-assignment driver IDs and
  /// strength-based multi-driver resolution.
  ///
  /// We keep the existing grouped last-write-wins path for read-modify-write
  /// combinational patterns, but nets with explicit non-default strengths
  /// (e.g. pullups/open-drain) need true multi-driver resolution.
  llvm::DenseSet<SignalId> distinctContinuousDriverSignals;

  /// Cache for detecting array-element drive patterns.
  /// Maps a DriveOp (by Operation pointer) to the constant element index and
  /// element bit width when the drive value is a process result produced by
  /// hw.array_inject. A negative elementIndex means "not an array element
  /// drive" (negative cache entry). This is used to implement read-modify-write
  /// semantics for multiple processes writing to different elements of the same
  /// unpacked array signal.
  struct ArrayElementDriveInfo {
    int64_t elementIndex; ///< -1 if not an array element drive
    unsigned elementBitWidth;
  };
  llvm::DenseMap<mlir::Operation *, ArrayElementDriveInfo>
      arrayElementDriveCache;

  /// Detect if a module-level drive is an array-element drive (i.e., the drive
  /// value comes from a process that does hw.array_inject on the driven signal).
  /// Returns the element index and bit width if detected, or std::nullopt.
  std::optional<std::pair<int64_t, unsigned>>
  detectArrayElementDrive(llhd::DriveOp driveOp);

  /// Memory event waiter for polling-based event detection.
  /// Used for UVM events stored as boolean fields in class instances where
  /// no signal is available to wait on.
  struct MemoryEventWaiter {
    enum class EdgeMode : uint8_t {
      AnyChange = 0,
      RisingEdge = 1,
      FallingEdge = 2,
    };

    /// The address of the memory location to poll.
    uint64_t address = 0;
    /// The last seen value at this address.
    uint64_t lastValue = 0;
    /// The size of the value in bytes (1 for bool/i1).
    unsigned valueSize = 1;
    /// Edge trigger policy for this waiter.
    /// Event types (!moore.event) use RisingEdge semantics.
    EdgeMode edgeMode = EdgeMode::AnyChange;
  };

  /// Map from process IDs to their memory event waiters.
  /// A process waiting on a memory event will have an entry here.
  llvm::DenseMap<ProcessId, MemoryEventWaiter> memoryEventWaiters;

  /// Check all memory event waiters and wake processes whose watched
  /// memory location has changed.
  void checkMemoryEventWaiters();

  /// Track the realTime (in femtoseconds) when each event was triggered.
  /// Per IEEE 1800-2017 §15.5.3, `.triggered` returns true only within the
  /// same time slot where the event was triggered. Events auto-clear when
  /// realTime advances.
  llvm::DenseMap<uint64_t, uint64_t> eventTriggerTime;

  /// Registered seq.firreg state keyed by op.
  llvm::DenseMap<mlir::Operation *, FirRegState> firRegStates;

  /// Pending firreg signal updates for two-phase evaluation.
  /// During NBA region execution, firregs evaluate inputs using pre-update
  /// signal values and defer their updates here. After all firregs have
  /// evaluated, the post-NBA callback applies these updates atomically.
  /// This implements correct Verilog non-blocking assignment semantics
  /// where all RHS values are sampled before any LHS updates occur.
  std::vector<std::pair<SignalId, SignalValue>> pendingFirRegUpdates;

  /// When true, executeFirReg defers signal updates to pendingFirRegUpdates
  /// instead of calling scheduler.updateSignal immediately.
  bool deferFirRegUpdates = false;

  /// Signals that were conditionally driven by a process in the current NBA
  /// batch, with their drive values.  When two always blocks drive the same
  /// register, only the block that explicitly assigns should take effect
  /// (Verilog last-write-wins).  The firreg (register-retention driver) must
  /// yield to the conditional process drive; otherwise the firreg's captured
  /// old value overwrites the process's new value.  The value is applied in
  /// the post-NBA callback alongside firreg updates so it is visible before
  /// cbValueChange callbacks fire (e.g. cocotb RisingEdge).
  llvm::DenseMap<SignalId, SignalValue> processConditionalDriveValues;

  /// Registered clocked assertion state keyed by op.
  llvm::DenseMap<mlir::Operation *, ClockedAssertionState> clockedAssertionStates;

  /// Counter of clocked assertion failures during simulation.
  size_t clockedAssertionFailures = 0;

  /// Callback for sim.terminate operation.
  std::function<void(bool, bool)> terminateCallback;

  /// Callback to check if abort has been requested.
  std::function<bool()> shouldAbortCallback;

  /// Callback to invoke when abort is triggered.
  std::function<void()> abortCallback;

  /// Callback for $dumpfile (VCD file creation).
  std::function<void(llvm::StringRef)> dumpfileCallback;

  /// Callback for $dumpvars (start VCD tracing).
  std::function<void()> dumpvarsCallback;

  /// Callback for $dumpoff (pause VCD recording).
  std::function<void()> dumpoffCallback;

  /// Callback for $dumpon (resume VCD recording).
  std::function<void()> dumponCallback;

  /// Flag indicating if termination has been requested.
  bool terminationRequested = false;
  /// Number of top modules initialized. In multi-top (dual-top) mode,
  /// sim.terminate success is deferred to let other modules continue.
  unsigned topModuleCount = 0;
  /// Whether we are currently in global initialization (constructors, etc.).
  /// During this phase, sim.terminate should not halt the process to allow
  /// UVM initialization to complete (re-entrant calls set uvm_top properly).
  bool inGlobalInit = false;

  /// True when simulation runs in compile mode.
  bool compileModeEnabled = false;

  /// Optional JIT manager used for thunk dispatch/deopt accounting.
  JITCompileManager *jitCompileManager = nullptr;

  /// First observed JIT deopt reason for each process ID.
  llvm::DenseMap<uint64_t, std::string> jitDeoptReasonByProcess;

  /// First observed JIT deopt detail for each process ID.
  llvm::DenseMap<uint64_t, std::string> jitDeoptDetailByProcess;

  /// External port signal map provided by the simulation context.
  /// Used to populate valueToSignal for non-ref-type module ports.
  const llvm::StringMap<SignalId> *externalPortSignals = nullptr;

  /// Per-process metadata for periodic clock toggle native thunks.
  llvm::DenseMap<ProcessId, PeriodicToggleClockThunkSpec>
      periodicToggleClockThunkSpecs;

  /// Block-level JIT compiler (lazily created when block JIT is enabled).
  std::unique_ptr<JITBlockCompiler> jitBlockCompiler;

  /// Per-process JIT block specs for compiled hot blocks.
  llvm::DenseMap<ProcessId, std::unique_ptr<JITBlockSpec>> jitBlockSpecs;

  /// Whether block-level JIT is enabled.
  bool blockJITEnabled = false;

  /// Cached direct process fast-path classification mask by process.
  llvm::DenseMap<ProcessId, uint8_t> directProcessFastPathKinds;

  /// Test hook: force native thunks to request deopt after dispatch.
  bool forceJitThunkDeoptRequests = false;

  /// Grace period for $finish with active forked children.
  /// When $finish(success=true) fires but forked children are still running
  /// (e.g., UVM phase hopper), we record the wall-clock time and allow a
  /// grace period for cleanup phases to run. After the grace period, we force
  /// termination. Uses wall-clock time (not simulation time) because UVM
  /// phase execution happens entirely at simulation time 0 in delta cycles.
  bool finishGracePeriodActive = false;
  std::chrono::steady_clock::time_point finishGracePeriodStart;
  /// Grace period duration: 30 seconds wall-clock. UVM phase dispatch in the
  /// interpreter takes ~10-15s, so 30s gives enough margin for phases to
  /// complete and produce output while still being much shorter than the
  /// 120s external timeout.
  static constexpr int kFinishGracePeriodSecs = 30;

  /// Cache of function lookups to avoid repeated moduleOp.lookupSymbol calls.
  /// Maps function name to a cached result:
  ///   - LLVM::LLVMFuncOp* if found as LLVM function
  ///   - func::FuncOp* cast to Operation* if found as MLIR function
  ///   - nullptr if not found (negative cache)
  /// The second element of the pair is 0 for LLVM, 1 for MLIR func, 2 for not found.
  struct CachedFuncLookup {
    mlir::Operation *op = nullptr;
    uint8_t kind = 0; // 0=LLVM, 1=func, 2=not found
  };
  llvm::StringMap<CachedFuncLookup> funcLookupCache;

  /// Function call profiling: counts how many times each func.call and
  /// call_indirect target is invoked. Keyed by callee name for easy reporting.
  /// Enabled when CIRCT_SIM_PROFILE_FUNCS env var is set.
  llvm::StringMap<uint64_t> funcCallProfile;

  struct JitRuntimeIndirectSiteData {
    uint64_t siteId = 0;
    std::string owner;
    std::string location;
    uint64_t callsTotal = 0;
    uint64_t unresolvedCalls = 0;
    uint64_t targetSetVersion = 0;
    llvm::StringMap<uint64_t> targetCalls;
  };
  llvm::DenseMap<mlir::Operation *, JitRuntimeIndirectSiteData>
      jitRuntimeIndirectSiteProfiles;
  llvm::DenseMap<ProcessId,
                 llvm::SmallVector<JitRuntimeIndirectSiteGuardSpec, 2>>
      jitProcessThunkIndirectSiteGuards;
  uint64_t jitRuntimeIndirectNextSiteId = 1;
  bool jitRuntimeIndirectProfileEnabled = false;

  struct CallIndirectRuntimeOverrideSiteInfo {
    bool hasStaticMethodIndex = false;
    int64_t staticMethodIndex = -1;
  };
  llvm::DenseMap<mlir::Operation *, CallIndirectRuntimeOverrideSiteInfo>
      callIndirectRuntimeOverrideSiteInfo;

  /// Cache runtime vtable slot lookups used by call_indirect runtime override.
  /// Key: (runtime_vtable_address, method_index), Value: function address.
  llvm::DenseMap<std::pair<uint64_t, int64_t>, uint64_t>
      callIndirectRuntimeVtableSlotCache;

  JitRuntimeIndirectSiteData &
  getOrCreateJitRuntimeIndirectSiteData(ProcessId procId,
                                        mlir::func::CallIndirectOp callOp);
  void noteJitRuntimeIndirectResolvedTarget(ProcessId procId,
                                            mlir::func::CallIndirectOp callOp,
                                            llvm::StringRef calleeName);
  void noteJitRuntimeIndirectUnresolved(ProcessId procId,
                                        mlir::func::CallIndirectOp callOp);
  bool getCachedCallIndirectStaticMethodIndex(
      mlir::func::CallIndirectOp callOp, int64_t &methodIndex);

  /// UVM fast-path profiling counters keyed by fast-path action name.
  /// Enabled together with CIRCT_SIM_PROFILE_FUNCS.
  llvm::StringMap<uint64_t> uvmFastPathProfile;

  /// Per-action hit counters used by hotness-triggered JIT promotion hooks.
  llvm::StringMap<uint64_t> uvmFastPathHitCount;

  /// Set of fast-path action names that crossed the JIT hotness threshold and
  /// were selected as promotion candidates.
  llvm::DenseSet<llvm::StringRef> uvmJitPromotedFastPaths;

  /// Backing storage for promoted fast-path keys. DenseSet<StringRef> keys
  /// reference entries in this map to keep stable storage.
  llvm::StringMap<char> uvmJitPromotedStorage;

  /// Optional hotness-gated JIT promotion hooks for UVM fast paths.
  /// Controlled by env vars:
  ///   CIRCT_SIM_UVM_JIT_HOT_THRESHOLD
  ///   CIRCT_SIM_UVM_JIT_PROMOTION_BUDGET
  ///   CIRCT_SIM_UVM_JIT_TRACE_PROMOTIONS
  uint64_t uvmJitHotThreshold = 0;
  int64_t uvmJitPromotionBudget = 0;
  bool uvmJitTracePromotions = false;
  bool profileSummaryAtExitEnabled = false;
  uint64_t memorySampleIntervalSteps = 0;
  uint64_t memoryDeltaWindowSamples = 0;
  uint64_t memorySummaryTopProcesses = 0;
  uint64_t memorySampleNextStep = 0;
  uint64_t memorySampleCount = 0;
  uint64_t memorySamplePeakStep = 0;
  uint64_t memorySamplePeakTotalBytes = 0;

  struct MemoryStateSnapshot {
    uint64_t globalBlocks = 0;
    uint64_t globalBytes = 0;
    uint64_t mallocBlocks = 0;
    uint64_t mallocBytes = 0;
    uint64_t nativeBlocks = 0;
    uint64_t nativeBytes = 0;
    uint64_t processBlocks = 0;
    uint64_t processBytes = 0;
    uint64_t dynamicStrings = 0;
    uint64_t dynamicStringBytes = 0;
    uint64_t configDbEntries = 0;
    uint64_t configDbBytes = 0;
    uint64_t analysisConnPorts = 0;
    uint64_t analysisConnEdges = 0;
    uint64_t seqFifoMaps = 0;
    uint64_t seqFifoItems = 0;
    ProcessId largestProcessId = InvalidProcessId;
    uint64_t largestProcessBytes = 0;

    uint64_t totalTrackedBytes() const {
      return globalBytes + mallocBytes + nativeBytes + processBytes +
             dynamicStringBytes + configDbBytes;
    }
  };

  struct MemoryStateSample {
    uint64_t step = 0;
    MemoryStateSnapshot snapshot;
  };

  MemoryStateSnapshot memoryPeakSnapshot;

  /// Persisted process results: when a process halts and its valueMap is
  /// cleared, any yield values are preserved here so that module-level drives
  /// depending on process results can still read them.
  llvm::DenseMap<mlir::Value, InterpretedValue> persistedProcessResults;
  std::deque<MemoryStateSample> memorySampleHistory;
  std::string memoryPeakLargestProcessFunc;

  bool profilingEnabled = false;

  /// Global kill-switch for function result memoization.
  /// Enabled with CIRCT_SIM_DISABLE_FUNC_RESULT_CACHE=1.
  bool disableFuncResultCache = false;

  /// Optional fast paths for high-volume UVM report traffic.
  /// These are opt-in via env vars to avoid changing default behavior.
  /// CIRCT_SIM_FASTPATH_UVM_REPORT_INFO=1
  /// CIRCT_SIM_FASTPATH_UVM_REPORT_WARNING=1
  /// CIRCT_SIM_FASTPATH_UVM_GET_REPORT_OBJECT=1
  bool fastPathUvmReportInfo = false;
  bool fastPathUvmReportWarning = false;
  bool fastPathUvmGetReportObject = false;

  /// Cached env flag for sequencer tracing (CIRCT_SIM_TRACE_SEQ).
  /// Read once at construction to avoid std::getenv on every call_indirect.
  bool traceSeqEnabled = false;

  /// Cached env flag for analysis port tracing (CIRCT_SIM_TRACE_ANALYSIS).
  /// Read once at construction to avoid std::getenv on every port operation.
  bool traceAnalysisEnabled = false;

  /// Per-port cap for verbose sequencer queue resolution diagnostics.
  uint64_t traceSeqResolveLimit = 0;
  llvm::DenseMap<uint64_t, uint32_t> traceSeqResolvePrints;

  /// Cached env flag for config_db tracing (CIRCT_SIM_TRACE_CONFIG_DB).
  /// When enabled, prints set/get keys and hit/miss outcomes.
  bool traceConfigDbEnabled = false;
  /// Cached env flag for UVM run_test entry diagnostics.
  bool traceUvmRunTestEnabled = false;
  /// Cached env flag to enforce at-most-once UVM run_test entry.
  bool enforceSingleUvmRunTestEntry = false;

  /// Cached env flag for fork/join diagnostics (CIRCT_SIM_TRACE_FORK_JOIN).
  bool traceForkJoinEnabled = false;
  /// Cached env flag for I3C-focused fork runtime diagnostics.
  /// Enables high-context parent/child trace lines at disable_fork and
  /// join_none resume boundaries.
  bool traceI3CForkRuntimeEnabled = false;

  /// Cached env flag for call_indirect site-cache diagnostics.
  bool traceCallIndirectSiteCacheEnabled = false;

  /// Optional per-print stdout flush for sim.proc.print.
  /// Disabled by default for throughput; enable with
  /// CIRCT_SIM_FLUSH_PROC_PRINT=1 for interactive debugging.
  bool flushProcPrintEnabled = false;

  struct RegisteredMonitorState {
    bool active = false;
    bool enabled = true;
    ProcessId ownerProcId = InvalidProcessId;
    mlir::Value message;
    std::string lastOutput;
  };

  /// Active $monitor registration (SystemVerilog allows one active monitor).
  RegisteredMonitorState registeredMonitor;

  /// Track a UVM fast-path hit and evaluate hotness-gated promotion hooks.
  void noteUvmFastPathActionHit(llvm::StringRef actionKey);

  /// Look up sequencer queue cache entry for a pull-port address.
  bool lookupUvmSequencerQueueCache(uint64_t portAddr, uint64_t &queueAddr);

  /// Insert or update sequencer queue cache entry for a pull-port address.
  void cacheUvmSequencerQueueAddress(uint64_t portAddr, uint64_t queueAddr);

  /// Invalidate sequencer queue cache entry for a pull-port address.
  void invalidateUvmSequencerQueueCache(uint64_t portAddr);

  /// Canonicalize an object address to its allocation-owner base address.
  /// This is a structural normalization (no dynamic function calls).
  uint64_t canonicalizeUvmObjectAddress(ProcessId procId, uint64_t addr);

  /// Seed analysis-port connection traversal from both direct and alias-equivalent
  /// port addresses (canonicalized owner base addresses).
  void seedAnalysisPortConnectionWorklist(
      ProcessId procId, uint64_t portAddr,
      llvm::SmallVectorImpl<uint64_t> &worklist);

  /// Canonicalize a queue hint to the owning sequencer address.
  /// Returns true when the resolved address is strong enough to cache.
  bool canonicalizeUvmSequencerQueueAddress(ProcessId procId,
                                            uint64_t &queueAddr,
                                            mlir::Operation *callSite);

  /// Resolve the queue address for a pull port through cache and connection
  /// graph traversal. Returns true when the resolved hint should be cached.
  bool resolveUvmSequencerQueueAddress(ProcessId procId, uint64_t portAddr,
                                       mlir::Operation *callSite,
                                       uint64_t &queueAddr);

  /// Register a process waiting in get/get_next_item on a sequencer queue.
  /// queueAddr==0 means unresolved queue and wakes on any push.
  void enqueueUvmSequencerGetWaiter(uint64_t queueAddr, ProcessId procId,
                                    mlir::Operation *retryOp);

  /// Remove a process from sequencer get wait-queues.
  void removeUvmSequencerGetWaiter(ProcessId procId);

  /// Wake one pending sequencer get waiter for a queue (or unresolved queue 0).
  bool wakeOneUvmSequencerGetWaiter(uint64_t queueAddr);

  /// Wake one queue-specific waiter first, then unresolved waiters.
  void wakeUvmSequencerGetWaiterForPush(uint64_t queueAddr);

  /// Record ownership mapping for sequence item -> sequencer address.
  void recordUvmSequencerItemOwner(uint64_t itemAddr, uint64_t sqrAddr);

  /// Consume ownership mapping for sequence item and return sequencer address.
  uint64_t takeUvmSequencerItemOwner(uint64_t itemAddr);

  /// Track a dequeued sequence item for the caller process and for pull-port /
  /// sequencer aliases used by item_done resolution.
  void recordUvmDequeuedItem(ProcessId procId, uint64_t portAddr,
                             uint64_t queueAddr, uint64_t itemAddr);

  /// Resolve and consume the last dequeued item for an item_done caller.
  uint64_t takeUvmDequeuedItemForDone(ProcessId procId, uint64_t doneAddr,
                                      mlir::Operation *callSite);

  /// Build a point-in-time snapshot of runtime memory/state dimensions.
  MemoryStateSnapshot collectMemoryStateSnapshot() const;

  /// Periodically sample memory-state dimensions and update high-water marks.
  void maybeSampleMemoryState(uint64_t totalSteps);

  /// Cache for external functions that are NOT intercepted: when we've already
  /// determined that an external function has no matching __moore_* handler,
  /// cache that fact to skip the 128-entry string comparison chain on
  /// subsequent calls. Maps external LLVMFuncOp Operation* -> true (always).
  llvm::DenseSet<mlir::Operation *> nonInterceptedExternals;

  /// Map of dynamic string pointers to their content (for runtime string
  /// handling). Key is the pointer value, value is {data, len}.
  llvm::DenseMap<int64_t, std::pair<const char *, int64_t>> dynamicStrings;

  /// Persistent storage for strings created by the interpreter (e.g. from
  /// __moore_int_to_string and __moore_string_concat). The deque ensures
  /// stable pointers so that dynamicStrings entries remain valid.
  std::deque<std::string> interpreterStrings;

  /// Intern table for interpreter-created dynamic strings to avoid
  /// unbounded duplicate allocations from repeated concat/name hot paths.
  llvm::StringMap<std::pair<int64_t, int64_t>> internedDynamicStrings;

  /// Cache packed-string literal values -> interned string pointer/length.
  llvm::DenseMap<int64_t, std::pair<int64_t, int64_t>> packedStringValueCache;

  //===--------------------------------------------------------------------===//
  // Global Variable and VTable Support
  //===--------------------------------------------------------------------===//

  /// Memory storage for LLVM global variables.
  /// Maps global name to memory block.
  llvm::StringMap<MemoryBlock> globalMemoryBlocks;

  /// Memory storage for dynamically allocated memory (malloc).
  /// Maps address to memory block.
  llvm::DenseMap<uint64_t, MemoryBlock> mallocBlocks;

  /// Native memory regions returned by runtime helpers (e.g., assoc array refs).
  /// Maps base address to size in bytes.
  llvm::DenseMap<uint64_t, size_t> nativeMemoryBlocks;

  /// Interface field shadow signals. When an interface instance is allocated
  /// (malloc'd struct wrapped in llhd.sig), shadow signals are created for
  /// each field. This maps (mallocBaseAddr + fieldByteOffset) → SignalId.
  /// Used to bridge interface memory writes → LLHD signal events so that
  /// processes reading interface fields via GEP+load get proper sensitivity.
  llvm::DenseMap<uint64_t, SignalId> interfaceFieldSignals;

  /// True when any interface field signal uses an instance-scoped name
  /// (not top-level synthetic "sig_*"). Used to gate aggressive tri-state
  /// mirror suppression heuristics that are only safe for flat topologies.
  bool hasInstanceScopedInterfaceFieldSignals = false;

  /// Reverse map: signal ID → memory address for interface field signals.
  llvm::DenseMap<SignalId, uint64_t> fieldSignalToAddr;

  /// Per-interface-memory byte initialization mask keyed by malloc base address.
  /// This fixes coarse whole-block initialization for interface structs where
  /// only some fields are written: loads from untouched fields should return X.
  llvm::DenseMap<uint64_t, std::vector<uint8_t>> interfaceMemoryByteInitMask;

  /// Maps interface pointer signal ID → list of shadow field signal IDs.
  /// Used during sensitivity derivation: when a process probes an interface
  /// pointer signal, its field shadow signals are added to the sensitivity.
  llvm::DenseMap<SignalId, llvm::SmallVector<SignalId, 4>> interfacePtrToFieldSignals;

  struct DeferredInterfaceSensitivityExpansion {
    ProcessId procId = InvalidProcessId;
    llvm::SmallVector<SignalId, 4> sourceSignals;
  };
  llvm::SmallVector<DeferredInterfaceSensitivityExpansion, 32>
      deferredInterfaceSensitivityExpansions;

  /// Interface field signal propagation links. When a child BFM interface
  /// field is initialized by copying from a parent interface field, this maps
  /// parentFieldSignalId → [childFieldSignalIds...] so that when the parent
  /// field shadow signal changes, all child copies are also driven.
  llvm::DenseMap<SignalId, llvm::SmallVector<SignalId, 2>>
      interfaceFieldPropagation;

  /// Interface field signals that have synthetic intra-interface propagation
  /// links (e.g. txSclkOutput -> sclk). Used to limit cascade propagation
  /// in store handling and avoid double-propagation on normal BFM links.
  llvm::DenseSet<SignalId> intraLinkedSignals;

  /// Raw address-based tri-state candidates discovered from module-level
  /// interface stores before shadow signals are created.
  struct InterfaceTriStateCandidate {
    uint64_t condAddr = 0;
    uint64_t srcAddr = 0;
    uint64_t destAddr = 0;
    unsigned condBitIndex = 0;
    InterpretedValue elseValue;
  };
  llvm::SmallVector<InterfaceTriStateCandidate, 8> interfaceTriStateCandidates;

  /// Resolved signal-based tri-state rules:
  ///   dest = cond[condBitIndex] ? src : elseValue
  struct InterfaceTriStateRule {
    SignalId condSigId = 0;
    SignalId srcSigId = 0;
    SignalId destSigId = 0;
    unsigned condBitIndex = 0;
    InterpretedValue elseValue;
  };
  llvm::SmallVector<InterfaceTriStateRule, 8> interfaceTriStateRules;

  /// Reverse map to quickly find tri-state rules affected by a source update.
  llvm::DenseMap<SignalId, llvm::SmallVector<unsigned, 2>>
      interfaceTriStateRulesBySource;

  /// Cache for tri-state-backed drive source field detection.
  /// Key = hash(drive op pointer, instance ID), value = source field SignalId
  /// (or 0 when no tri-state-backed field source was detected).
  llvm::DenseMap<uint64_t, SignalId> triStateDriveSourceFieldCache;

  /// Last observed condition bit per tri-state destination field. Used by
  /// suppressed mirror-store handling to apply rule-derived release values only
  /// on cond true->false transitions (or first known false).
  llvm::DenseMap<SignalId, bool> interfaceTriStateCondLastValue;
  llvm::DenseSet<SignalId> interfaceTriStateCondSeen;

  /// Reverse map: childFieldAddr → parentFieldAddr, for setting up propagation.
  llvm::DenseMap<uint64_t, uint64_t> childToParentFieldAddr;

  /// Records (srcAddr, destAddr) pairs from child module-level stores that
  /// copy parent interface fields to child interface fields during init.
  /// Populated in executeChildModuleLevelOps(), consumed after
  /// createInterfaceFieldShadowSignals() to create propagation links.
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> childModuleCopyPairs;

  /// Records (srcSignalId, destAddr) pairs when module-level stores copy a
  /// probed LLHD signal value into an interface field (e.g. wire -> iface field).
  /// These are resolved after interface shadow signals are created so runtime
  /// signal changes can directly refresh all mirrored interface fields.
  llvm::SmallVector<std::pair<SignalId, uint64_t>, 8>
      interfaceSignalCopyPairs;

  /// Deferred child module-level ops: saved during initializeChildInstances(),
  /// executed after executeModuleLevelLLVMOps() so parent signal values are
  /// available when child modules probe them.
  struct DeferredChildModuleOps {
    hw::HWModuleOp childModule;
    InstanceId instanceId;
    InstanceInputMapping inputMap;
    DiscoveredOps childOps;
    hw::InstanceOp instOp;
  };
  llvm::SmallVector<DeferredChildModuleOps, 0> deferredChildModuleOps;

  /// Native analysis port connection map. Maps port object address to
  /// list of connected imp/export object addresses. Used to bypass UVM's
  /// "Late Connection" phase check that incorrectly rejects connect() calls
  /// during connect_phase.
  llvm::DenseMap<uint64_t, llvm::SmallVector<uint64_t, 2>> analysisPortConnections;

  /// Native sequencer item FIFO. Maps sequencer address to a queue of
  /// sequence item addresses pushed by finish_item() and consumed by
  /// seq_item_pull_port::get(). Implements a simple rendezvous between
  /// the sequence producer and driver consumer.
  llvm::DenseMap<uint64_t, std::deque<uint64_t>> sequencerItemFifo;

  struct SequencerGetWaiter {
    ProcessId procId = InvalidProcessId;
    mlir::Operation *retryOp = nullptr;
  };

  /// Processes blocked in get/get_next_item by resolved queue address.
  /// queueAddr==0 is the unresolved fallback bucket (wake on any push).
  llvm::DenseMap<uint64_t, std::deque<SequencerGetWaiter>>
      sequencerGetWaitersByQueue;

  /// Reverse map for fast waiter removal on process finalize/kill.
  llvm::DenseMap<ProcessId, uint64_t> sequencerGetWaitQueueByProc;

  /// Maps sequence item address to the sequencer address that owns it.
  /// Set during start_item() interception so finish_item() knows which
  /// sequencer FIFO to push the item into.
  llvm::DenseMap<uint64_t, uint64_t> itemToSequencer;
  uint64_t uvmSeqItemOwnerStores = 0;
  uint64_t uvmSeqItemOwnerErases = 0;
  uint64_t uvmSeqItemOwnerPeak = 0;

  /// Maps item address to the process waiting for item_done.
  /// Set during finish_item() when the sequence suspends waiting for the
  /// driver to complete the transaction via item_done().
  llvm::DenseMap<uint64_t, ProcessId> finishItemWaiters;

  /// Set of item addresses that have received item_done from the driver.
  /// Checked by finish_item poll to determine when to resume.
  llvm::DenseSet<uint64_t> itemDoneReceived;

  /// Maps pull-port / sequencer aliases to dequeued items not yet completed by
  /// item_done. A deque is required because some benches may dequeue multiple
  /// items before signaling completion.
  llvm::DenseMap<uint64_t, std::deque<uint64_t>> lastDequeuedItem;

  /// Outstanding dequeued items per driver process (in dequeue order).
  llvm::DenseMap<ProcessId, std::deque<uint64_t>> lastDequeuedItemByProc;

  /// Cache of resolved sequencer queue address per pull-port object.
  /// This avoids repeating expensive port->export->component resolution on
  /// every get/get_next_item call.
  llvm::DenseMap<uint64_t, uint64_t> portToSequencerQueue;
  uint64_t uvmSeqQueueCacheMaxEntries = 0;
  bool uvmSeqQueueCacheEvictOnCap = false;
  uint64_t uvmSeqQueueCacheHits = 0;
  uint64_t uvmSeqQueueCacheMisses = 0;
  uint64_t uvmSeqQueueCacheInstalls = 0;
  uint64_t uvmSeqQueueCacheCapacitySkips = 0;
  uint64_t uvmSeqQueueCacheEvictions = 0;

  /// Tracks valid associative array base addresses returned by __moore_assoc_create.
  /// Used to distinguish properly-initialized arrays from uninitialized class members.
  llvm::DenseSet<uint64_t> validAssocArrayAddresses;

  /// UVM config_db storage. Maps "{inst_name}.{field_name}" keys to
  /// opaque value bytes. Used by config_db_implementation_t interceptors.
  std::map<std::string, std::vector<uint8_t>> configDbEntries;

  /// Per-object rand_mode state. Key is "classPtr:propertyName", value is
  /// mode (1=enabled, 0=disabled). Default is enabled (1).
  std::map<std::string, int32_t> randModeState;

  /// Per-object constraint_mode state. Key is "classPtr:constraintName",
  /// value is mode (1=enabled, 0=disabled). Default is enabled (1).
  std::map<std::string, int32_t> constraintModeState;

  /// Per-object RNG state. Key is classPtr address, value is the RNG.
  /// Used by obj.srandom(seed) to seed per-object randomization.
  /// IEEE 1800-2017 §18.13: Each class instance has its own RNG state.
  std::map<uint64_t, std::mt19937> perObjectRng;

  /// UVM phase objection handles. Maps phase pointer → MooreObjectionHandle.
  /// Used by intercepted get_objection/raise_objection/drop_objection to
  /// bypass the interpreted get_objection() which fails on get_name() vtable
  /// dispatch during objection object creation.
  std::map<uint64_t, int64_t> phaseObjectionHandles;

  /// Processes blocked waiting for an objection handle to drop to zero.
  struct ObjectionWaiter {
    ProcessId procId;
    mlir::Operation *retryOp = nullptr;
  };
  llvm::DenseMap<int64_t, llvm::SmallVector<ObjectionWaiter, 4>>
      objectionZeroWaiters;
  llvm::DenseMap<ProcessId, int64_t> objectionWaitHandleByProc;

  /// Per-process state for uvm_objection::wait_for interception.
  /// Tracks whether a given handle was ever raised in this wait context and
  /// any active drain-time deadline in simulation time.
  struct ObjectionWaitForState {
    int64_t handle = -1;
    bool wasEverRaised = false;
    uint32_t zeroYields = 0;
    bool drainArmed = false;
    uint64_t drainDeadlineFs = 0;
  };
  std::map<ProcessId, ObjectionWaitForState> objectionWaitForStateByProc;

  /// Per-process mapping of execute_phase's task phase address.
  /// When execute_phase is intercepted for a task phase, the phase address
  /// is stored for the current process. When the outer sim.fork join creates
  /// a child, the mapping is propagated. The child's join_any handler uses
  /// it for objection polling.
  std::map<ProcessId, uint64_t> executePhaseBlockingPhaseMap;

  /// Maps task phase address → ProcessId of the master_phase_process child.
  /// Used by the join_any polling to check if the traversal is still running.
  std::map<uint64_t, ProcessId> masterPhaseProcessChild;

  /// Per-process yield counter for execute_phase objection polling grace period.
  std::map<ProcessId, int> executePhaseYieldCounts;
  /// Tracks whether a process has observed a positive objection count for
  /// the currently monitored execute_phase.
  std::map<ProcessId, bool> executePhaseSawPositiveObjection;

  /// Active execute_phase monitor-fork poll state keyed by process.
  std::map<ProcessId, uint64_t> executePhaseMonitorPollPhase;
  std::map<ProcessId, uint64_t> executePhaseMonitorPollToken;

  /// join_none + nearby disable_fork parent-resume gate state.
  std::map<ProcessId, ForkId> joinNoneDisableForkResumeFork;
  std::map<ProcessId, uint64_t> joinNoneDisableForkResumeToken;
  std::map<ProcessId, unsigned> joinNoneDisableForkResumePollCount;

  /// Snapshot of the parent function name when a fork child is created.
  /// Used by native-thunk policy to keep known-sensitive monitor sampling
  /// fork branches on the interpreter path.
  std::map<ProcessId, std::string> forkSpawnParentFunctionName;

  /// Deferred disable_fork state for children that may still be consuming
  /// pending wakeups in scheduler/interpreter state.
  std::map<ProcessId, uint64_t> disableForkDeferredToken;
  std::map<ProcessId, unsigned> disableForkDeferredPollCount;

  /// Per-process phase address currently being executed by the phase hopper.
  /// Set when execute_phase is entered, used by raise/drop_objection
  /// interceptors to associate objections with the correct phase.
  /// Must be per-process because multiple processes run phases concurrently.
  std::map<ProcessId, uint64_t> currentExecutingPhaseAddr;

  /// Native backing queue for uvm_phase_hopper methods.
  /// Key: phase hopper object pointer, value: queued phase pointers.
  llvm::DenseMap<uint64_t, std::deque<uint64_t>> phaseHopperQueue;

  /// Processes blocked in phase_hopper::get/peek waiting for queue data.
  /// Key: phase hopper object pointer, value: waiter process IDs.
  llvm::DenseMap<uint64_t, llvm::SmallVector<ProcessId, 4>> phaseHopperWaiters;

  /// Processes blocked in wait(condition) where the condition depends on
  /// __moore_queue_size(queue). Key: queue object pointer.
  struct QueueWaiter {
    ProcessId procId;
    mlir::Operation *retryOp = nullptr;
  };
  llvm::DenseMap<uint64_t, llvm::SmallVector<QueueWaiter, 4>>
      queueNotEmptyWaiters;

  /// Reverse index to remove queue waiters when processes complete/deopt.
  llvm::DenseMap<ProcessId, uint64_t> queueWaitAddrByProc;

  /// Function phase IMP sequencing: tracks which function phase IMP nodes
  /// have completed their traversal. The UVM phase graph IMP nodes for
  /// function phases (build, connect, end_of_elaboration, start_of_simulation)
  /// may not have predecessor relationships set, so we enforce ordering
  /// natively. Key = phase IMP address, Value = true when completed.
  std::map<uint64_t, bool> functionPhaseImpCompleted;

  /// Maps phase IMP address to its sequence index.
  /// build=0, connect=1, end_of_elaboration=2, start_of_simulation=3
  std::map<uint64_t, int> functionPhaseImpOrder;

  /// Ordered list of function phase IMP addresses (populated at runtime).
  std::vector<uint64_t> functionPhaseImpSequence;

  /// Processes waiting for IMP ordering: maps the IMP address they're waiting
  /// on to a list of {procId, resumeOp} pairs. When finish_phase marks an IMP
  /// as completed, we wake up all processes in the corresponding wait list.
  /// For task phases waiting on "all function phases", key = 0 (sentinel).
  struct ImpWaiter {
    ProcessId procId;
    mlir::Block::iterator resumeOp;
  };
  std::map<uint64_t, std::vector<ImpWaiter>> impWaitingProcesses;

  /// Get the per-object RNG for the given object address.
  /// If the object was explicitly seeded (via obj.srandom()), returns its
  /// per-object RNG.  Otherwise, per IEEE 1800-2017 §18.13, falls back to the
  /// calling process's RNG so that process::get_randstate/set_randstate can
  /// capture and replay the randomization sequence.
  /// When procId is InvalidProcessId (e.g. class_get_randstate), a per-object
  /// RNG is auto-created as a last resort.
  std::mt19937 &getObjectRng(uint64_t objAddr,
                              ProcessId procId = InvalidProcessId) {
    auto it = perObjectRng.find(objAddr);
    if (it != perObjectRng.end())
      return it->second;
    // Fall back to process RNG when available (IEEE 1800-2017 §18.13)
    if (procId != InvalidProcessId) {
      auto procIt = processStates.find(procId);
      if (procIt != processStates.end())
        return procIt->second.randomGenerator;
    }
    // Last resort: create per-object RNG seeded with object address
    auto [insertIt, _] = perObjectRng.emplace(
        objAddr, std::mt19937(static_cast<uint32_t>(objAddr)));
    return insertIt->second;
  }

  /// The object address of the last __moore_randomize_basic call.
  /// Used to associate subsequent __moore_randomize_with_range/ranges calls
  /// with the same per-object RNG, since those calls don't receive the
  /// object pointer directly.
  uint64_t lastRandomizeObjAddr = 0;

  /// Pending seed from old-style @srandom() stubs (no object pointer).
  /// When set, the next __moore_randomize_basic call will use this seed
  /// instead of the default per-object seed.
  std::optional<uint32_t> pendingSrandomSeed;

  /// Tracks active function call depth for recursion detection.
  /// Maps funcOp operation pointer to current call depth. Used to enable
  /// save/restore of SSA values only for recursive calls (depth > 1).
  llvm::DenseMap<mlir::Operation *, unsigned> funcCallDepth;

  /// Global address counter for malloc and cross-process memory.
  /// This ensures no address overlap between different processes.
  uint64_t globalNextAddress = 0x100000;

  /// Memory storage for module-level (hw.module body) allocas.
  /// These are allocas defined outside of llhd.process blocks but inside
  /// the hw.module body, accessible by all processes in the module.
  llvm::DenseMap<mlir::Value, MemoryBlock> moduleLevelAllocas;

  /// Stable base address for each module-level alloca Value.
  /// Kept separate from moduleInitValueMap so address-based lookups do not
  /// depend on transient SSA value-map merges.
  llvm::DenseMap<mlir::Value, uint64_t> moduleLevelAllocaBaseAddr;

  /// Value map for module-level initialization values.
  /// This stores values computed during executeModuleLevelLLVMOps() so
  /// processes can access values defined at module level.
  llvm::DenseMap<mlir::Value, InterpretedValue> moduleInitValueMap;

  /// Per-instance module-level initialization values for child modules.
  /// Child modules can be instantiated multiple times, and their module-level
  /// SSA values are instance-specific. Storing them by instance avoids
  /// collisions in moduleInitValueMap (which is shared across top modules).
  llvm::DenseMap<InstanceId, llvm::DenseMap<mlir::Value, InterpretedValue>>
      instanceModuleInitValueMaps;

  /// Map from simulated addresses to function names (for vtable entries).
  /// When a vtable entry is loaded, we store the function name it maps to.
  llvm::DenseMap<uint64_t, std::string> addressToFunction;

  /// Map from global variable names to their simulated base addresses.
  llvm::StringMap<uint64_t> globalAddresses;

  /// Reverse map from simulated addresses to global variable names.
  /// Used for looking up string content by virtual address (e.g., in fmt.dyn_string).
  llvm::DenseMap<uint64_t, std::string> addressToGlobal;

  /// Native UVM factory type registry. Maps type name → wrapper address.
  /// Populated by the fast-path factory.register interceptor (which calls
  /// get_type_name once instead of 7 times). Used by find_wrapper_by_name.
  llvm::StringMap<uint64_t> nativeFactoryTypeNames;

  /// Tracks wrappers whose *_registry_*::initialize fast path has run.
  /// Avoids repeated initialization work for the same wrapper object.
  llvm::DenseSet<uint64_t> nativeFactoryInitializedWrappers;

  /// Tracks deduplicated uvm_phase::add edges (hash of call arguments).
  /// Used by a function-body fast path to elide duplicate add calls.
  llvm::DenseSet<uint64_t> nativePhaseAddEdgeKeys;

  /// Tracks deduplicated uvm_phase::add call sites (hash of call arguments).
  /// Used by func.call fast path to skip duplicate calls before interpretation.
  llvm::DenseSet<uint64_t> nativePhaseAddCallKeys;

  /// Native random seed table for uvm_create_random_seed fast-path.
  /// Maps inst_id → (type_id → (seed, count)).
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::pair<uint32_t, uint32_t>>>
      nativeRandomSeedTable;

  /// Next available address for global memory allocation.
  uint64_t nextGlobalAddress = 0x10000000;

  /// Address range index for O(log n) global/malloc address lookups.
  /// Maps base address to (endAddr, globalName). Populated lazily after
  /// global initialization, replaces O(n) linear scans through globalAddresses.
  struct AddrRangeEntry {
    uint64_t endAddr;
    llvm::StringRef globalName; // empty for malloc blocks
    bool isMalloc;
  };
  std::map<uint64_t, AddrRangeEntry> addrRangeIndex;
  bool addrRangeIndexDirty = true;

  /// Rebuild the address range index from globalAddresses and mallocBlocks.
  void rebuildAddrRangeIndex();

  /// Register a new malloc-backed address range when the index is already
  /// materialized. This avoids full index rebuilds on every runtime allocation.
  void noteMallocBlockAllocated(uint64_t baseAddr, uint64_t size);

  /// Create shadow LLHD signals for interface struct fields.
  /// Called after module-level ops are executed, when interface malloc addresses
  /// are known. Scans for llhd.sig ops whose init is a malloc'd pointer and
  /// creates per-field shadow signals based on GEP usage patterns.
  void createInterfaceFieldShadowSignals();

  /// Queue a process for post-init interface-pointer sensitivity expansion.
  /// Continuous-assignment processes are registered before interface shadow
  /// signals exist, so they may only be sensitive to pointer-holding signals.
  /// After createInterfaceFieldShadowSignals(), this queue is expanded to
  /// include field-level sensitivities.
  void queueDeferredInterfaceSensitivityExpansion(
      ProcessId procId, llvm::ArrayRef<SignalId> sourceSignals);

  /// Expand deferred pointer sensitivities to interface field sensitivities.
  void expandDeferredInterfaceSensitivityExpansions();

  /// Find a global or malloc memory block by address using the range index.
  /// Returns the MemoryBlock pointer and sets offset, or nullptr if not found.
  MemoryBlock *findBlockByAddress(uint64_t addr, uint64_t &offset);

  /// Read an object vtable pointer from the standard runtime object header
  /// layout ([i32 class_id][ptr vtable_ptr]...) across both interpreter-managed
  /// and native memory-backed allocations.
  bool readObjectVTableAddress(uint64_t objectAddr, uint64_t &vtableAddr);

  // Return true when a function name corresponds to AHB monitor sampling
  // logic where field provenance tracing should be enabled.
  static bool isAhbMonitorSampleFunctionForTrace(llvm::StringRef funcName);

  // Read raw bytes from interpreter/native tracked memory for trace diagnostics.
  bool readRawMemoryBytesForTrace(uint64_t addr, size_t size,
                                  llvm::SmallVectorImpl<uint8_t> &out);

  // Optional AHB transaction payload trace at analysis write/dispatch sites.
  void traceAhbTxnPayload(ProcessId procId, llvm::StringRef stage,
                          llvm::StringRef calleeName,
                          llvm::StringRef impName, uint64_t portAddr,
                          uint64_t txnAddr);

  /// Resolve the base address for a module-level alloca value.
  uint64_t getModuleLevelAllocaBaseAddress(mlir::Value value) const;

  //===--------------------------------------------------------------------===//
  // UVM Root Re-entrancy Support
  //===--------------------------------------------------------------------===//
  //
  // The UVM library has a re-entrancy issue in the uvm_root singleton pattern:
  // - m_uvm_get_root() checks if m_inst == null, and if so, creates uvm_root
  // - uvm_root::new() sets m_inst = this BEFORE returning
  // - uvm_root::new() calls uvm_component::new(), which calls get_root()
  // - The re-entrant call sees m_inst != null but uvm_top is still null
  // - This causes a false "uvm_top has been overwritten" warning
  //
  // We fix this by tracking when m_uvm_get_root is executing and skipping
  // the m_inst != uvm_top check during re-entrant calls.

  /// Depth of m_uvm_get_root calls (0 = not in get_root, >1 = re-entrant).
  size_t uvmGetRootDepth = 0;

  /// Count of observed UVM run_test entries this simulation.
  uint64_t uvmRunTestEntryCount = 0;


  /// The uvm_root instance being constructed (simulated address).
  uint64_t uvmRootBeingConstructed = 0;

  //===--------------------------------------------------------------------===//
  // RTTI Parent Table (for $cast hierarchy checking)
  //===--------------------------------------------------------------------===//

  /// Map from typeId -> parentTypeId. table[typeId] = parentTypeId; 0 = root.
  llvm::SmallVector<int32_t> rttiParentTable;

  /// Whether the RTTI parent table has been loaded from the module.
  bool rttiTableLoaded = false;

  /// Load the RTTI parent table from the module's circt.rtti_parent_table attr.
  void loadRTTIParentTable();

  /// Check if srcTypeId is the same as or derived from targetTypeId,
  /// using the RTTI parent table to walk the class hierarchy.
  bool checkRTTICast(int32_t srcTypeId, int32_t targetTypeId);

  /// Initialize LLVM global variables, especially vtables.
  mlir::LogicalResult initializeGlobals(const DiscoveredGlobalOps &globalOps);

  /// Execute LLVM global constructors (llvm.mlir.global_ctors).
  /// This calls functions like __moore_global_init_* that initialize
  /// UVM globals (e.g., uvm_top) at simulation startup.
  mlir::LogicalResult
  executeGlobalConstructors(const DiscoveredGlobalOps &globalOps);

  /// Cached global ops discovered during first initialize() call, reused by
  /// finalizeInit() to execute global constructors.
  DiscoveredGlobalOps cachedGlobalOps;
  /// True once LLVM globals have been allocated and their addresses assigned.
  /// This keeps addressof results stable across multiple initializeModule()
  /// calls in a single simulation.
  bool globalsInitialized = false;

  /// Execute module-level LLVM operations (alloca, call, store) that are
  /// defined in the hw.module body but outside of llhd.process blocks.
  /// This initializes module-level string variables and other dynamic state.
  mlir::LogicalResult executeModuleLevelLLVMOps(hw::HWModuleOp hwModule);

  /// Execute deferred child module-level LLVM ops. Called after
  /// executeModuleLevelLLVMOps() so parent signal values (from malloc etc.)
  /// are available when child modules probe parent signals.
  void executeChildModuleLevelOps();

  /// Interpret an llvm.mlir.addressof operation.
  mlir::LogicalResult interpretLLVMAddressOf(ProcessId procId,
                                              mlir::LLVM::AddressOfOp addrOfOp);
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETER_H
