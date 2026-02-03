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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <deque>
#include <map>
#include <optional>
#include <random>

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
  /// The function being executed.
  mlir::func::FuncOp funcOp;

  /// The block within the function where execution should resume.
  mlir::Block *resumeBlock = nullptr;

  /// The operation iterator within the block where execution should resume.
  mlir::Block::iterator resumeOp;

  /// The call operation that invoked this function (for setting results).
  mlir::Operation *callOp = nullptr;

  /// Arguments passed to the function (for re-entry if needed).
  llvm::SmallVector<InterpretedValue, 4> args;

  CallStackFrame() = default;
  CallStackFrame(mlir::func::FuncOp func, mlir::Block *block,
                 mlir::Block::iterator op, mlir::Operation *call)
      : funcOp(func), resumeBlock(block), resumeOp(op), callOp(call) {}
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

  /// Last operation executed by this process (for diagnostics).
  mlir::Operation *lastOp = nullptr;

  /// Total operations executed by this process.
  size_t totalSteps = 0;

  /// Total operations in this process/initial body (used for step budget).
  size_t opCount = 0;

  /// Cached sensitivity list entries for combinational wait reuse.
  llvm::SmallVector<SensitivityEntry, 4> lastSensitivityEntries;

  /// Cached values for the last sensitivity list.
  llvm::SmallVector<SignalValue, 4> lastSensitivityValues;

  /// Cache of derived wait sensitivities keyed by wait operations.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<SensitivityEntry, 4>>
      waitSensitivityCache;

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

  /// Call stack for resuming execution after a wait inside a function.
  /// When a wait (e.g., sim.fork with blocking join) occurs inside a nested
  /// function call, we push the function's context onto this stack so that
  /// when the process resumes, we can continue from the correct point inside
  /// the function rather than skipping to the next process-level operation.
  llvm::SmallVector<CallStackFrame, 4> callStack;

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

  /// Parent process ID for shared memory in fork/join.
  /// When a child process is created by sim.fork, parent-scope allocas are
  /// accessed through this chain rather than deep-copied.  Only allocas
  /// defined within the fork body are local to the child.
  ProcessId parentProcessId = 0;

  ProcessExecutionState() = default;
  explicit ProcessExecutionState(llhd::ProcessOp op)
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

  /// Initialize the interpreter for a hardware module.
  /// This walks the module to find all signals and processes,
  /// including those in child module instances.
  mlir::LogicalResult initialize(hw::HWModuleOp hwModule);

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

  /// Get the number of registered signals.
  size_t getNumSignals() const { return valueToSignal.size(); }

  /// Get the number of registered processes.
  size_t getNumProcesses() const { return processStates.size(); }

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

  /// Enable collection of operation execution statistics.
  void setCollectOpStats(bool enable) { collectOpStats = enable; }

  /// Set a callback to be called when sim.terminate is executed.
  /// The callback receives (success, verbose) parameters.
  void setTerminateCallback(std::function<void(bool, bool)> callback) {
    terminateCallback = std::move(callback);
  }

  /// Check if termination has been requested.
  bool isTerminationRequested() const { return terminationRequested; }

  /// Set a callback to check if abort has been requested (e.g., by timeout).
  void setShouldAbortCallback(std::function<bool()> callback) {
    shouldAbortCallback = std::move(callback);
  }

  /// Set a callback to invoke when abort is triggered.
  void setAbortCallback(std::function<void()> callback) {
    abortCallback = std::move(callback);
  }

  /// Check if abort has been requested (calls the shouldAbortCallback).
  bool isAbortRequested() const {
    return shouldAbortCallback && shouldAbortCallback();
  }

  /// Get the bit width of a type. Made public for use by helper functions.
  static unsigned getTypeWidth(mlir::Type type);
  /// Determine signal encoding based on the type.
  static SignalEncoding getSignalEncoding(mlir::Type type);

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

  /// Register a single initial block from a seq.initial operation.
  ProcessId registerInitialBlock(seq::InitialOp initialOp);

  /// Register a module-level drive operation.
  void registerModuleDrive(llhd::DriveOp driveOp, InstanceId instanceId,
                           const InstanceInputMapping &inputMap);

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

  /// Resolve a signal ID from an arbitrary value.
  SignalId resolveSignalId(mlir::Value value) const;

  /// Get the stored signal value type for a signal ID (nested type).
  mlir::Type getSignalValueType(SignalId sigId) const;

  /// Convert aggregate values between LLVM and HW layout conventions.
  llvm::APInt convertLLVMToHWLayout(llvm::APInt value, mlir::Type llvmType,
                                    mlir::Type hwType) const;
  llvm::APInt convertHWToLLVMLayout(llvm::APInt value, mlir::Type hwType,
                                    mlir::Type llvmType) const;
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

  /// Execute a single continuous assignment (static module-level drive).
  void executeContinuousAssignment(llhd::DriveOp driveOp);

  /// Schedule a combinational update of an instance output signal.
  void scheduleInstanceOutputUpdate(
      SignalId signalId, mlir::Value outputValue, InstanceId instanceId,
      const InstanceInputMapping *inputMap);

  /// Evaluate a value for continuous assignments by reading from signal state.
  InterpretedValue evaluateContinuousValue(mlir::Value value);

  /// Helper for iterative continuous-value evaluation with cycle detection.
  InterpretedValue evaluateContinuousValueImpl(mlir::Value value);

  /// Evaluate an llhd.combinational op and return its yielded values.
  bool evaluateCombinationalOp(llhd::CombinationalOp combOp,
                               llvm::SmallVectorImpl<InterpretedValue> &results);

  //===--------------------------------------------------------------------===//
  // Process Execution
  //===--------------------------------------------------------------------===//

  /// Execute one step of a process (interpret one operation).
  /// Returns true if the process should continue, false if suspended/halted.
  bool executeStep(ProcessId procId);

  /// Execute a process until it suspends or halts.
  void executeProcess(ProcessId procId);

  /// Resume a process after a wait condition is satisfied.
  void resumeProcess(ProcessId procId);

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

  //===--------------------------------------------------------------------===//
  // Sim Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

  /// Interpret a sim.proc.print operation.
  mlir::LogicalResult interpretProcPrint(ProcessId procId,
                                          sim::PrintFormattedProcOp printOp);

  /// Interpret a sim.terminate operation.
  mlir::LogicalResult interpretTerminate(ProcessId procId,
                                          sim::TerminateOp terminateOp);

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

  //===--------------------------------------------------------------------===//
  // Seq Dialect Operation Handlers
  //===--------------------------------------------------------------------===//

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

  /// Interpret an llvm.getelementptr operation.
  mlir::LogicalResult interpretLLVMGEP(ProcessId procId,
                                        mlir::LLVM::GEPOp gepOp);

  /// Interpret an llvm.call operation.
  mlir::LogicalResult interpretLLVMCall(ProcessId procId,
                                         mlir::LLVM::CallOp callOp);

  /// Interpret an LLVM function body.
  /// If callOperands is provided, signal mappings are created for BlockArguments
  /// when the corresponding call operand resolves to a signal ID.
  mlir::LogicalResult
  interpretLLVMFuncBody(ProcessId procId, mlir::LLVM::LLVMFuncOp funcOp,
                        llvm::ArrayRef<InterpretedValue> args,
                        llvm::SmallVectorImpl<InterpretedValue> &results,
                        llvm::ArrayRef<mlir::Value> callOperands = {});

  /// Get the size in bytes for an LLVM type.
  unsigned getLLVMTypeSize(mlir::Type type);

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
  void registerProcessState(ProcessId procId, ProcessExecutionState &&state);

  /// Resume any processes awaiting completion of the target process.
  void notifyProcessAwaiters(ProcessId procId);

  /// Finalize a process (finished or killed) and wake awaiters.
  void finalizeProcess(ProcessId procId, bool killed);

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
  size_t maxProcessSteps = 50000;

  /// Name of the top module (for hierarchical path construction).
  std::string moduleName;

  /// Active instance context for signal/process resolution.
  InstanceId activeInstanceId = 0;

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

  /// Map from process IDs to execution states.
  /// NOTE: This uses std::map rather than DenseMap to guarantee reference
  /// stability.  evaluateCombinationalOp inserts and erases temporary entries
  /// during getValue calls; with DenseMap this can trigger a rehash and
  /// invalidate references held by callers (e.g., interpretWait's
  /// `auto &state = processStates[procId]`).  std::map provides stable
  /// references across inserts and erases.
  std::map<ProcessId, ProcessExecutionState> processStates;

  /// Map from process handle values to process IDs.
  llvm::DenseMap<uint64_t, ProcessId> processHandleToId;

  /// Map from process IDs to lists of awaiters.
  llvm::DenseMap<ProcessId, llvm::SmallVector<ProcessId, 4>> processAwaiters;

  /// Next temporary process ID used for combinational evaluation.
  ProcessId nextTempProcId = 1ull << 60;

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

  /// Static module-level drives (not connected to process results).
  struct StaticModuleDrive {
    llhd::DriveOp driveOp;
    InstanceId instanceId = 0;
    InstanceInputMapping inputMap;
  };
  llvm::SmallVector<StaticModuleDrive, 4> staticModuleDrives;

  /// Memory event waiter for polling-based event detection.
  /// Used for UVM events stored as boolean fields in class instances where
  /// no signal is available to wait on.
  struct MemoryEventWaiter {
    /// The address of the memory location to poll.
    uint64_t address = 0;
    /// The last seen value at this address.
    uint64_t lastValue = 0;
    /// The size of the value in bytes (1 for bool/i1).
    unsigned valueSize = 1;
    /// True if we're waiting for a rising edge (0→1) trigger.
    /// For event types (!moore.event), we only wake on 0→1 transitions.
    bool waitForRisingEdge = false;
  };

  /// Map from process IDs to their memory event waiters.
  /// A process waiting on a memory event will have an entry here.
  llvm::DenseMap<ProcessId, MemoryEventWaiter> memoryEventWaiters;

  /// Check all memory event waiters and wake processes whose watched
  /// memory location has changed.
  void checkMemoryEventWaiters();

  /// Registered seq.firreg state keyed by op.
  llvm::DenseMap<mlir::Operation *, FirRegState> firRegStates;

  /// Callback for sim.terminate operation.
  std::function<void(bool, bool)> terminateCallback;

  /// Callback to check if abort has been requested.
  std::function<bool()> shouldAbortCallback;

  /// Callback to invoke when abort is triggered.
  std::function<void()> abortCallback;

  /// Flag indicating if termination has been requested.
  bool terminationRequested = false;
  /// Whether we are currently in global initialization (constructors, etc.).
  /// During this phase, sim.terminate should not halt the process to allow
  /// UVM initialization to complete (re-entrant calls set uvm_top properly).
  bool inGlobalInit = false;

  /// Map of dynamic string pointers to their content (for runtime string
  /// handling). Key is the pointer value, value is {data, len}.
  llvm::DenseMap<int64_t, std::pair<const char *, int64_t>> dynamicStrings;

  /// Persistent storage for strings created by the interpreter (e.g. from
  /// __moore_int_to_string and __moore_string_concat). The deque ensures
  /// stable pointers so that dynamicStrings entries remain valid.
  std::deque<std::string> interpreterStrings;

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

  /// Tracks valid associative array base addresses returned by __moore_assoc_create.
  /// Used to distinguish properly-initialized arrays from uninitialized class members.
  llvm::DenseSet<uint64_t> validAssocArrayAddresses;

  /// Global address counter for malloc and cross-process memory.
  /// This ensures no address overlap between different processes.
  uint64_t globalNextAddress = 0x100000;

  /// Memory storage for module-level (hw.module body) allocas.
  /// These are allocas defined outside of llhd.process blocks but inside
  /// the hw.module body, accessible by all processes in the module.
  llvm::DenseMap<mlir::Value, MemoryBlock> moduleLevelAllocas;

  /// Value map for module-level initialization values.
  /// This stores values computed during executeModuleLevelLLVMOps() so
  /// processes can access values defined at module level.
  llvm::DenseMap<mlir::Value, InterpretedValue> moduleInitValueMap;

  /// Map from simulated addresses to function names (for vtable entries).
  /// When a vtable entry is loaded, we store the function name it maps to.
  llvm::DenseMap<uint64_t, std::string> addressToFunction;

  /// Map from global variable names to their simulated base addresses.
  llvm::StringMap<uint64_t> globalAddresses;

  /// Reverse map from simulated addresses to global variable names.
  /// Used for looking up string content by virtual address (e.g., in fmt.dyn_string).
  llvm::DenseMap<uint64_t, std::string> addressToGlobal;

  /// Next available address for global memory allocation.
  uint64_t nextGlobalAddress = 0x10000000;

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

  /// The uvm_root instance being constructed (simulated address).
  uint64_t uvmRootBeingConstructed = 0;

  /// Initialize LLVM global variables, especially vtables.
  mlir::LogicalResult initializeGlobals();

  /// Execute LLVM global constructors (llvm.mlir.global_ctors).
  /// This calls functions like __moore_global_init_* that initialize
  /// UVM globals (e.g., uvm_top) at simulation startup.
  mlir::LogicalResult executeGlobalConstructors();

  /// Execute module-level LLVM operations (alloca, call, store) that are
  /// defined in the hw.module body but outside of llhd.process blocks.
  /// This initializes module-level string variables and other dynamic state.
  mlir::LogicalResult executeModuleLevelLLVMOps(hw::HWModuleOp hwModule);

  /// Interpret an llvm.mlir.addressof operation.
  mlir::LogicalResult interpretLLVMAddressOf(ProcessId procId,
                                              mlir::LLVM::AddressOfOp addrOfOp);
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETER_H
