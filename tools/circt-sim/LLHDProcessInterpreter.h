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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

/// Execution state for an LLHD process or seq.initial block being interpreted.
struct ProcessExecutionState {
  /// The process operation being executed (either llhd.process or seq.initial).
  mlir::Operation *processOrInitialOp = nullptr;

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

  /// Flag indicating whether the process is waiting.
  bool waiting = false;

  /// Flag indicating this is a seq.initial block (runs once at time 0).
  bool isInitialBlock = false;

  /// The next block to branch to after a wait.
  mlir::Block *destBlock = nullptr;

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
  mlir::LogicalResult initializeChildInstances(const DiscoveredOps &ops);

  /// Get the signal ID for an MLIR value (signal reference).
  SignalId getSignalId(mlir::Value signalRef) const;

  /// Get the signal name for a signal ID.
  llvm::StringRef getSignalName(SignalId id) const;

  /// Get the number of registered signals.
  size_t getNumSignals() const { return valueToSignal.size(); }

  /// Get the number of registered processes.
  size_t getNumProcesses() const { return processStates.size(); }

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

  /// Get the bit width of a type. Made public for use by helper functions.
  static unsigned getTypeWidth(mlir::Type type);

private:
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
  void registerModuleDrive(llhd::DriveOp driveOp);

  /// Execute module-level drives for a process after it yields.
  void executeModuleDrives(ProcessId procId);

  /// Execute instance output updates that depend on a process result.
  void executeInstanceOutputUpdates(ProcessId procId);

  /// Register combinational processes for static module-level drives.
  /// These drives need to re-execute when their input signals change.
  void registerContinuousAssignments(hw::HWModuleOp hwModule);

  /// Register seq.firreg operations using pre-discovered operations.
  void registerFirRegs(const DiscoveredOps &ops);

  /// Execute a single seq.firreg register update.
  void executeFirReg(seq::FirRegOp regOp);

  /// Resolve a signal ID from an arbitrary value.
  SignalId resolveSignalId(mlir::Value value) const;

  /// Collect signal IDs referenced by a value expression.
  void collectSignalIds(mlir::Value value,
                        llvm::SmallVectorImpl<SignalId> &signals) const;

  /// Collect process IDs that a value depends on (via llhd.process results).
  void collectProcessIds(mlir::Value value,
                         llvm::SmallVectorImpl<ProcessId> &processIds) const;

  /// Collect signal IDs referenced inside an llhd.combinational op.
  void collectSignalIdsFromCombinational(
      llhd::CombinationalOp combOp,
      llvm::SmallVectorImpl<SignalId> &signals) const;

  /// Execute a single continuous assignment (static module-level drive).
  void executeContinuousAssignment(llhd::DriveOp driveOp);

  /// Schedule a combinational update of an instance output signal.
  void scheduleInstanceOutputUpdate(SignalId signalId, mlir::Value outputValue);

  /// Evaluate a value for continuous assignments by reading from signal state.
  InterpretedValue evaluateContinuousValue(mlir::Value value);

  /// Helper that tracks the recursion stack to detect cycles.
  InterpretedValue evaluateContinuousValueImpl(
      mlir::Value value, llvm::DenseSet<mlir::Value> &inProgress);

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
  mlir::LogicalResult
  interpretFuncBody(ProcessId procId, mlir::func::FuncOp funcOp,
                    llvm::ArrayRef<InterpretedValue> args,
                    llvm::SmallVectorImpl<InterpretedValue> &results);

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

  /// Maximum number of ops a process may execute before forcing a stop.
  size_t maxProcessSteps = 50000;

  /// Name of the top module (for hierarchical path construction).
  std::string moduleName;

  /// Root MLIR module (for symbol table lookups).
  mlir::ModuleOp rootModule;

  /// Set of module names that have already been processed (to avoid
  /// processing the same module multiple times for different instances).
  llvm::StringSet<> processedModules;

  /// Map from MLIR signal values to signal IDs.
  llvm::DenseMap<mlir::Value, SignalId> valueToSignal;

  /// Map from instance result values to child module output values.
  llvm::DenseMap<mlir::Value, mlir::Value> instanceOutputMap;

  struct InstanceOutputUpdate {
    SignalId signalId = 0;
    mlir::Value outputValue;
    llvm::SmallVector<ProcessId, 4> processIds;
  };
  llvm::SmallVector<InstanceOutputUpdate, 16> instanceOutputUpdates;

  /// Map from child module input block arguments to instance operand values.
  llvm::DenseMap<mlir::Value, mlir::Value> inputValueMap;

  /// Map from signal IDs to signal names.
  llvm::DenseMap<SignalId, std::string> signalIdToName;

  /// Pending epsilon drives - for immediate blocking assignment semantics.
  /// When a signal is driven with epsilon delay within the same process,
  /// this map holds the value so subsequent probes can see it immediately.
  /// Key is SignalId, value is the pending value.
  llvm::DenseMap<SignalId, InterpretedValue> pendingEpsilonDrives;

  /// Map from process IDs to execution states.
  llvm::DenseMap<ProcessId, ProcessExecutionState> processStates;

  /// Next temporary process ID used for combinational evaluation.
  ProcessId nextTempProcId = 1ull << 60;

  /// Map from llhd.process ops to process IDs.
  llvm::DenseMap<mlir::Operation *, ProcessId> opToProcessId;

  /// Operation execution statistics (by op name).
  llvm::StringMap<uint64_t> opStats;

  bool collectOpStats = false;

  /// Module-level drives connected to process results.
  /// Each entry is a pair of (DriveOp, ProcessId).
  llvm::SmallVector<std::pair<llhd::DriveOp, ProcessId>, 4> moduleDrives;

  /// Static module-level drives (not connected to process results).
  llvm::SmallVector<llhd::DriveOp, 4> staticModuleDrives;

  struct FirRegState {
    SignalId signalId = 0;
    InterpretedValue prevClock;
    bool hasPrevClock = false;
  };

  /// Registered seq.firreg state keyed by op.
  llvm::DenseMap<mlir::Operation *, FirRegState> firRegStates;

  /// Callback for sim.terminate operation.
  std::function<void(bool, bool)> terminateCallback;

  /// Flag indicating if termination has been requested.
  bool terminationRequested = false;

  /// Map of dynamic string pointers to their content (for runtime string
  /// handling). Key is the pointer value, value is {data, len}.
  llvm::DenseMap<int64_t, std::pair<const char *, int64_t>> dynamicStrings;

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
