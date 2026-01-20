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
#include "circt/Dialect/Sim/EventQueue.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

// Forward declarations for SCF and Func dialects
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

  /// Get as uint64_t (only valid for widths <= 64).
  uint64_t getUInt64() const {
    return isUnknown ? 0 : value.getZExtValue();
  }

  /// Convert to SignalValue for use with ProcessScheduler.
  SignalValue toSignalValue() const {
    if (isUnknown)
      return SignalValue::makeX(width);
    return SignalValue(value.getZExtValue(), width);
  }

  /// Create from a SignalValue.
  static InterpretedValue fromSignalValue(const SignalValue &sv) {
    if (sv.isUnknown())
      return makeX(sv.getWidth());
    return InterpretedValue(sv.getValue(), sv.getWidth());
  }

private:
  llvm::APInt value;
  bool isUnknown;
  unsigned width;
};

//===----------------------------------------------------------------------===//
// ProcessExecutionState - Per-process execution state
//===----------------------------------------------------------------------===//

/// Execution state for an LLHD process being interpreted.
struct ProcessExecutionState {
  /// The process operation being executed.
  llhd::ProcessOp processOp;

  /// Current basic block being executed.
  mlir::Block *currentBlock;

  /// Iterator to the current operation within the block.
  mlir::Block::iterator currentOp;

  /// SSA value map: maps MLIR Values to their interpreted runtime values.
  llvm::DenseMap<mlir::Value, InterpretedValue> valueMap;

  /// Flag indicating whether the process has halted.
  bool halted = false;

  /// Flag indicating whether the process is waiting.
  bool waiting = false;

  /// The next block to branch to after a wait.
  mlir::Block *destBlock = nullptr;

  /// Operands to pass to the destination block after a wait.
  llvm::SmallVector<InterpretedValue, 4> destOperands;

  ProcessExecutionState() = default;
  explicit ProcessExecutionState(llhd::ProcessOp op)
      : processOp(op), currentBlock(nullptr), halted(false), waiting(false) {}
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
  /// This walks the module to find all signals and processes.
  mlir::LogicalResult initialize(hw::HWModuleOp hwModule);

  /// Get the signal ID for an MLIR value (signal reference).
  SignalId getSignalId(mlir::Value signalRef) const;

  /// Get the signal name for a signal ID.
  llvm::StringRef getSignalName(SignalId id) const;

  /// Get the number of registered signals.
  size_t getNumSignals() const { return valueToSignal.size(); }

  /// Get the number of registered processes.
  size_t getNumProcesses() const { return processStates.size(); }

private:
  //===--------------------------------------------------------------------===//
  // Signal Registration
  //===--------------------------------------------------------------------===//

  /// Register all signals from the module.
  mlir::LogicalResult registerSignals(hw::HWModuleOp hwModule);

  /// Register a single signal from an llhd.sig operation.
  SignalId registerSignal(llhd::SignalOp sigOp);

  //===--------------------------------------------------------------------===//
  // Process Registration
  //===--------------------------------------------------------------------===//

  /// Register all processes from the module.
  mlir::LogicalResult registerProcesses(hw::HWModuleOp hwModule);

  /// Register a single process from an llhd.process operation.
  ProcessId registerProcess(llhd::ProcessOp processOp);

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
  // Value Management
  //===--------------------------------------------------------------------===//

  /// Get the interpreted value for an SSA value.
  InterpretedValue getValue(ProcessId procId, mlir::Value value);

  /// Set the interpreted value for an SSA value.
  void setValue(ProcessId procId, mlir::Value value, InterpretedValue val);

  /// Get the bit width of a type.
  static unsigned getTypeWidth(mlir::Type type);

  //===--------------------------------------------------------------------===//
  // Member Variables
  //===--------------------------------------------------------------------===//

  /// Reference to the process scheduler.
  ProcessScheduler &scheduler;

  /// Map from MLIR signal values to signal IDs.
  llvm::DenseMap<mlir::Value, SignalId> valueToSignal;

  /// Map from signal IDs to signal names.
  llvm::DenseMap<SignalId, std::string> signalIdToName;

  /// Map from process IDs to execution states.
  llvm::DenseMap<ProcessId, ProcessExecutionState> processStates;

  /// Map from llhd.process ops to process IDs.
  llvm::DenseMap<mlir::Operation *, ProcessId> opToProcessId;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_SIM_LLHDPROCESSINTERPRETER_H
