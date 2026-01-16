//===- LLHDProcessInterpreter.cpp - LLHD process interpretation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLHDProcessInterpreter class for interpreting
// LLHD process bodies during event-driven simulation.
//
//===----------------------------------------------------------------------===//

#include "LLHDProcessInterpreter.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// LLHDProcessInterpreter Implementation
//===----------------------------------------------------------------------===//

LLHDProcessInterpreter::LLHDProcessInterpreter(ProcessScheduler &scheduler)
    : scheduler(scheduler) {}

LogicalResult LLHDProcessInterpreter::initialize(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initializing for module '"
                          << hwModule.getName() << "'\n");

  // Register all signals first
  if (failed(registerSignals(hwModule)))
    return failure();

  // Then register all processes
  if (failed(registerProcesses(hwModule)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Registered "
                          << getNumSignals() << " signals and "
                          << getNumProcesses() << " processes\n");

  return success();
}

//===----------------------------------------------------------------------===//
// Signal Registration
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::registerSignals(hw::HWModuleOp hwModule) {
  // First, register module ports that are ref types (signal references)
  for (auto portInfo : hwModule.getPortList()) {
    if (auto refType = dyn_cast<llhd::RefType>(portInfo.type)) {
      std::string name = portInfo.getName().str();
      Type innerType = refType.getNestedType();
      unsigned width = getTypeWidth(innerType);

      SignalId sigId = scheduler.registerSignal(name, width);
      signalIdToName[sigId] = name;

      // Map the block argument to the signal
      if (portInfo.isInput()) {
        auto &body = hwModule.getBody();
        if (!body.empty()) {
          Value arg = body.getArgument(portInfo.argNum);
          valueToSignal[arg] = sigId;
          LLVM_DEBUG(llvm::dbgs() << "  Registered port signal '" << name
                                  << "' with ID " << sigId << " (width=" << width
                                  << ")\n");
        }
      }
    }
  }

  // Walk the module body to find all llhd.sig operations
  hwModule.walk([&](llhd::SignalOp sigOp) {
    registerSignal(sigOp);
  });

  // Also register signals from llhd.output operations
  hwModule.walk([&](llhd::OutputOp outputOp) {
    // llhd.output creates a signal implicitly
    std::string name = outputOp.getName().value_or("").str();
    if (name.empty()) {
      name = "output_" + std::to_string(valueToSignal.size());
    }

    Type innerType = outputOp.getValue().getType();
    unsigned width = getTypeWidth(innerType);

    SignalId sigId = scheduler.registerSignal(name, width);
    valueToSignal[outputOp.getResult()] = sigId;
    signalIdToName[sigId] = name;

    LLVM_DEBUG(llvm::dbgs() << "  Registered output signal '" << name
                            << "' with ID " << sigId << " (width=" << width
                            << ")\n");
  });

  return success();
}

SignalId LLHDProcessInterpreter::registerSignal(llhd::SignalOp sigOp) {
  // Get signal name - use the optional name attribute if present
  std::string name = sigOp.getName().value_or("").str();
  if (name.empty()) {
    // Generate a name based on the SSA value
    name = "sig_" + std::to_string(valueToSignal.size());
  }

  // Get the type of the signal (the inner type, not the ref type)
  Type innerType = sigOp.getInit().getType();
  unsigned width = getTypeWidth(innerType);

  // Register with the scheduler
  SignalId sigId = scheduler.registerSignal(name, width);

  // Store the mapping
  valueToSignal[sigOp.getResult()] = sigId;
  signalIdToName[sigId] = name;

  // Set the initial value if the init operand is a constant
  if (auto constOp = sigOp.getInit().getDefiningOp<hw::ConstantOp>()) {
    APInt initValue = constOp.getValue();
    SignalValue sv(initValue.getZExtValue(), width);
    scheduler.updateSignal(sigId, sv);
    LLVM_DEBUG(llvm::dbgs() << "  Set initial value to " << initValue << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Registered signal '" << name << "' with ID "
                          << sigId << " (width=" << width << ")\n");

  return sigId;
}

SignalId LLHDProcessInterpreter::getSignalId(Value signalRef) const {
  auto it = valueToSignal.find(signalRef);
  if (it != valueToSignal.end())
    return it->second;
  return 0; // Invalid signal ID
}

llvm::StringRef LLHDProcessInterpreter::getSignalName(SignalId id) const {
  auto it = signalIdToName.find(id);
  if (it != signalIdToName.end())
    return it->second;
  return "";
}

//===----------------------------------------------------------------------===//
// Process Registration
//===----------------------------------------------------------------------===//

LogicalResult
LLHDProcessInterpreter::registerProcesses(hw::HWModuleOp hwModule) {
  // Walk the module body to find all llhd.process operations
  hwModule.walk([&](llhd::ProcessOp processOp) {
    registerProcess(processOp);
  });

  // Also handle llhd.combinational operations
  hwModule.walk([&](llhd::CombinationalOp combOp) {
    // TODO: Handle combinational processes in Phase 1B
    LLVM_DEBUG(llvm::dbgs() << "  Found combinational process (TODO)\n");
  });

  return success();
}

ProcessId LLHDProcessInterpreter::registerProcess(llhd::ProcessOp processOp) {
  // Generate a process name
  std::string name = "llhd_process_" + std::to_string(processStates.size());

  // Create the execution state for this process
  ProcessExecutionState state(processOp);

  // Register with the scheduler, providing a callback that executes this process
  ProcessId procId = scheduler.registerProcess(
      name, [this, procId = processStates.size() + 1]() {
        executeProcess(procId);
      });

  // Store the state
  state.currentBlock = &processOp.getBody().front();
  state.currentOp = state.currentBlock->begin();
  processStates[procId] = std::move(state);
  opToProcessId[processOp.getOperation()] = procId;

  LLVM_DEBUG(llvm::dbgs() << "  Registered process '" << name << "' with ID "
                          << procId << "\n");

  // Schedule the process to run at time 0 (initialization)
  scheduler.scheduleProcess(procId, SchedulingRegion::Active);

  return procId;
}

//===----------------------------------------------------------------------===//
// Process Execution
//===----------------------------------------------------------------------===//

void LLHDProcessInterpreter::executeProcess(ProcessId procId) {
  auto it = processStates.find(procId);
  if (it == processStates.end()) {
    LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Unknown process ID "
                            << procId << "\n");
    return;
  }

  ProcessExecutionState &state = it->second;

  // If resuming from a wait, set up the destination block
  // Note: waiting flag may already be cleared by resumeProcess, so check destBlock
  if (state.destBlock) {
    LLVM_DEBUG(llvm::dbgs() << "  Resuming to destination block\n");
    state.currentBlock = state.destBlock;
    state.currentOp = state.currentBlock->begin();

    // Transfer destination operands to block arguments
    for (auto [arg, val] :
         llvm::zip(state.currentBlock->getArguments(), state.destOperands)) {
      state.valueMap[arg] = val;
    }

    state.waiting = false;
    state.destBlock = nullptr;
    state.destOperands.clear();
  }

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executing process "
                          << procId << "\n");

  // Execute operations until we suspend or halt
  while (!state.halted && !state.waiting) {
    if (!executeStep(procId))
      break;
  }
}

bool LLHDProcessInterpreter::executeStep(ProcessId procId) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return false;

  ProcessExecutionState &state = it->second;

  // Check if we've reached the end of the block
  if (state.currentOp == state.currentBlock->end()) {
    // Shouldn't happen - blocks should end with terminators
    LLVM_DEBUG(llvm::dbgs()
               << "  Warning: Reached end of block without terminator\n");
    return false;
  }

  Operation *op = &*state.currentOp;
  ++state.currentOp;

  LLVM_DEBUG(llvm::dbgs() << "  Executing: " << *op << "\n");

  // Interpret the operation
  if (failed(interpretOperation(procId, op))) {
    LLVM_DEBUG(llvm::dbgs() << "  Failed to interpret operation\n");
    return false;
  }

  return !state.halted && !state.waiting;
}

void LLHDProcessInterpreter::resumeProcess(ProcessId procId) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return;

  // Clear waiting state and schedule for execution
  it->second.waiting = false;
  scheduler.scheduleProcess(procId, SchedulingRegion::Active);
}

//===----------------------------------------------------------------------===//
// Time Conversion
//===----------------------------------------------------------------------===//

SimTime LLHDProcessInterpreter::convertTime(llhd::TimeAttr timeAttr) {
  // TimeAttr has: time value, time unit, delta, epsilon
  // We need to convert to femtoseconds

  uint64_t realTime = timeAttr.getTime();
  llvm::StringRef unit = timeAttr.getTimeUnit();

  // Convert to femtoseconds based on unit
  // 1 fs = 1
  // 1 ps = 1000 fs
  // 1 ns = 1000000 fs
  // 1 us = 1000000000 fs
  // 1 ms = 1000000000000 fs
  // 1 s  = 1000000000000000 fs

  uint64_t multiplier = 1;
  if (unit == "fs")
    multiplier = 1;
  else if (unit == "ps")
    multiplier = 1000;
  else if (unit == "ns")
    multiplier = 1000000;
  else if (unit == "us")
    multiplier = 1000000000;
  else if (unit == "ms")
    multiplier = 1000000000000ULL;
  else if (unit == "s")
    multiplier = 1000000000000000ULL;

  uint64_t timeFemtoseconds = realTime * multiplier;

  // Delta and epsilon are stored in the deltaStep and region
  unsigned delta = timeAttr.getDelta();
  unsigned epsilon = timeAttr.getEpsilon();

  // For now, we treat epsilon as additional delta steps
  // (This is a simplification - proper handling would need separate tracking)
  return SimTime(timeFemtoseconds, delta + epsilon);
}

SimTime LLHDProcessInterpreter::convertTimeValue(ProcessId procId,
                                                  Value timeValue) {
  // Look up the interpreted value for the time
  InterpretedValue val = getValue(procId, timeValue);

  // If it's from a constant_time op, we should have stored the TimeAttr
  if (auto constTimeOp = timeValue.getDefiningOp<llhd::ConstantTimeOp>()) {
    return convertTime(constTimeOp.getValueAttr());
  }

  // For other cases, treat as femtoseconds value
  return SimTime(val.getUInt64());
}

//===----------------------------------------------------------------------===//
// Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretOperation(ProcessId procId,
                                                          Operation *op) {
  // Dispatch to specific handlers based on operation type

  // LLHD operations
  if (auto probeOp = dyn_cast<llhd::ProbeOp>(op))
    return interpretProbe(procId, probeOp);

  if (auto driveOp = dyn_cast<llhd::DriveOp>(op))
    return interpretDrive(procId, driveOp);

  if (auto waitOp = dyn_cast<llhd::WaitOp>(op))
    return interpretWait(procId, waitOp);

  if (auto haltOp = dyn_cast<llhd::HaltOp>(op))
    return interpretHalt(procId, haltOp);

  if (auto constTimeOp = dyn_cast<llhd::ConstantTimeOp>(op))
    return interpretConstantTime(procId, constTimeOp);

  // HW constant operations
  if (auto constOp = dyn_cast<hw::ConstantOp>(op)) {
    APInt value = constOp.getValue();
    setValue(procId, constOp.getResult(),
             InterpretedValue(value));
    return success();
  }

  // Control flow operations
  if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(op)) {
    auto &state = processStates[procId];
    state.currentBlock = branchOp.getDest();
    state.currentOp = state.currentBlock->begin();

    // Transfer operands to block arguments
    for (auto [arg, operand] : llvm::zip(state.currentBlock->getArguments(),
                                          branchOp.getDestOperands())) {
      state.valueMap[arg] = getValue(procId, operand);
    }
    return success();
  }

  if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(op)) {
    auto &state = processStates[procId];
    InterpretedValue cond = getValue(procId, condBranchOp.getCondition());

    if (!cond.isX() && cond.getUInt64() != 0) {
      // True branch
      state.currentBlock = condBranchOp.getTrueDest();
      state.currentOp = state.currentBlock->begin();
      for (auto [arg, operand] :
           llvm::zip(state.currentBlock->getArguments(),
                     condBranchOp.getTrueDestOperands())) {
        state.valueMap[arg] = getValue(procId, operand);
      }
    } else {
      // False branch (or X treated as false)
      state.currentBlock = condBranchOp.getFalseDest();
      state.currentOp = state.currentBlock->begin();
      for (auto [arg, operand] :
           llvm::zip(state.currentBlock->getArguments(),
                     condBranchOp.getFalseDestOperands())) {
        state.valueMap[arg] = getValue(procId, operand);
      }
    }
    return success();
  }

  // Arithmetic/comb operations - basic support
  if (auto addOp = dyn_cast<comb::AddOp>(op)) {
    InterpretedValue lhs = getValue(procId, addOp.getOperand(0));
    InterpretedValue rhs = getValue(procId, addOp.getOperand(1));

    if (lhs.isX() || rhs.isX()) {
      setValue(procId, addOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(addOp.getType())));
    } else {
      APInt result = lhs.getAPInt() + rhs.getAPInt();
      setValue(procId, addOp.getResult(), InterpretedValue(result));
    }
    return success();
  }

  // For unhandled operations, issue a warning but continue
  LLVM_DEBUG(llvm::dbgs() << "  Warning: Unhandled operation: "
                          << op->getName().getStringRef() << "\n");

  // For operations with results, set them to X
  for (Value result : op->getResults()) {
    setValue(procId, result,
             InterpretedValue::makeX(getTypeWidth(result.getType())));
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretProbe(ProcessId procId,
                                                      llhd::ProbeOp probeOp) {
  // Get the signal ID for the probed signal
  SignalId sigId = getSignalId(probeOp.getSignal());
  if (sigId == 0) {
    LLVM_DEBUG(llvm::dbgs() << "  Error: Unknown signal in probe\n");
    return failure();
  }

  // Get the current signal value from the scheduler
  const SignalValue &sigVal = scheduler.getSignalValue(sigId);

  // Convert to InterpretedValue and store
  InterpretedValue val = InterpretedValue::fromSignalValue(sigVal);
  setValue(procId, probeOp.getResult(), val);

  LLVM_DEBUG(llvm::dbgs() << "  Probed signal " << sigId << " = "
                          << (sigVal.isUnknown() ? "X"
                                                  : std::to_string(sigVal.getValue()))
                          << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretDrive(ProcessId procId,
                                                      llhd::DriveOp driveOp) {
  // Get the signal ID
  SignalId sigId = getSignalId(driveOp.getSignal());
  if (sigId == 0) {
    LLVM_DEBUG(llvm::dbgs() << "  Error: Unknown signal in drive\n");
    return failure();
  }

  // Check enable condition if present
  if (driveOp.getEnable()) {
    InterpretedValue enableVal = getValue(procId, driveOp.getEnable());
    if (enableVal.isX() || enableVal.getUInt64() == 0) {
      LLVM_DEBUG(llvm::dbgs() << "  Drive disabled (enable = "
                              << (enableVal.isX() ? "X" : "0") << ")\n");
      return success(); // Drive is disabled
    }
  }

  // Get the value to drive
  InterpretedValue driveVal = getValue(procId, driveOp.getValue());

  // Get the delay time
  SimTime delay = convertTimeValue(procId, driveOp.getTime());

  // Calculate the target time
  SimTime currentTime = scheduler.getCurrentTime();
  SimTime targetTime = currentTime.advanceTime(delay.realTime);
  if (delay.deltaStep > 0) {
    targetTime.deltaStep = delay.deltaStep;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Scheduling drive to signal " << sigId
                          << " at time " << targetTime.realTime << " fs"
                          << " (delay " << delay.realTime << " fs)\n");

  // Schedule the signal update
  SignalValue newVal = driveVal.toSignalValue();
  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::NBA,
      Event([this, sigId, newVal]() {
        scheduler.updateSignal(sigId, newVal);
      }));

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretWait(ProcessId procId,
                                                     llhd::WaitOp waitOp) {
  auto &state = processStates[procId];

  // Get the destination block
  state.destBlock = waitOp.getDest();

  // Store destination operands
  state.destOperands.clear();
  for (Value operand : waitOp.getDestOperands()) {
    state.destOperands.push_back(getValue(procId, operand));
  }

  // Mark as waiting
  state.waiting = true;

  // Handle delay-based wait
  if (waitOp.getDelay()) {
    SimTime delay = convertTimeValue(procId, waitOp.getDelay());
    SimTime targetTime = scheduler.getCurrentTime().advanceTime(delay.realTime);

    LLVM_DEBUG(llvm::dbgs() << "  Wait delay " << delay.realTime
                            << " fs until time " << targetTime.realTime << "\n");

    // Schedule resumption
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, procId]() { resumeProcess(procId); }));
  }

  // Handle event-based wait (sensitivity list)
  // Note: The 'observed' operands are probe results (values), not signal refs.
  // We need to trace back to find the original signal by looking at the
  // defining probe operation.
  if (!waitOp.getObserved().empty()) {
    SensitivityList waitList;
    for (Value observed : waitOp.getObserved()) {
      // Try to trace back to the signal through llhd.prb
      SignalId sigId = 0;
      if (auto probeOp = observed.getDefiningOp<llhd::ProbeOp>()) {
        sigId = getSignalId(probeOp.getSignal());
      } else {
        // Maybe it's directly a signal reference (shouldn't happen per spec,
        // but handle it gracefully)
        sigId = getSignalId(observed);
      }

      if (sigId != 0) {
        waitList.addLevel(sigId);
        LLVM_DEBUG(llvm::dbgs() << "  Waiting on signal " << sigId << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not find signal for "
                                    "observed value\n");
      }
    }

    // Register the wait sensitivity with the scheduler
    if (!waitList.empty()) {
      scheduler.suspendProcessForEvents(procId, waitList);
    }
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretHalt(ProcessId procId,
                                                     llhd::HaltOp haltOp) {
  auto &state = processStates[procId];
  state.halted = true;

  LLVM_DEBUG(llvm::dbgs() << "  Process halted\n");

  // Terminate the process in the scheduler
  scheduler.terminateProcess(procId);

  return success();
}

LogicalResult
LLHDProcessInterpreter::interpretConstantTime(ProcessId procId,
                                               llhd::ConstantTimeOp timeOp) {
  // Store the time value - we'll convert it when needed
  // For now, store a placeholder value that we can use to look up the op later
  setValue(procId, timeOp.getResult(), InterpretedValue(0, 64));

  return success();
}

//===----------------------------------------------------------------------===//
// Value Management
//===----------------------------------------------------------------------===//

InterpretedValue LLHDProcessInterpreter::getValue(ProcessId procId,
                                                   Value value) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return InterpretedValue::makeX(getTypeWidth(value.getType()));

  auto &valueMap = it->second.valueMap;
  auto valIt = valueMap.find(value);
  if (valIt != valueMap.end())
    return valIt->second;

  // Check if this is a constant defined outside the process
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    APInt constVal = constOp.getValue();
    InterpretedValue iv(constVal);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is a probe operation defined outside the process
  // These need to be evaluated lazily when their values are needed
  if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
    SignalId sigId = getSignalId(probeOp.getSignal());
    if (sigId != 0) {
      const SignalValue &sv = scheduler.getSignalValue(sigId);
      InterpretedValue iv = InterpretedValue::fromSignalValue(sv);
      valueMap[value] = iv;
      LLVM_DEBUG(llvm::dbgs() << "  Lazy probe of signal " << sigId << " = "
                              << (sv.isUnknown() ? "X"
                                                  : std::to_string(sv.getValue()))
                              << "\n");
      return iv;
    }
  }

  // Check if this is a constant_time operation
  if (auto constTimeOp = value.getDefiningOp<llhd::ConstantTimeOp>()) {
    // Return a placeholder - actual time conversion happens in convertTimeValue
    InterpretedValue iv(0, 64);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is a signal reference
  auto sigIt = valueToSignal.find(value);
  if (sigIt != valueToSignal.end()) {
    // This is a signal reference, return the signal value
    const SignalValue &sv = scheduler.getSignalValue(sigIt->second);
    return InterpretedValue::fromSignalValue(sv);
  }

  // Unknown value - return X
  LLVM_DEBUG(llvm::dbgs() << "  Warning: Unknown value, returning X\n");
  return InterpretedValue::makeX(getTypeWidth(value.getType()));
}

void LLHDProcessInterpreter::setValue(ProcessId procId, Value value,
                                       InterpretedValue val) {
  auto it = processStates.find(procId);
  if (it == processStates.end())
    return;

  it->second.valueMap[value] = val;
}

unsigned LLHDProcessInterpreter::getTypeWidth(Type type) {
  // Handle integer types
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  // Handle LLHD ref types
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return getTypeWidth(refType.getNestedType());

  // Handle LLHD time type (arbitrary width for now)
  if (isa<llhd::TimeType>(type))
    return 64;

  // Default to 1 bit for unknown types
  return 1;
}
