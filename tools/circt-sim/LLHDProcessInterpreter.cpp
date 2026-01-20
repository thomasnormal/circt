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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  if (auto icmpOp = dyn_cast<comb::ICmpOp>(op)) {
    InterpretedValue lhs = getValue(procId, icmpOp.getLhs());
    InterpretedValue rhs = getValue(procId, icmpOp.getRhs());

    if (lhs.isX() || rhs.isX()) {
      setValue(procId, icmpOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(icmpOp.getType())));
      return success();
    }

    bool result = false;
    const llvm::APInt &lhsVal = lhs.getAPInt();
    const llvm::APInt &rhsVal = rhs.getAPInt();
    switch (icmpOp.getPredicate()) {
    case comb::ICmpPredicate::eq:
    case comb::ICmpPredicate::ceq:
    case comb::ICmpPredicate::weq:
      result = lhsVal == rhsVal;
      break;
    case comb::ICmpPredicate::ne:
    case comb::ICmpPredicate::cne:
    case comb::ICmpPredicate::wne:
      result = lhsVal != rhsVal;
      break;
    case comb::ICmpPredicate::slt:
      result = lhsVal.slt(rhsVal);
      break;
    case comb::ICmpPredicate::sle:
      result = lhsVal.sle(rhsVal);
      break;
    case comb::ICmpPredicate::sgt:
      result = lhsVal.sgt(rhsVal);
      break;
    case comb::ICmpPredicate::sge:
      result = lhsVal.sge(rhsVal);
      break;
    case comb::ICmpPredicate::ult:
      result = lhsVal.ult(rhsVal);
      break;
    case comb::ICmpPredicate::ule:
      result = lhsVal.ule(rhsVal);
      break;
    case comb::ICmpPredicate::ugt:
      result = lhsVal.ugt(rhsVal);
      break;
    case comb::ICmpPredicate::uge:
      result = lhsVal.uge(rhsVal);
      break;
    }

    setValue(procId, icmpOp.getResult(), InterpretedValue(result ? 1 : 0, 1));
    return success();
  }

  if (auto andOp = dyn_cast<comb::AndOp>(op)) {
    llvm::APInt result;
    bool hasResult = false;
    for (Value operand : andOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, andOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(andOp.getType())));
        return success();
      }
      if (!hasResult) {
        result = value.getAPInt();
        hasResult = true;
      } else {
        result &= value.getAPInt();
      }
    }
    if (!hasResult)
      result = llvm::APInt(getTypeWidth(andOp.getType()), 0);
    setValue(procId, andOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    llvm::APInt result;
    bool hasResult = false;
    for (Value operand : orOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, orOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(orOp.getType())));
        return success();
      }
      if (!hasResult) {
        result = value.getAPInt();
        hasResult = true;
      } else {
        result |= value.getAPInt();
      }
    }
    if (!hasResult)
      result = llvm::APInt(getTypeWidth(orOp.getType()), 0);
    setValue(procId, orOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    llvm::APInt result;
    bool hasResult = false;
    for (Value operand : xorOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, xorOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(xorOp.getType())));
        return success();
      }
      if (!hasResult) {
        result = value.getAPInt();
        hasResult = true;
      } else {
        result ^= value.getAPInt();
      }
    }
    if (!hasResult)
      result = llvm::APInt(getTypeWidth(xorOp.getType()), 0);
    setValue(procId, xorOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto shlOp = dyn_cast<comb::ShlOp>(op)) {
    InterpretedValue lhs = getValue(procId, shlOp.getLhs());
    InterpretedValue rhs = getValue(procId, shlOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, shlOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(shlOp.getType())));
      return success();
    }
    uint64_t shift = rhs.getAPInt().getLimitedValue();
    setValue(procId, shlOp.getResult(),
             InterpretedValue(lhs.getAPInt().shl(shift)));
    return success();
  }

  if (auto shruOp = dyn_cast<comb::ShrUOp>(op)) {
    InterpretedValue lhs = getValue(procId, shruOp.getLhs());
    InterpretedValue rhs = getValue(procId, shruOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, shruOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(shruOp.getType())));
      return success();
    }
    uint64_t shift = rhs.getAPInt().getLimitedValue();
    setValue(procId, shruOp.getResult(),
             InterpretedValue(lhs.getAPInt().lshr(shift)));
    return success();
  }

  if (auto shrsOp = dyn_cast<comb::ShrSOp>(op)) {
    InterpretedValue lhs = getValue(procId, shrsOp.getLhs());
    InterpretedValue rhs = getValue(procId, shrsOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, shrsOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(shrsOp.getType())));
      return success();
    }
    uint64_t shift = rhs.getAPInt().getLimitedValue();
    setValue(procId, shrsOp.getResult(),
             InterpretedValue(lhs.getAPInt().ashr(shift)));
    return success();
  }

  if (auto subOp = dyn_cast<comb::SubOp>(op)) {
    InterpretedValue lhs = getValue(procId, subOp.getLhs());
    InterpretedValue rhs = getValue(procId, subOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, subOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(subOp.getType())));
      return success();
    }
    setValue(procId, subOp.getResult(),
             InterpretedValue(lhs.getAPInt() - rhs.getAPInt()));
    return success();
  }

  if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
    llvm::APInt result;
    bool hasResult = false;
    for (Value operand : mulOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, mulOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(mulOp.getType())));
        return success();
      }
      if (!hasResult) {
        result = value.getAPInt();
        hasResult = true;
      } else {
        result *= value.getAPInt();
      }
    }
    if (!hasResult)
      result = llvm::APInt(getTypeWidth(mulOp.getType()), 0);
    setValue(procId, mulOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto divsOp = dyn_cast<comb::DivSOp>(op)) {
    InterpretedValue lhs = getValue(procId, divsOp.getLhs());
    InterpretedValue rhs = getValue(procId, divsOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, divsOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(divsOp.getType())));
      return success();
    }
    setValue(procId, divsOp.getResult(),
             InterpretedValue(lhs.getAPInt().sdiv(rhs.getAPInt())));
    return success();
  }

  if (auto divuOp = dyn_cast<comb::DivUOp>(op)) {
    InterpretedValue lhs = getValue(procId, divuOp.getLhs());
    InterpretedValue rhs = getValue(procId, divuOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, divuOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(divuOp.getType())));
      return success();
    }
    setValue(procId, divuOp.getResult(),
             InterpretedValue(lhs.getAPInt().udiv(rhs.getAPInt())));
    return success();
  }

  if (auto modsOp = dyn_cast<comb::ModSOp>(op)) {
    InterpretedValue lhs = getValue(procId, modsOp.getLhs());
    InterpretedValue rhs = getValue(procId, modsOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, modsOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(modsOp.getType())));
      return success();
    }
    setValue(procId, modsOp.getResult(),
             InterpretedValue(lhs.getAPInt().srem(rhs.getAPInt())));
    return success();
  }

  if (auto moduOp = dyn_cast<comb::ModUOp>(op)) {
    InterpretedValue lhs = getValue(procId, moduOp.getLhs());
    InterpretedValue rhs = getValue(procId, moduOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, moduOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(moduOp.getType())));
      return success();
    }
    setValue(procId, moduOp.getResult(),
             InterpretedValue(lhs.getAPInt().urem(rhs.getAPInt())));
    return success();
  }

  if (auto muxOp = dyn_cast<comb::MuxOp>(op)) {
    InterpretedValue cond = getValue(procId, muxOp.getCond());
    if (cond.isX()) {
      InterpretedValue trueVal = getValue(procId, muxOp.getTrueValue());
      InterpretedValue falseVal = getValue(procId, muxOp.getFalseValue());
      if (!trueVal.isX() && !falseVal.isX() &&
          trueVal.getAPInt() == falseVal.getAPInt()) {
        setValue(procId, muxOp.getResult(), trueVal);
      } else {
        setValue(procId, muxOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(muxOp.getType())));
      }
      return success();
    }
    InterpretedValue selected =
        cond.getUInt64() != 0 ? getValue(procId, muxOp.getTrueValue())
                              : getValue(procId, muxOp.getFalseValue());
    setValue(procId, muxOp.getResult(), selected);
    return success();
  }

  if (auto replOp = dyn_cast<comb::ReplicateOp>(op)) {
    InterpretedValue input = getValue(procId, replOp.getInput());
    if (input.isX()) {
      setValue(procId, replOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(replOp.getType())));
      return success();
    }
    unsigned inputWidth = input.getWidth();
    unsigned multiple = replOp.getMultiple();
    llvm::APInt result(getTypeWidth(replOp.getType()), 0);
    for (unsigned i = 0; i < multiple; ++i) {
      llvm::APInt chunk = input.getAPInt().zext(result.getBitWidth());
      result = result.shl(inputWidth) | chunk;
    }
    setValue(procId, replOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto ttOp = dyn_cast<comb::TruthTableOp>(op)) {
    auto inputs = ttOp.getInputs();
    auto table = ttOp.getLookupTable();
    size_t inputCount = inputs.size();
    auto values = table.getValue();
    if (values.size() != (1ULL << inputCount)) {
      setValue(procId, ttOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }

    llvm::SmallVector<int8_t, 8> bits;
    bits.reserve(inputCount);
    bool hasUnknown = false;
    for (Value input : inputs) {
      InterpretedValue value = getValue(procId, input);
      if (value.isX()) {
        bits.push_back(-1);
        hasUnknown = true;
      } else {
        bits.push_back(value.getUInt64() & 0x1);
      }
    }

    auto tableValueAt = [&](uint64_t index) -> bool {
      return llvm::cast<BoolAttr>(values[index]).getValue();
    };

    if (!hasUnknown) {
      uint64_t index = 0;
      for (int8_t bit : bits)
        index = (index << 1) | static_cast<uint8_t>(bit);
      setValue(procId, ttOp.getResult(), InterpretedValue(tableValueAt(index), 1));
      return success();
    }

    bool init = false;
    bool combined = false;
    for (uint64_t mask = 0; mask < (1ULL << inputCount); ++mask) {
      bool matches = true;
      for (size_t idx = 0; idx < inputCount; ++idx) {
        int8_t bit = bits[idx];
        if (bit < 0)
          continue;
        uint8_t current = (mask >> (inputCount - 1 - idx)) & 1;
        if (current != static_cast<uint8_t>(bit)) {
          matches = false;
          break;
        }
      }
      if (!matches)
        continue;
      bool value = tableValueAt(mask);
      if (!init) {
        combined = value;
        init = true;
      } else if (combined != value) {
        setValue(procId, ttOp.getResult(), InterpretedValue::makeX(1));
        return success();
      }
    }

    if (!init) {
      setValue(procId, ttOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }

    setValue(procId, ttOp.getResult(), InterpretedValue(combined, 1));
    return success();
  }

  if (auto reverseOp = dyn_cast<comb::ReverseOp>(op)) {
    InterpretedValue input = getValue(procId, reverseOp.getInput());
    if (input.isX()) {
      setValue(procId, reverseOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(reverseOp.getType())));
      return success();
    }
    setValue(procId, reverseOp.getResult(),
             InterpretedValue(input.getAPInt().reverseBits()));
    return success();
  }

  if (auto parityOp = dyn_cast<comb::ParityOp>(op)) {
    InterpretedValue input = getValue(procId, parityOp.getInput());
    if (input.isX()) {
      setValue(procId, parityOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }
    bool parity = (input.getAPInt().popcount() & 1) != 0;
    setValue(procId, parityOp.getResult(), InterpretedValue(parity, 1));
    return success();
  }

  if (auto extractOp = dyn_cast<comb::ExtractOp>(op)) {
    InterpretedValue input = getValue(procId, extractOp.getInput());
    if (input.isX()) {
      setValue(procId, extractOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(extractOp.getType())));
      return success();
    }
    uint32_t lowBit = extractOp.getLowBit();
    uint32_t width = getTypeWidth(extractOp.getType());
    if (lowBit + width > input.getWidth()) {
      setValue(procId, extractOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(extractOp.getType())));
      return success();
    }
    llvm::APInt result = input.getAPInt().lshr(lowBit).trunc(width);
    setValue(procId, extractOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto concatOp = dyn_cast<comb::ConcatOp>(op)) {
    llvm::APInt result;
    bool hasResult = false;
    for (Value operand : concatOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, concatOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(concatOp.getType())));
        return success();
      }
      if (!hasResult) {
        result = value.getAPInt();
        hasResult = true;
        continue;
      }
      unsigned rhsWidth = value.getWidth();
      llvm::APInt chunk = value.getAPInt().zext(result.getBitWidth() + rhsWidth);
      result = result.zext(result.getBitWidth() + rhsWidth);
      result = (result.shl(rhsWidth)) | chunk;
    }
    unsigned expectedWidth = getTypeWidth(concatOp.getType());
    if (!hasResult) {
      result = llvm::APInt(expectedWidth, 0);
    } else if (result.getBitWidth() < expectedWidth) {
      result = result.zext(expectedWidth);
    } else if (result.getBitWidth() > expectedWidth) {
      result = result.trunc(expectedWidth);
    }
    setValue(procId, concatOp.getResult(), InterpretedValue(result));
    return success();
  }

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

  //===--------------------------------------------------------------------===//
  // Arith Dialect Operations
  //===--------------------------------------------------------------------===//

  if (auto arithConstOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConstOp.getValue())) {
      setValue(procId, arithConstOp.getResult(),
               InterpretedValue(intAttr.getValue()));
    } else {
      setValue(procId, arithConstOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithConstOp.getType())));
    }
    return success();
  }

  if (auto arithAddIOp = dyn_cast<mlir::arith::AddIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithAddIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithAddIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithAddIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithAddIOp.getType())));
    } else {
      setValue(procId, arithAddIOp.getResult(),
               InterpretedValue(lhs.getAPInt() + rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithSubIOp = dyn_cast<mlir::arith::SubIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithSubIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithSubIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithSubIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithSubIOp.getType())));
    } else {
      setValue(procId, arithSubIOp.getResult(),
               InterpretedValue(lhs.getAPInt() - rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithMulIOp = dyn_cast<mlir::arith::MulIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithMulIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithMulIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithMulIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithMulIOp.getType())));
    } else {
      setValue(procId, arithMulIOp.getResult(),
               InterpretedValue(lhs.getAPInt() * rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithDivSIOp = dyn_cast<mlir::arith::DivSIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithDivSIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithDivSIOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithDivSIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithDivSIOp.getType())));
    } else {
      setValue(procId, arithDivSIOp.getResult(),
               InterpretedValue(lhs.getAPInt().sdiv(rhs.getAPInt())));
    }
    return success();
  }

  if (auto arithDivUIOp = dyn_cast<mlir::arith::DivUIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithDivUIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithDivUIOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithDivUIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithDivUIOp.getType())));
    } else {
      setValue(procId, arithDivUIOp.getResult(),
               InterpretedValue(lhs.getAPInt().udiv(rhs.getAPInt())));
    }
    return success();
  }

  if (auto arithRemSIOp = dyn_cast<mlir::arith::RemSIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithRemSIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithRemSIOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithRemSIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithRemSIOp.getType())));
    } else {
      setValue(procId, arithRemSIOp.getResult(),
               InterpretedValue(lhs.getAPInt().srem(rhs.getAPInt())));
    }
    return success();
  }

  if (auto arithRemUIOp = dyn_cast<mlir::arith::RemUIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithRemUIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithRemUIOp.getRhs());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithRemUIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithRemUIOp.getType())));
    } else {
      setValue(procId, arithRemUIOp.getResult(),
               InterpretedValue(lhs.getAPInt().urem(rhs.getAPInt())));
    }
    return success();
  }

  if (auto arithAndIOp = dyn_cast<mlir::arith::AndIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithAndIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithAndIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithAndIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithAndIOp.getType())));
    } else {
      setValue(procId, arithAndIOp.getResult(),
               InterpretedValue(lhs.getAPInt() & rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithOrIOp = dyn_cast<mlir::arith::OrIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithOrIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithOrIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithOrIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithOrIOp.getType())));
    } else {
      setValue(procId, arithOrIOp.getResult(),
               InterpretedValue(lhs.getAPInt() | rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithXOrIOp = dyn_cast<mlir::arith::XOrIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithXOrIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithXOrIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithXOrIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithXOrIOp.getType())));
    } else {
      setValue(procId, arithXOrIOp.getResult(),
               InterpretedValue(lhs.getAPInt() ^ rhs.getAPInt()));
    }
    return success();
  }

  if (auto arithShLIOp = dyn_cast<mlir::arith::ShLIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithShLIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithShLIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithShLIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithShLIOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, arithShLIOp.getResult(),
               InterpretedValue(lhs.getAPInt().shl(shift)));
    }
    return success();
  }

  if (auto arithShRUIOp = dyn_cast<mlir::arith::ShRUIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithShRUIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithShRUIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithShRUIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithShRUIOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, arithShRUIOp.getResult(),
               InterpretedValue(lhs.getAPInt().lshr(shift)));
    }
    return success();
  }

  if (auto arithShRSIOp = dyn_cast<mlir::arith::ShRSIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithShRSIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithShRSIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithShRSIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithShRSIOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, arithShRSIOp.getResult(),
               InterpretedValue(lhs.getAPInt().ashr(shift)));
    }
    return success();
  }

  if (auto arithCmpIOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithCmpIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithCmpIOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithCmpIOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }

    bool result = false;
    const llvm::APInt &lhsVal = lhs.getAPInt();
    const llvm::APInt &rhsVal = rhs.getAPInt();
    switch (arithCmpIOp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      result = lhsVal == rhsVal;
      break;
    case mlir::arith::CmpIPredicate::ne:
      result = lhsVal != rhsVal;
      break;
    case mlir::arith::CmpIPredicate::slt:
      result = lhsVal.slt(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::sle:
      result = lhsVal.sle(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::sgt:
      result = lhsVal.sgt(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::sge:
      result = lhsVal.sge(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::ult:
      result = lhsVal.ult(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::ule:
      result = lhsVal.ule(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::ugt:
      result = lhsVal.ugt(rhsVal);
      break;
    case mlir::arith::CmpIPredicate::uge:
      result = lhsVal.uge(rhsVal);
      break;
    }
    setValue(procId, arithCmpIOp.getResult(), InterpretedValue(result ? 1 : 0, 1));
    return success();
  }

  if (auto arithSelectOp = dyn_cast<mlir::arith::SelectOp>(op)) {
    InterpretedValue cond = getValue(procId, arithSelectOp.getCondition());
    if (cond.isX()) {
      InterpretedValue trueVal =
          getValue(procId, arithSelectOp.getTrueValue());
      InterpretedValue falseVal =
          getValue(procId, arithSelectOp.getFalseValue());
      if (!trueVal.isX() && !falseVal.isX() &&
          trueVal.getAPInt() == falseVal.getAPInt()) {
        setValue(procId, arithSelectOp.getResult(), trueVal);
      } else {
        setValue(
            procId, arithSelectOp.getResult(),
            InterpretedValue::makeX(getTypeWidth(arithSelectOp.getType())));
      }
      return success();
    }
    InterpretedValue selected =
        cond.getUInt64() != 0
            ? getValue(procId, arithSelectOp.getTrueValue())
            : getValue(procId, arithSelectOp.getFalseValue());
    setValue(procId, arithSelectOp.getResult(), selected);
    return success();
  }

  if (auto arithExtUIOp = dyn_cast<mlir::arith::ExtUIOp>(op)) {
    InterpretedValue input = getValue(procId, arithExtUIOp.getIn());
    if (input.isX()) {
      setValue(procId, arithExtUIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithExtUIOp.getType())));
    } else {
      unsigned outWidth = getTypeWidth(arithExtUIOp.getType());
      setValue(procId, arithExtUIOp.getResult(),
               InterpretedValue(input.getAPInt().zext(outWidth)));
    }
    return success();
  }

  if (auto arithExtSIOp = dyn_cast<mlir::arith::ExtSIOp>(op)) {
    InterpretedValue input = getValue(procId, arithExtSIOp.getIn());
    if (input.isX()) {
      setValue(procId, arithExtSIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithExtSIOp.getType())));
    } else {
      unsigned outWidth = getTypeWidth(arithExtSIOp.getType());
      setValue(procId, arithExtSIOp.getResult(),
               InterpretedValue(input.getAPInt().sext(outWidth)));
    }
    return success();
  }

  if (auto arithTruncIOp = dyn_cast<mlir::arith::TruncIOp>(op)) {
    InterpretedValue input = getValue(procId, arithTruncIOp.getIn());
    if (input.isX()) {
      setValue(procId, arithTruncIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithTruncIOp.getType())));
    } else {
      unsigned outWidth = getTypeWidth(arithTruncIOp.getType());
      setValue(procId, arithTruncIOp.getResult(),
               InterpretedValue(input.getAPInt().trunc(outWidth)));
    }
    return success();
  }

  if (auto arithIndexCastOp = dyn_cast<mlir::arith::IndexCastOp>(op)) {
    InterpretedValue input = getValue(procId, arithIndexCastOp.getIn());
    if (input.isX()) {
      setValue(procId, arithIndexCastOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithIndexCastOp.getType())));
    } else {
      unsigned outWidth = getTypeWidth(arithIndexCastOp.getType());
      if (outWidth > input.getWidth()) {
        setValue(procId, arithIndexCastOp.getResult(),
                 InterpretedValue(input.getAPInt().zext(outWidth)));
      } else if (outWidth < input.getWidth()) {
        setValue(procId, arithIndexCastOp.getResult(),
                 InterpretedValue(input.getAPInt().trunc(outWidth)));
      } else {
        setValue(procId, arithIndexCastOp.getResult(), input);
      }
    }
    return success();
  }

  if (auto arithIndexCastUIOp = dyn_cast<mlir::arith::IndexCastUIOp>(op)) {
    InterpretedValue input = getValue(procId, arithIndexCastUIOp.getIn());
    if (input.isX()) {
      setValue(procId, arithIndexCastUIOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arithIndexCastUIOp.getType())));
    } else {
      unsigned outWidth = getTypeWidth(arithIndexCastUIOp.getType());
      if (outWidth > input.getWidth()) {
        setValue(procId, arithIndexCastUIOp.getResult(),
                 InterpretedValue(input.getAPInt().zext(outWidth)));
      } else if (outWidth < input.getWidth()) {
        setValue(procId, arithIndexCastUIOp.getResult(),
                 InterpretedValue(input.getAPInt().trunc(outWidth)));
      } else {
        setValue(procId, arithIndexCastUIOp.getResult(), input);
      }
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // SCF Dialect Operations (control flow for loops/conditionals)
  //===--------------------------------------------------------------------===//

  if (auto scfIfOp = dyn_cast<mlir::scf::IfOp>(op)) {
    return interpretSCFIf(procId, scfIfOp);
  }

  if (auto scfForOp = dyn_cast<mlir::scf::ForOp>(op)) {
    return interpretSCFFor(procId, scfForOp);
  }

  if (auto scfWhileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
    return interpretSCFWhile(procId, scfWhileOp);
  }

  if (auto scfYieldOp = dyn_cast<mlir::scf::YieldOp>(op)) {
    // scf.yield is handled within the parent op interpretation
    return success();
  }

  if (auto scfConditionOp = dyn_cast<mlir::scf::ConditionOp>(op)) {
    // scf.condition is handled within the while loop interpretation
    return success();
  }

  //===--------------------------------------------------------------------===//
  // Func Dialect Operations (function calls)
  //===--------------------------------------------------------------------===//

  if (auto callOp = dyn_cast<mlir::func::CallOp>(op)) {
    return interpretFuncCall(procId, callOp);
  }

  if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    // Return is handled by the call interpreter
    return success();
  }

  //===--------------------------------------------------------------------===//
  // HW Array Operations
  //===--------------------------------------------------------------------===//

  if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(op)) {
    // Create an array from the input values
    // For interpretation, we pack all elements into a single APInt
    auto arrayType = hw::type_cast<hw::ArrayType>(arrayCreateOp.getType());
    unsigned elementWidth = getTypeWidth(arrayType.getElementType());
    unsigned numElements = arrayType.getNumElements();
    unsigned totalWidth = elementWidth * numElements;

    APInt result(totalWidth, 0);
    bool hasX = false;

    // Elements are stored in reverse order (index 0 at the LSB)
    for (size_t i = 0; i < numElements; ++i) {
      InterpretedValue elem = getValue(procId, arrayCreateOp.getInputs()[i]);
      if (elem.isX()) {
        hasX = true;
        break;
      }
      APInt elemVal = elem.getAPInt();
      if (elemVal.getBitWidth() < elementWidth)
        elemVal = elemVal.zext(elementWidth);
      else if (elemVal.getBitWidth() > elementWidth)
        elemVal = elemVal.trunc(elementWidth);
      // Insert element at position (numElements - 1 - i) * elementWidth
      // to maintain proper array ordering
      unsigned offset = (numElements - 1 - i) * elementWidth;
      result.insertBits(elemVal, offset);
    }

    if (hasX) {
      setValue(procId, arrayCreateOp.getResult(),
               InterpretedValue::makeX(totalWidth));
    } else {
      setValue(procId, arrayCreateOp.getResult(), InterpretedValue(result));
    }
    return success();
  }

  if (auto arrayGetOp = dyn_cast<hw::ArrayGetOp>(op)) {
    InterpretedValue arrayVal = getValue(procId, arrayGetOp.getInput());
    InterpretedValue indexVal = getValue(procId, arrayGetOp.getIndex());

    if (arrayVal.isX() || indexVal.isX()) {
      setValue(procId, arrayGetOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(arrayGetOp.getType())));
      return success();
    }

    auto arrayType = cast<hw::ArrayType>(arrayGetOp.getInput().getType());
    unsigned elementWidth = getTypeWidth(arrayType.getElementType());
    unsigned numElements = arrayType.getNumElements();
    uint64_t idx = indexVal.getAPInt().getZExtValue();

    if (idx >= numElements) {
      // Out of bounds - return X
      setValue(procId, arrayGetOp.getResult(),
               InterpretedValue::makeX(elementWidth));
      return success();
    }

    // Extract element at position (numElements - 1 - idx) * elementWidth
    unsigned offset = (numElements - 1 - idx) * elementWidth;
    APInt element = arrayVal.getAPInt().extractBits(elementWidth, offset);
    setValue(procId, arrayGetOp.getResult(), InterpretedValue(element));
    return success();
  }

  if (auto arraySliceOp = dyn_cast<hw::ArraySliceOp>(op)) {
    InterpretedValue arrayVal = getValue(procId, arraySliceOp.getInput());
    InterpretedValue indexVal = getValue(procId, arraySliceOp.getLowIndex());

    auto resultType = hw::type_cast<hw::ArrayType>(arraySliceOp.getType());
    if (arrayVal.isX() || indexVal.isX()) {
      unsigned resultWidth = getTypeWidth(resultType.getElementType()) *
                             resultType.getNumElements();
      setValue(procId, arraySliceOp.getResult(),
               InterpretedValue::makeX(resultWidth));
      return success();
    }

    auto inputType = hw::type_cast<hw::ArrayType>(arraySliceOp.getInput().getType());
    unsigned elementWidth = getTypeWidth(inputType.getElementType());
    unsigned inputElements = inputType.getNumElements();
    unsigned resultElements = resultType.getNumElements();
    uint64_t lowIdx = indexVal.getAPInt().getZExtValue();

    if (lowIdx + resultElements > inputElements) {
      // Out of bounds
      unsigned resultWidth = elementWidth * resultElements;
      setValue(procId, arraySliceOp.getResult(),
               InterpretedValue::makeX(resultWidth));
      return success();
    }

    // Extract slice starting at (inputElements - (lowIdx + resultElements))
    unsigned offset = (inputElements - (lowIdx + resultElements)) * elementWidth;
    unsigned sliceWidth = resultElements * elementWidth;
    APInt slice = arrayVal.getAPInt().extractBits(sliceWidth, offset);
    setValue(procId, arraySliceOp.getResult(), InterpretedValue(slice));
    return success();
  }

  if (auto arrayConcatOp = dyn_cast<hw::ArrayConcatOp>(op)) {
    auto resultType = hw::type_cast<hw::ArrayType>(arrayConcatOp.getType());
    unsigned resultWidth = getTypeWidth(resultType.getElementType()) *
                           resultType.getNumElements();

    APInt result(resultWidth, 0);
    bool hasX = false;
    unsigned bitOffset = resultWidth;

    for (Value input : arrayConcatOp.getInputs()) {
      InterpretedValue val = getValue(procId, input);
      if (val.isX()) {
        hasX = true;
        break;
      }
      unsigned inputWidth = val.getWidth();
      bitOffset -= inputWidth;
      result.insertBits(val.getAPInt(), bitOffset);
    }

    if (hasX) {
      setValue(procId, arrayConcatOp.getResult(),
               InterpretedValue::makeX(resultWidth));
    } else {
      setValue(procId, arrayConcatOp.getResult(), InterpretedValue(result));
    }
    return success();
  }

  //===--------------------------------------------------------------------===//
  // LLHD Time Operations
  //===--------------------------------------------------------------------===//

  if (auto currentTimeOp = dyn_cast<llhd::CurrentTimeOp>(op)) {
    // Return the current simulation time
    SimTime currentTime = scheduler.getCurrentTime();
    setValue(procId, currentTimeOp.getResult(),
             InterpretedValue(currentTime.realTime, 64));
    return success();
  }

  if (auto timeToIntOp = dyn_cast<llhd::TimeToIntOp>(op)) {
    // Convert time to integer femtoseconds
    if (auto constTimeOp =
            timeToIntOp.getInput().getDefiningOp<llhd::ConstantTimeOp>()) {
      SimTime time = convertTime(constTimeOp.getValueAttr());
      setValue(procId, timeToIntOp.getResult(),
               InterpretedValue(time.realTime, 64));
    } else {
      InterpretedValue input = getValue(procId, timeToIntOp.getInput());
      setValue(procId, timeToIntOp.getResult(), input);
    }
    return success();
  }

  if (auto intToTimeOp = dyn_cast<llhd::IntToTimeOp>(op)) {
    // Convert integer femtoseconds to time
    InterpretedValue input = getValue(procId, intToTimeOp.getInput());
    setValue(procId, intToTimeOp.getResult(), input);
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
// SCF Dialect Operation Interpreters
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretSCFIf(ProcessId procId,
                                                      mlir::scf::IfOp ifOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting scf.if\n");

  InterpretedValue cond = getValue(procId, ifOp.getCondition());

  // Determine which branch to execute
  mlir::Region *region = nullptr;
  if (cond.isX()) {
    // X condition - if both branches produce the same result, use it,
    // otherwise produce X for all results
    if (ifOp.getNumResults() > 0) {
      for (Value result : ifOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
    }
    return success();
  }

  if (cond.getUInt64() != 0) {
    region = &ifOp.getThenRegion();
  } else {
    if (ifOp.getElseRegion().empty()) {
      // No else branch
      return success();
    }
    region = &ifOp.getElseRegion();
  }

  // Execute the selected region
  llvm::SmallVector<InterpretedValue, 4> yieldValues;
  if (failed(interpretRegion(procId, *region, {}, yieldValues)))
    return failure();

  // Map yield values to the if op results
  for (auto [result, yieldVal] :
       llvm::zip(ifOp.getResults(), yieldValues)) {
    setValue(procId, result, yieldVal);
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSCFFor(ProcessId procId,
                                                       mlir::scf::ForOp forOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting scf.for\n");

  InterpretedValue lb = getValue(procId, forOp.getLowerBound());
  InterpretedValue ub = getValue(procId, forOp.getUpperBound());
  InterpretedValue step = getValue(procId, forOp.getStep());

  if (lb.isX() || ub.isX() || step.isX()) {
    // Cannot determine loop bounds - set results to X
    for (Value result : forOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Initialize iteration values
  llvm::SmallVector<InterpretedValue, 4> iterValues;
  for (Value initArg : forOp.getInitArgs()) {
    iterValues.push_back(getValue(procId, initArg));
  }

  int64_t lbVal = lb.getAPInt().getSExtValue();
  int64_t ubVal = ub.getAPInt().getSExtValue();
  int64_t stepVal = step.getAPInt().getSExtValue();

  // Prevent infinite loops
  const size_t maxIterations = 100000;
  size_t iterCount = 0;

  for (int64_t iv = lbVal; iv < ubVal && iterCount < maxIterations;
       iv += stepVal, ++iterCount) {
    // Set up block arguments: induction variable, then iter args
    llvm::SmallVector<InterpretedValue, 4> blockArgs;
    blockArgs.push_back(InterpretedValue(iv, getTypeWidth(forOp.getInductionVar().getType())));
    blockArgs.append(iterValues.begin(), iterValues.end());

    // Execute the loop body
    llvm::SmallVector<InterpretedValue, 4> yieldValues;
    if (failed(interpretRegion(procId, forOp.getRegion(), blockArgs, yieldValues)))
      return failure();

    // Update iteration values for next iteration
    iterValues = std::move(yieldValues);
  }

  if (iterCount >= maxIterations) {
    LLVM_DEBUG(llvm::dbgs() << "  Warning: scf.for reached max iterations\n");
  }

  // Map final iteration values to loop results
  for (auto [result, iterVal] : llvm::zip(forOp.getResults(), iterValues)) {
    setValue(procId, result, iterVal);
  }

  return success();
}

LogicalResult
LLHDProcessInterpreter::interpretSCFWhile(ProcessId procId,
                                           mlir::scf::WhileOp whileOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting scf.while\n");

  // Initialize with input arguments
  llvm::SmallVector<InterpretedValue, 4> iterValues;
  for (Value operand : whileOp.getOperands()) {
    iterValues.push_back(getValue(procId, operand));
  }

  const size_t maxIterations = 100000;
  size_t iterCount = 0;

  while (iterCount < maxIterations) {
    ++iterCount;

    // Execute the "before" region (condition check)
    llvm::SmallVector<InterpretedValue, 4> conditionResults;
    if (failed(interpretWhileCondition(procId, whileOp.getBefore(), iterValues,
                                        conditionResults)))
      return failure();

    // The first result is the condition
    if (conditionResults.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "  scf.while: no condition result\n");
      break;
    }

    InterpretedValue cond = conditionResults[0];
    if (cond.isX() || cond.getUInt64() == 0) {
      // Exit the loop - use remaining condition results as final values
      for (size_t i = 1; i < conditionResults.size() &&
                         i - 1 < whileOp.getResults().size(); ++i) {
        setValue(procId, whileOp.getResult(i - 1), conditionResults[i]);
      }
      break;
    }

    // Execute the "after" region (loop body)
    llvm::SmallVector<InterpretedValue, 4> afterValues(
        conditionResults.begin() + 1, conditionResults.end());
    llvm::SmallVector<InterpretedValue, 4> yieldValues;
    if (failed(interpretRegion(procId, whileOp.getAfter(), afterValues, yieldValues)))
      return failure();

    // Update iteration values for next condition check
    iterValues = std::move(yieldValues);
  }

  if (iterCount >= maxIterations) {
    LLVM_DEBUG(llvm::dbgs() << "  Warning: scf.while reached max iterations\n");
    // Set results to X on overflow
    for (Value result : whileOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretWhileCondition(
    ProcessId procId, mlir::Region &region,
    llvm::ArrayRef<InterpretedValue> args,
    llvm::SmallVectorImpl<InterpretedValue> &results) {
  if (region.empty())
    return failure();

  Block &block = region.front();

  // Set block arguments
  for (auto [arg, val] : llvm::zip(block.getArguments(), args)) {
    setValue(procId, arg, val);
  }

  // Execute operations until we hit scf.condition
  for (Operation &op : block) {
    if (auto condOp = dyn_cast<mlir::scf::ConditionOp>(&op)) {
      // Gather the condition and forwarded values
      results.push_back(getValue(procId, condOp.getCondition()));
      for (Value arg : condOp.getArgs()) {
        results.push_back(getValue(procId, arg));
      }
      return success();
    }

    if (failed(interpretOperation(procId, &op)))
      return failure();
  }

  return failure();
}

LogicalResult LLHDProcessInterpreter::interpretRegion(
    ProcessId procId, mlir::Region &region,
    llvm::ArrayRef<InterpretedValue> args,
    llvm::SmallVectorImpl<InterpretedValue> &results) {
  if (region.empty())
    return success();

  Block &block = region.front();

  // Set block arguments
  for (auto [arg, val] : llvm::zip(block.getArguments(), args)) {
    setValue(procId, arg, val);
  }

  // Execute operations until we hit a yield
  for (Operation &op : block) {
    if (auto yieldOp = dyn_cast<mlir::scf::YieldOp>(&op)) {
      // Gather yielded values
      for (Value operand : yieldOp.getOperands()) {
        results.push_back(getValue(procId, operand));
      }
      return success();
    }

    if (failed(interpretOperation(procId, &op)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Func Dialect Operation Interpreters
//===----------------------------------------------------------------------===//

LogicalResult
LLHDProcessInterpreter::interpretFuncCall(ProcessId procId,
                                           mlir::func::CallOp callOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting func.call to '"
                          << callOp.getCallee() << "'\n");

  // Find the called function
  auto *symbolOp = mlir::SymbolTable::lookupNearestSymbolFrom(
      callOp.getOperation(), callOp.getCalleeAttr());
  auto funcOp = dyn_cast_or_null<mlir::func::FuncOp>(symbolOp);
  if (!funcOp) {
    LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not find function '"
                            << callOp.getCallee() << "'\n");
    // Set results to X
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Check if function body is available (external functions cannot be called)
  if (funcOp.isExternal()) {
    LLVM_DEBUG(llvm::dbgs() << "  Warning: External function '"
                            << callOp.getCallee() << "' cannot be interpreted\n");
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Get call arguments
  llvm::SmallVector<InterpretedValue, 4> args;
  for (Value operand : callOp.getOperands()) {
    args.push_back(getValue(procId, operand));
  }

  // Execute the function body
  llvm::SmallVector<InterpretedValue, 4> returnValues;
  if (failed(interpretFuncBody(procId, funcOp, args, returnValues)))
    return failure();

  // Map return values to call results
  for (auto [result, retVal] : llvm::zip(callOp.getResults(), returnValues)) {
    setValue(procId, result, retVal);
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretFuncBody(
    ProcessId procId, mlir::func::FuncOp funcOp,
    llvm::ArrayRef<InterpretedValue> args,
    llvm::SmallVectorImpl<InterpretedValue> &results) {
  if (funcOp.getBody().empty())
    return failure();

  Block &entryBlock = funcOp.getBody().front();

  // Set function arguments
  for (auto [arg, val] : llvm::zip(entryBlock.getArguments(), args)) {
    setValue(procId, arg, val);
  }

  // Execute operations until we hit a return
  Block *currentBlock = &entryBlock;
  size_t maxOps = 100000; // Prevent infinite loops
  size_t opCount = 0;

  while (currentBlock && opCount < maxOps) {
    for (Operation &op : *currentBlock) {
      ++opCount;
      if (opCount >= maxOps) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Function reached max operations\n");
        return failure();
      }

      if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(&op)) {
        // Gather return values
        for (Value operand : returnOp.getOperands()) {
          results.push_back(getValue(procId, operand));
        }
        return success();
      }

      // Handle branch operations
      if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(&op)) {
        // Transfer operands to block arguments
        Block *dest = branchOp.getDest();
        for (auto [arg, operand] :
             llvm::zip(dest->getArguments(), branchOp.getDestOperands())) {
          setValue(procId, arg, getValue(procId, operand));
        }
        currentBlock = dest;
        break;
      }

      if (auto condBranchOp = dyn_cast<mlir::cf::CondBranchOp>(&op)) {
        InterpretedValue cond = getValue(procId, condBranchOp.getCondition());
        Block *dest;
        if (!cond.isX() && cond.getUInt64() != 0) {
          dest = condBranchOp.getTrueDest();
          for (auto [arg, operand] :
               llvm::zip(dest->getArguments(),
                         condBranchOp.getTrueDestOperands())) {
            setValue(procId, arg, getValue(procId, operand));
          }
        } else {
          dest = condBranchOp.getFalseDest();
          for (auto [arg, operand] :
               llvm::zip(dest->getArguments(),
                         condBranchOp.getFalseDestOperands())) {
            setValue(procId, arg, getValue(procId, operand));
          }
        }
        currentBlock = dest;
        break;
      }

      if (failed(interpretOperation(procId, &op)))
        return failure();
    }
  }

  return failure();
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

  // Handle index type (use 64-bit for indices)
  if (isa<IndexType>(type))
    return 64;

  // Handle hw.array types
  if (auto arrayType = dyn_cast<hw::ArrayType>(type))
    return getTypeWidth(arrayType.getElementType()) * arrayType.getNumElements();

  // Handle hw.struct types
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    unsigned totalWidth = 0;
    for (auto field : structType.getElements())
      totalWidth += getTypeWidth(field.type);
    return totalWidth;
  }

  // Default to 1 bit for unknown types
  return 1;
}
