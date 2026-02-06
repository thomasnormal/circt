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
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Runtime/MooreRuntime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <fstream>
#include <sstream>

#define DEBUG_TYPE "llhd-interpreter"

using namespace mlir;
using namespace circt;
using namespace circt::sim;

static bool getSignalInitValue(Value initValue, unsigned width,
                               llvm::APInt &outValue);
static size_t countRegionOps(mlir::Region &region);
static bool isProcessCacheableBody(Operation *op);
static Type unwrapSignalType(Type type) {
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return refType.getNestedType();
  return type;
}

namespace circt::sim {

struct ScopedInstanceContext {
  ScopedInstanceContext(LLHDProcessInterpreter &interpreter, InstanceId instance)
      : interpreter(interpreter),
        previous(interpreter.activeInstanceId) {
    interpreter.activeInstanceId = instance;
  }

  ~ScopedInstanceContext() { interpreter.activeInstanceId = previous; }

  ScopedInstanceContext(const ScopedInstanceContext &) = delete;
  ScopedInstanceContext &operator=(const ScopedInstanceContext &) = delete;

private:
  LLHDProcessInterpreter &interpreter;
  InstanceId previous = 0;
};

struct ScopedInputValueMap {
  explicit ScopedInputValueMap(
      LLHDProcessInterpreter &interpreter,
      const InstanceInputMapping &mapping)
      : interpreter(interpreter) {
    for (const auto &entry : mapping) {
      auto it = interpreter.inputValueMap.find(entry.arg);
      if (it != interpreter.inputValueMap.end())
        previous.emplace_back(entry.arg, it->second);
      else
        added.push_back(entry.arg);
      interpreter.inputValueMap[entry.arg] = entry.value;

      auto instIt = interpreter.inputValueInstanceMap.find(entry.arg);
      if (instIt != interpreter.inputValueInstanceMap.end())
        previousInstances.emplace_back(entry.arg, instIt->second);
      else
        addedInstances.push_back(entry.arg);
      interpreter.inputValueInstanceMap[entry.arg] = entry.instanceId;
    }
  }

  ~ScopedInputValueMap() {
    for (const auto &entry : previous)
      interpreter.inputValueMap[entry.first] = entry.second;
    for (const auto &key : added)
      interpreter.inputValueMap.erase(key);
    for (const auto &entry : previousInstances)
      interpreter.inputValueInstanceMap[entry.first] = entry.second;
    for (const auto &key : addedInstances)
      interpreter.inputValueInstanceMap.erase(key);
  }

  ScopedInputValueMap(const ScopedInputValueMap &) = delete;
  ScopedInputValueMap &operator=(const ScopedInputValueMap &) = delete;

private:
  LLHDProcessInterpreter &interpreter;
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>, 8> previous;
  llvm::SmallVector<mlir::Value, 8> added;
  llvm::SmallVector<std::pair<mlir::Value, InstanceId>, 8> previousInstances;
  llvm::SmallVector<mlir::Value, 8> addedInstances;
};

} // namespace circt::sim

static bool getMaskedUInt64(const InterpretedValue &value,
                            unsigned targetWidth, uint64_t &out) {
  if (value.isX() || targetWidth > 64)
    return false;
  uint64_t v = value.getUInt64();
  unsigned width = value.getWidth();
  if (width < 64) {
    uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
    v &= mask;
  }
  if (targetWidth < 64) {
    uint64_t mask = (targetWidth == 64) ? ~0ULL : ((1ULL << targetWidth) - 1);
    v &= mask;
  }
  out = v;
  return true;
}

//===----------------------------------------------------------------------===//
// LLHDProcessInterpreter Implementation
//===----------------------------------------------------------------------===//

LLHDProcessInterpreter::LLHDProcessInterpreter(ProcessScheduler &scheduler)
    : scheduler(scheduler), forkJoinManager(scheduler),
      syncPrimitivesManager(scheduler) {}

void LLHDProcessInterpreter::dumpProcessStates(llvm::raw_ostream &os) const {
  os << "[circt-sim] Process states:\n";
  for (const auto &entry : processStates) {
    ProcessId procId = entry.first;
    const ProcessExecutionState &state = entry.second;
    const Process *proc = scheduler.getProcess(procId);
    os << "  proc " << procId;
    if (proc)
      os << " '" << proc->getName() << "'";
    os << " type=" << (state.isInitialBlock ? "initial" : "process");
    if (proc)
      os << " state=" << getProcessStateName(proc->getState());
    os << " waiting=" << (state.waiting ? "1" : "0")
       << " halted=" << (state.halted ? "1" : "0")
       << " steps=" << state.totalSteps;
    if (state.lastOp)
      os << " lastOp=" << state.lastOp->getName().getStringRef();
    os << "\n";
  }
  os.flush();
}

void LLHDProcessInterpreter::dumpOpStats(llvm::raw_ostream &os,
                                         size_t topN) const {
  if (opStats.empty())
    return;

  llvm::SmallVector<std::pair<llvm::StringRef, uint64_t>, 16> entries;
  entries.reserve(opStats.size());
  for (const auto &entry : opStats)
    entries.push_back({entry.getKey(), entry.getValue()});

  llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
    if (lhs.second != rhs.second)
      return lhs.second > rhs.second;
    return lhs.first < rhs.first;
  });

  os << "\n=== Op Stats (top " << topN << ") ===\n";
  size_t limit = std::min(topN, entries.size());
  for (size_t i = 0; i < limit; ++i) {
    os << entries[i].first << ": " << entries[i].second << "\n";
  }
  os << "========================\n";
}

void LLHDProcessInterpreter::dumpProcessStats(llvm::raw_ostream &os,
                                              size_t topN) const {
  if (processStates.empty())
    return;

  struct ProcEntry {
    ProcessId id;
    uint64_t steps;
    size_t opCount;
    uint64_t cacheSkips;
    uint64_t sensCacheHits;
    llvm::StringRef name;
  };

  llvm::SmallVector<ProcEntry, 16> entries;
  entries.reserve(processStates.size());
  for (const auto &entry : processStates) {
    ProcessId procId = entry.first;
    const ProcessExecutionState &state = entry.second;
    llvm::StringRef name;
    if (const Process *proc = scheduler.getProcess(procId))
      name = proc->getName();
    size_t opCount = state.opCount;
    if (opCount == 0) {
      if (auto processOp = state.getProcessOp()) {
        opCount = countRegionOps(processOp.getBody());
      } else if (auto initialOp = state.getInitialOp()) {
        opCount = countRegionOps(initialOp.getBody());
      }
    }
    entries.push_back({procId, state.totalSteps, opCount, state.cacheSkips,
                       state.waitSensitivityCacheHits, name});
  }

  llvm::sort(entries, [](const ProcEntry &lhs, const ProcEntry &rhs) {
    if (lhs.steps != rhs.steps)
      return lhs.steps > rhs.steps;
    return lhs.id < rhs.id;
  });

  os << "\n=== Process Stats (top " << topN << ") ===\n";
  size_t limit = std::min(topN, entries.size());
  for (size_t i = 0; i < limit; ++i) {
    os << "proc " << entries[i].id;
    if (!entries[i].name.empty())
      os << " '" << entries[i].name << "'";
    os << " steps=" << entries[i].steps << " ops=" << entries[i].opCount
       << " skips=" << entries[i].cacheSkips
       << " sens_cache=" << entries[i].sensCacheHits << "\n";
  }
  os << "===========================\n";
}

LogicalResult LLHDProcessInterpreter::initialize(hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initializing for module '"
                          << hwModule.getName() << "'\n");

  // Store the module name for hierarchical path construction
  moduleName = hwModule.getName().str();

  // Store the root module for symbol lookup
  rootModule = hwModule->getParentOfType<ModuleOp>();

  // === STACK OVERFLOW FIX ===
  // Use a single iterative pass to discover all operations instead of
  // multiple recursive walk() calls. This prevents stack overflow on
  // large designs (165k+ lines with deep nesting).
  DiscoveredOps discoveredOps;
  discoverOpsIteratively(hwModule, discoveredOps);

  // Register all signals first (using pre-discovered ops)
  if (failed(registerSignals(hwModule, discoveredOps)))
    return failure();

  // Register seq.firreg operations before processes (using pre-discovered ops).
  registerFirRegs(discoveredOps, 0, InstanceInputMapping{});

  // Export signals to MooreRuntime signal registry for DPI/VPI access
  exportSignalsToRegistry();

  // Then register all processes (using pre-discovered ops)
  if (failed(registerProcesses(discoveredOps)))
    return failure();

  // Recursively process child module instances before registering
  // continuous assignments so instance outputs are mapped.
  // Note: initializeChildInstances does its own iterative discovery for each child
  if (failed(initializeChildInstances(discoveredOps, 0)))
    return failure();

  // Register combinational processes for static module-level drives
  // (continuous assignments like port connections).
  registerContinuousAssignments(hwModule, 0, InstanceInputMapping{});

  // Initialize LLVM global variables (especially vtables) using iterative discovery
  if (failed(initializeGlobals()))
    return failure();

  inGlobalInit = true;
  // Execute LLVM global constructors (e.g., __moore_global_init_uvm_pkg::uvm_top)
  // This initializes UVM globals like uvm_top before processes start
  if (failed(executeGlobalConstructors()))
    return failure();

  // Execute module-level LLVM ops (alloca, call, store) that initialize
  // module-level variables like strings before processes start.
  if (failed(executeModuleLevelLLVMOps(hwModule)))
    return failure();

  inGlobalInit = false;

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Registered "
                          << getNumSignals() << " signals and "
                          << getNumProcesses() << " processes\n");

  return success();
}

LogicalResult
LLHDProcessInterpreter::initializeChildInstances(const DiscoveredOps &ops,
                                                 InstanceId parentInstanceId) {
  // Process all pre-discovered hw.instance operations (no walk() needed)
  for (hw::InstanceOp instOp : ops.instances) {
    // Get the referenced module name
    StringRef childModuleName = instOp.getReferencedModuleName();

    LLVM_DEBUG(llvm::dbgs() << "  Found instance '" << instOp.getInstanceName()
                            << "' of module '" << childModuleName << "'\n");

    // Look up the child module in the symbol table
    if (!rootModule) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: No root module for symbol lookup\n");
      continue;
    }

    auto childModule =
        rootModule.lookupSymbol<hw::HWModuleOp>(childModuleName);
    if (!childModule) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: Could not find module '"
                              << childModuleName << "'\n");
      continue;
    }

    InstanceId instanceId = nextInstanceId++;

    llvm::SmallVector<std::string, 8> outputNames;
    for (auto portInfo : childModule.getPortList()) {
      if (!portInfo.isInput())
        outputNames.push_back(portInfo.getName().str());
    }

    InstanceInputMapping instanceInputMap;
    llvm::SmallVector<std::pair<SignalId, InstanceOutputInfo>, 8>
        pendingInstanceOutputs;

    // Map child module input block arguments to parent signals for EACH instance.
    // This is needed so that when we evaluate instance outputs, the input mappings
    // are available (and to allow firreg reset init to see the parent signals).
    auto &childBody = childModule.getBody();
    if (!childBody.empty()) {
      for (auto portInfo : childModule.getPortList()) {
        if (!portInfo.isInput())
          continue;
        unsigned operandIdx = portInfo.argNum;
        if (operandIdx >= instOp.getNumOperands())
          continue;
        auto childArg = cast<mlir::BlockArgument>(
            childBody.getArgument(operandIdx));
        Value operand = instOp.getOperand(operandIdx);
        instanceInputMap.push_back({childArg, operand, parentInstanceId});
        SignalId sigId = 0;
        {
          ScopedInstanceContext scope(*this, parentInstanceId);
          sigId = resolveSignalId(operand);
        }
        if (sigId != 0) {
          instanceValueToSignal[instanceId][childArg] = sigId;
          LLVM_DEBUG(llvm::dbgs()
                     << "    Mapped child input '"
                     << portInfo.getName() << "' to signal " << sigId << "\n");
        }
      }
    }
    instanceInputMaps[instanceId] = instanceInputMap;

    // Map instance results to child module outputs for instance evaluation.
    if (auto *bodyBlock = childModule.getBodyBlock()) {
      if (auto outputOp =
              dyn_cast<hw::OutputOp>(bodyBlock->getTerminator())) {
        unsigned resultCount = instOp.getNumResults();
        unsigned outputCount = outputOp.getNumOperands();
        unsigned mapCount = std::min(resultCount, outputCount);
        for (unsigned i = 0; i < mapCount; ++i) {
          InstanceOutputInfo info;
          info.outputValue = outputOp.getOperand(i);
          info.inputMap = instanceInputMap;
          info.instanceId = instanceId;
          instanceOutputMap[parentInstanceId][instOp.getResult(i)] = info;
          std::string outName;
          if (i < outputNames.size() && !outputNames[i].empty())
            outName = outputNames[i];
          else
            outName = "out_" + std::to_string(i);
          std::string hierName =
              instOp.getInstanceName().str() + "." + outName;
          Type outType = outputOp.getOperand(i).getType();
          unsigned width = getTypeWidth(outType);
          SignalId outSigId =
              scheduler.registerSignal(hierName, width,
                                       getSignalEncoding(outType));
          valueToSignal[instOp.getResult(i)] = outSigId;
          signalIdToName[outSigId] = hierName;
          signalIdToType[outSigId] = unwrapSignalType(outType);

          pendingInstanceOutputs.push_back({outSigId, info});
        }
        if (resultCount != outputCount) {
          LLVM_DEBUG(llvm::dbgs()
                     << "    Warning: Instance output count mismatch for '"
                     << instOp.getInstanceName() << "' (results="
                     << resultCount << ", outputs=" << outputCount << ")\n");
        }
      }
    }

    DiscoveredOps childOps;
    auto cacheIt = discoveredOpsCache.find(childModuleName);
    if (cacheIt == discoveredOpsCache.end()) {
      discoverOpsIteratively(childModule, childOps);
      discoveredOpsCache.try_emplace(childModuleName, childOps);
    } else {
      childOps = cacheIt->second;
    }

    // Register signals from child module using pre-discovered ops (per instance)
    for (llhd::SignalOp sigOp : childOps.signals) {
      std::string name = sigOp.getName().value_or("").str();
      if (name.empty())
        name = "sig_" + std::to_string(valueToSignal.size());
      std::string hierName = instOp.getInstanceName().str() + "." + name;

      Type innerType = sigOp.getInit().getType();
      unsigned width = getTypeWidth(innerType);

      SignalId sigId =
          scheduler.registerSignal(hierName, width,
                                   getSignalEncoding(innerType));
      instanceValueToSignal[instanceId][sigOp.getResult()] = sigId;
      signalIdToName[sigId] = hierName;
      signalIdToType[sigId] = innerType;

      llvm::APInt initValue;
      if (getSignalInitValue(sigOp.getInit(), width, initValue))
        scheduler.updateSignal(sigId, SignalValue(initValue));

      LLVM_DEBUG(llvm::dbgs() << "    Registered child signal '" << hierName
                              << "' with ID " << sigId << "\n");
    }

    // Register seq.firreg operations for the child module (per instance).
    registerFirRegs(childOps, instanceId, instanceInputMap);

    // Register processes from child module using pre-discovered ops
    for (llhd::ProcessOp processOp : childOps.processes) {
      std::string procName = instOp.getInstanceName().str() + ".llhd_process_" +
                             std::to_string(processStates.size());

      ProcessExecutionState state(processOp);
      state.instanceId = instanceId;
      state.inputMap = instanceInputMap;
      state.cacheable = isProcessCacheableBody(processOp);
      ProcessId procId = scheduler.registerProcess(procName, []() {});
      if (auto *process = scheduler.getProcess(procId))
        process->setCallback([this, procId]() { executeProcess(procId); });

      state.currentBlock = &processOp.getBody().front();
      state.currentOp = state.currentBlock->begin();
      registerProcessState(procId, std::move(state));
      instanceOpToProcessId[instanceId][processOp.getOperation()] = procId;

      LLVM_DEBUG(llvm::dbgs() << "    Registered child process '" << procName
                              << "' with ID " << procId << "\n");

      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    }

    // Register module-level llhd.drv operations using pre-discovered ops.
    for (llhd::DriveOp driveOp : childOps.moduleDrives) {
      registerModuleDrive(driveOp, instanceId, instanceInputMap);
    }

    // Register seq.initial blocks from child module using pre-discovered ops
    for (seq::InitialOp initialOp : childOps.initials) {
      std::string initName = instOp.getInstanceName().str() + ".seq_initial_" +
                             std::to_string(processStates.size());

      ProcessExecutionState state(initialOp);
      state.instanceId = instanceId;
      state.inputMap = instanceInputMap;
      state.cacheable = false;
      ProcessId procId = scheduler.registerProcess(initName, []() {});
      if (auto *process = scheduler.getProcess(procId))
        process->setCallback([this, procId]() { executeProcess(procId); });

      state.currentBlock = initialOp.getBodyBlock();
      state.currentOp = state.currentBlock->begin();
      registerProcessState(procId, std::move(state));
      instanceOpToProcessId[instanceId][initialOp.getOperation()] = procId;

      LLVM_DEBUG(llvm::dbgs() << "    Registered child initial block '" << initName
                              << "' with ID " << procId << "\n");

      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    }

    // Recursively process child module's instances using discovered ops
    (void)initializeChildInstances(childOps, instanceId);

    // Process pendingInstanceOutputs for EACH instance. Each instance needs its
    // own output updates even if they reference the same child module outputs.
    for (const auto &pending : pendingInstanceOutputs) {
      InstanceOutputUpdate update;
      update.signalId = pending.first;
      update.outputValue = pending.second.outputValue;
      update.instanceId = pending.second.instanceId;
      update.inputMap = pending.second.inputMap;
      {
        ScopedInstanceContext instScope(*this, update.instanceId);
        if (update.inputMap.empty()) {
          collectProcessIds(update.outputValue, update.processIds);
        } else {
          ScopedInputValueMap scope(*this, update.inputMap);
          collectProcessIds(update.outputValue, update.processIds);
        }
      }
      instanceOutputUpdates.push_back(update);

      llvm::SmallVector<SignalId, 4> sourceSignals;
      {
        ScopedInstanceContext instScope(*this, update.instanceId);
        if (update.inputMap.empty()) {
          collectSignalIds(update.outputValue, sourceSignals);
        } else {
          ScopedInputValueMap scope(*this, update.inputMap);
          collectSignalIds(update.outputValue, sourceSignals);
        }
      }
      if (sourceSignals.empty() && update.processIds.empty()) {
        scheduleInstanceOutputUpdate(update.signalId, update.outputValue,
                                     update.instanceId,
                                     update.inputMap.empty()
                                         ? nullptr
                                         : &update.inputMap);
      } else if (!sourceSignals.empty()) {
        std::string procName = "inst_out_" + std::to_string(update.signalId);
        auto inputMap = update.inputMap;
        InstanceId instanceId = update.instanceId;
        ProcessId procId = scheduler.registerProcess(
            procName, [this, signalId = update.signalId,
                       outputValue = update.outputValue, instanceId,
                       inputMap]() {
              scheduleInstanceOutputUpdate(
                  signalId, outputValue, instanceId,
                  inputMap.empty() ? nullptr : &inputMap);
            });
        auto *process = scheduler.getProcess(procId);
        if (process) {
          process->setCombinational(true);
          for (SignalId srcSigId : sourceSignals)
            scheduler.addSensitivity(procId, srcSigId);
        }
        scheduleInstanceOutputUpdate(update.signalId, update.outputValue,
                                     update.instanceId,
                                     update.inputMap.empty()
                                         ? nullptr
                                         : &update.inputMap);
        scheduler.scheduleProcess(procId, SchedulingRegion::Active);
      }
    }

    // Register combinational processes for static module-level drives in child.
    // This MUST happen after input mapping so that collectSignalIds can resolve
    // child input block arguments to parent signals for proper sensitivity setup.
    registerContinuousAssignments(childModule, instanceId, instanceInputMap);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static bool isProcessCacheableBody(Operation *op) {
  bool cacheable = true;
  op->walk([&](Operation *nestedOp) {
    if (isa<sim::TerminateOp, sim::PrintFormattedProcOp, sim::SimForkOp,
            sim::SimJoinOp, sim::SimJoinAnyOp, sim::SimDisableForkOp,
            llhd::HaltOp, seq::YieldOp>(nestedOp)) {
      cacheable = false;
      return WalkResult::interrupt();
    }
    if (dyn_cast<CallOpInterface>(nestedOp)) {
      cacheable = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return cacheable;
}

/// Flatten an aggregate constant (struct or array) into a single APInt.
/// This is used for initializing signals with aggregate types.
/// For 4-state logic, structs typically have the form {value, unknown}.
static Type stripTypeAliases(Type type) {
  while (auto alias = dyn_cast<hw::TypeAliasType>(type))
    type = alias.getInnerType();
  return type;
}

static APInt flattenAggregateAttr(Attribute fieldAttr, Type fieldType) {
  fieldType = stripTypeAliases(fieldType);
  unsigned fieldWidth = LLHDProcessInterpreter::getTypeWidth(fieldType);
  APInt result(fieldWidth, 0);

  if (auto intAttr = dyn_cast<IntegerAttr>(fieldAttr)) {
    APInt fieldValue = intAttr.getValue();
    if (fieldValue.getBitWidth() < fieldWidth)
      fieldValue = fieldValue.zext(fieldWidth);
    else if (fieldValue.getBitWidth() > fieldWidth)
      fieldValue = fieldValue.trunc(fieldWidth);
    return fieldValue;
  }

  auto arrayAttr = dyn_cast<ArrayAttr>(fieldAttr);
  if (!arrayAttr)
    return result;

  if (auto structType = dyn_cast<hw::StructType>(fieldType)) {
    auto elements = structType.getElements();
    unsigned bitOffset = fieldWidth;

    for (size_t i = 0; i < arrayAttr.size() && i < elements.size(); ++i) {
      unsigned elemWidth =
          LLHDProcessInterpreter::getTypeWidth(elements[i].type);
      bitOffset -= elemWidth;
      APInt elemValue = flattenAggregateAttr(arrayAttr[i], elements[i].type);
      if (elemValue.getBitWidth() < elemWidth)
        elemValue = elemValue.zext(elemWidth);
      else if (elemValue.getBitWidth() > elemWidth)
        elemValue = elemValue.trunc(elemWidth);
      result.insertBits(elemValue, bitOffset);
    }
    return result;
  }

  if (auto arrayType = dyn_cast<hw::ArrayType>(fieldType)) {
    unsigned elementWidth =
        LLHDProcessInterpreter::getTypeWidth(arrayType.getElementType());
    unsigned bitOffset = fieldWidth;

    for (Attribute elemAttr : arrayAttr) {
      bitOffset -= elementWidth;
      APInt elemValue =
          flattenAggregateAttr(elemAttr, arrayType.getElementType());
      if (elemValue.getBitWidth() < elementWidth)
        elemValue = elemValue.zext(elementWidth);
      else if (elemValue.getBitWidth() > elementWidth)
        elemValue = elemValue.trunc(elementWidth);
      result.insertBits(elemValue, bitOffset);
    }
  }

  return result;
}

static APInt flattenAggregateConstant(hw::AggregateConstantOp aggConstOp) {
  return flattenAggregateAttr(aggConstOp.getFields(),
                              aggConstOp.getResult().getType());
}

static size_t countRegionOps(mlir::Region &region) {
  llvm::SmallVector<mlir::Region *, 16> regionWorklist;
  regionWorklist.push_back(&region);
  size_t count = 0;

  while (!regionWorklist.empty()) {
    mlir::Region *curRegion = regionWorklist.pop_back_val();
    for (mlir::Block &block : *curRegion) {
      for (mlir::Operation &op : block) {
        ++count;
        for (mlir::Region &nested : op.getRegions())
          regionWorklist.push_back(&nested);
      }
    }
  }

  return count;
}

/// Extract an initial APInt value from a signal init operand if possible.
/// Returns true if a constant value was found and normalized to the signal width.
static bool getSignalInitValue(Value initValue, unsigned width,
                               llvm::APInt &outValue) {
  if (auto constOp = initValue.getDefiningOp<hw::ConstantOp>()) {
    outValue = constOp.getValue();
  } else if (auto aggConstOp =
                 initValue.getDefiningOp<hw::AggregateConstantOp>()) {
    outValue = flattenAggregateConstant(aggConstOp);
  } else if (auto bitcastOp = initValue.getDefiningOp<hw::BitcastOp>()) {
    if (auto constOp = bitcastOp.getInput().getDefiningOp<hw::ConstantOp>()) {
      outValue = constOp.getValue();
    } else if (auto aggConstOp = bitcastOp.getInput()
                                     .getDefiningOp<hw::AggregateConstantOp>()) {
      outValue = flattenAggregateConstant(aggConstOp);
    } else {
      return false;
    }
  } else {
    return false;
  }

  if (outValue.getBitWidth() < width)
    outValue = outValue.zext(width);
  else if (outValue.getBitWidth() > width)
    outValue = outValue.trunc(width);
  return true;
}

/// Normalize two APInt values to have the same bit width for binary operations.
/// If the widths differ, both are extended/truncated to the target width.
/// This prevents assertion failures in APInt binary operators that require
/// matching bit widths.
static void normalizeWidths(llvm::APInt &lhs, llvm::APInt &rhs,
                            unsigned targetWidth) {
  // Adjust lhs to target width
  if (lhs.getBitWidth() < targetWidth) {
    lhs = lhs.zext(targetWidth);
  } else if (lhs.getBitWidth() > targetWidth) {
    lhs = lhs.trunc(targetWidth);
  }

  // Adjust rhs to target width
  if (rhs.getBitWidth() < targetWidth) {
    rhs = rhs.zext(targetWidth);
  } else if (rhs.getBitWidth() > targetWidth) {
    rhs = rhs.trunc(targetWidth);
  }
}

//===----------------------------------------------------------------------===//
// Iterative Operation Discovery (Stack Overflow Prevention)
//===----------------------------------------------------------------------===//

void LLHDProcessInterpreter::discoverOpsIteratively(hw::HWModuleOp hwModule,
                                                     DiscoveredOps &ops) {
  // Use an explicit worklist to traverse operations iteratively instead of
  // using walk(), which is recursive and causes stack overflow on large designs.
  // This single pass replaces 17+ separate walk() calls.

  llvm::SmallVector<Operation *, 256> worklist;
  llvm::SmallVector<Region *, 64> regionWorklist;

  // Start with the module's body region
  regionWorklist.push_back(&hwModule.getBody());

  while (!regionWorklist.empty()) {
    Region *region = regionWorklist.pop_back_val();

    // Process all blocks in this region
    for (Block &block : *region) {
      // Process all operations in this block
      for (Operation &op : block) {
        // Classify the operation by type
        if (auto instOp = dyn_cast<hw::InstanceOp>(&op)) {
          ops.instances.push_back(instOp);
        } else if (auto sigOp = dyn_cast<llhd::SignalOp>(&op)) {
          ops.signals.push_back(sigOp);
        } else if (auto outputOp = dyn_cast<llhd::OutputOp>(&op)) {
          ops.outputs.push_back(outputOp);
        } else if (auto processOp = dyn_cast<llhd::ProcessOp>(&op)) {
          ops.processes.push_back(processOp);
          // Don't recurse into process bodies for discovering module-level ops
        } else if (auto combOp = dyn_cast<llhd::CombinationalOp>(&op)) {
          ops.combinationals.push_back(combOp);
          // Don't recurse into combinational bodies
        } else if (auto initialOp = dyn_cast<seq::InitialOp>(&op)) {
          ops.initials.push_back(initialOp);
          // Don't recurse into initial bodies
        } else if (auto driveOp = dyn_cast<llhd::DriveOp>(&op)) {
          // Only collect module-level drives (not inside processes/initials)
          if (!op.getParentOfType<llhd::ProcessOp>() &&
              !op.getParentOfType<seq::InitialOp>()) {
            ops.moduleDrives.push_back(driveOp);
          }
        } else if (auto firRegOp = dyn_cast<seq::FirRegOp>(&op)) {
          ops.firRegs.push_back(firRegOp);
        }

        // Add nested regions to worklist (but skip process/initial/combinational bodies
        // since we don't want to discover ops inside those as module-level)
        if (!isa<llhd::ProcessOp, seq::InitialOp, llhd::CombinationalOp>(&op)) {
          for (Region &nestedRegion : op.getRegions()) {
            regionWorklist.push_back(&nestedRegion);
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Iterative discovery found: "
                          << ops.instances.size() << " instances, "
                          << ops.signals.size() << " signals, "
                          << ops.outputs.size() << " outputs, "
                          << ops.processes.size() << " processes, "
                          << ops.combinationals.size() << " combinationals, "
                          << ops.initials.size() << " initials, "
                          << ops.moduleDrives.size() << " module drives, "
                          << ops.firRegs.size() << " firRegs\n");
}

void LLHDProcessInterpreter::discoverGlobalOpsIteratively(
    DiscoveredGlobalOps &ops) {
  if (!rootModule)
    return;

  // Use an explicit worklist to traverse operations iteratively
  llvm::SmallVector<Region *, 64> regionWorklist;
  regionWorklist.push_back(&rootModule.getBodyRegion());

  while (!regionWorklist.empty()) {
    Region *region = regionWorklist.pop_back_val();

    for (Block &block : *region) {
      for (Operation &op : block) {
        // Classify global operations
        if (auto globalOp = dyn_cast<LLVM::GlobalOp>(&op)) {
          ops.globals.push_back(globalOp);
        } else if (auto ctorsOp = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ops.ctors.push_back(ctorsOp);
        }

        // Add nested regions (but skip hw.module bodies - we process those separately)
        if (!isa<hw::HWModuleOp>(&op)) {
          for (Region &nestedRegion : op.getRegions()) {
            regionWorklist.push_back(&nestedRegion);
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "  Global discovery found: "
                          << ops.globals.size() << " globals, "
                          << ops.ctors.size() << " global ctors\n");
}

//===----------------------------------------------------------------------===//
// Signal Registration
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::registerSignals(
    hw::HWModuleOp hwModule, const DiscoveredOps &ops) {
  // First, register module ports that are ref types (signal references)
  for (auto portInfo : hwModule.getPortList()) {
    if (auto refType = dyn_cast<llhd::RefType>(portInfo.type)) {
      std::string name = portInfo.getName().str();
      Type innerType = refType.getNestedType();
      unsigned width = getTypeWidth(innerType);

      SignalId sigId =
          scheduler.registerSignal(name, width, getSignalEncoding(innerType));
      signalIdToName[sigId] = name;
      signalIdToType[sigId] = innerType;

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

  // Register all pre-discovered llhd.sig operations (no walk() needed)
  for (llhd::SignalOp sigOp : ops.signals) {
    registerSignal(sigOp);
  }

  // Register all pre-discovered llhd.output operations (no walk() needed)
  for (llhd::OutputOp outputOp : ops.outputs) {
    // llhd.output creates a signal implicitly
    std::string name = outputOp.getName().value_or("").str();
    if (name.empty()) {
      name = "output_" + std::to_string(valueToSignal.size());
    }

    Type innerType = outputOp.getValue().getType();
    unsigned width = getTypeWidth(innerType);

    SignalId sigId =
        scheduler.registerSignal(name, width, getSignalEncoding(innerType));
    valueToSignal[outputOp.getResult()] = sigId;
    signalIdToName[sigId] = name;
    signalIdToType[sigId] = innerType;

    LLVM_DEBUG(llvm::dbgs() << "  Registered output signal '" << name
                            << "' with ID " << sigId << " (width=" << width
                            << ")\n");
  }

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
  SignalId sigId =
      scheduler.registerSignal(name, width, getSignalEncoding(innerType));

  // Store the mapping
  valueToSignal[sigOp.getResult()] = sigId;
  signalIdToName[sigId] = name;
  signalIdToType[sigId] = innerType;

  llvm::APInt initValue;
  if (getSignalInitValue(sigOp.getInit(), width, initValue)) {
    scheduler.updateSignal(sigId, SignalValue(initValue));
    LLVM_DEBUG(llvm::dbgs() << "  Set initial value to " << initValue << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Registered signal '" << name << "' with ID "
                          << sigId << " (width=" << width << ")\n");

  return sigId;
}

SignalId LLHDProcessInterpreter::getSignalId(Value signalRef) const {
  SignalId sigId = getSignalIdInInstance(signalRef, activeInstanceId);
  if (sigId != 0)
    return sigId;
  SignalId uniqueSigId = 0;
  for (const auto &ctx : instanceValueToSignal) {
    auto instIt = ctx.second.find(signalRef);
    if (instIt == ctx.second.end())
      continue;
    if (uniqueSigId != 0 && uniqueSigId != instIt->second)
      return 0;
    uniqueSigId = instIt->second;
  }
  if (uniqueSigId != 0)
    return uniqueSigId;
  return 0; // Invalid signal ID
}

SignalId LLHDProcessInterpreter::getSignalIdInInstance(Value signalRef,
                                                       InstanceId instanceId) const {
  if (instanceId != 0) {
    auto ctxIt = instanceValueToSignal.find(instanceId);
    if (ctxIt != instanceValueToSignal.end()) {
      auto it = ctxIt->second.find(signalRef);
      if (it != ctxIt->second.end())
        return it->second;
    }
  }
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

Type LLHDProcessInterpreter::getSignalValueType(SignalId sigId) const {
  auto it = signalIdToType.find(sigId);
  if (it != signalIdToType.end())
    return it->second;
  for (const auto &entry : valueToSignal) {
    if (entry.second == sigId)
      return unwrapSignalType(entry.first.getType());
  }
  for (const auto &ctx : instanceValueToSignal) {
    for (const auto &entry : ctx.second) {
      if (entry.second == sigId)
        return unwrapSignalType(entry.first.getType());
    }
  }
  return Type();
}

static llvm::APInt adjustAPIntWidth(llvm::APInt value, unsigned targetWidth) {
  if (value.getBitWidth() == targetWidth)
    return value;
  if (value.getBitWidth() < targetWidth)
    return value.zext(targetWidth);
  return value.trunc(targetWidth);
}

llvm::APInt LLHDProcessInterpreter::convertLLVMToHWLayout(
    llvm::APInt value, Type llvmType, Type hwType) const {
  unsigned llvmWidth = getTypeWidth(llvmType);
  unsigned hwWidth = getTypeWidth(hwType);
  value = adjustAPIntWidth(value, llvmWidth);

  if (auto llvmStructType = dyn_cast<LLVM::LLVMStructType>(llvmType)) {
    if (auto hwStructType = dyn_cast<hw::StructType>(hwType)) {
      APInt result = APInt::getZero(hwWidth);
      auto hwElements = hwStructType.getElements();
      auto llvmBody = llvmStructType.getBody();
      size_t count = std::min(hwElements.size(), llvmBody.size());

      unsigned llvmOffset = 0;
      unsigned hwOffset = hwWidth;
      for (size_t i = 0; i < count; ++i) {
        unsigned llvmFieldWidth = getTypeWidth(llvmBody[i]);
        unsigned hwFieldWidth = getTypeWidth(hwElements[i].type);
        hwOffset -= hwFieldWidth;
        APInt fieldBits = value.extractBits(llvmFieldWidth, llvmOffset);
        APInt converted =
            convertLLVMToHWLayout(fieldBits, llvmBody[i], hwElements[i].type);
        converted = adjustAPIntWidth(converted, hwFieldWidth);
        result.insertBits(converted, hwOffset);
        llvmOffset += llvmFieldWidth;
      }
      return adjustAPIntWidth(result, hwWidth);
    }
  }

  if (auto llvmArrayType = dyn_cast<LLVM::LLVMArrayType>(llvmType)) {
    if (auto hwArrayType = dyn_cast<hw::ArrayType>(hwType)) {
      unsigned llvmElemWidth = getTypeWidth(llvmArrayType.getElementType());
      unsigned hwElemWidth = getTypeWidth(hwArrayType.getElementType());
      unsigned numElements = hwArrayType.getNumElements();
      unsigned llvmElements = llvmArrayType.getNumElements();
      size_t count = std::min(numElements, llvmElements);

      APInt result = APInt::getZero(hwWidth);
      for (size_t i = 0; i < count; ++i) {
        unsigned llvmOffset = i * llvmElemWidth;
        unsigned hwOffset = (numElements - 1 - i) * hwElemWidth;
        APInt fieldBits = value.extractBits(llvmElemWidth, llvmOffset);
        APInt converted = convertLLVMToHWLayout(
            fieldBits, llvmArrayType.getElementType(),
            hwArrayType.getElementType());
        converted = adjustAPIntWidth(converted, hwElemWidth);
        result.insertBits(converted, hwOffset);
      }
      return adjustAPIntWidth(result, hwWidth);
    }
  }

  return adjustAPIntWidth(value, hwWidth);
}

llvm::APInt LLHDProcessInterpreter::convertHWToLLVMLayout(
    llvm::APInt value, Type hwType, Type llvmType) const {
  unsigned hwWidth = getTypeWidth(hwType);
  unsigned llvmWidth = getTypeWidth(llvmType);
  value = adjustAPIntWidth(value, hwWidth);

  if (auto hwStructType = dyn_cast<hw::StructType>(hwType)) {
    if (auto llvmStructType = dyn_cast<LLVM::LLVMStructType>(llvmType)) {
      APInt result = APInt::getZero(llvmWidth);
      auto hwElements = hwStructType.getElements();
      auto llvmBody = llvmStructType.getBody();
      size_t count = std::min(hwElements.size(), llvmBody.size());

      unsigned hwOffset = hwWidth;
      unsigned llvmOffset = 0;
      for (size_t i = 0; i < count; ++i) {
        unsigned hwFieldWidth = getTypeWidth(hwElements[i].type);
        unsigned llvmFieldWidth = getTypeWidth(llvmBody[i]);
        hwOffset -= hwFieldWidth;
        APInt fieldBits = value.extractBits(hwFieldWidth, hwOffset);
        APInt converted = convertHWToLLVMLayout(
            fieldBits, hwElements[i].type, llvmBody[i]);
        converted = adjustAPIntWidth(converted, llvmFieldWidth);
        result.insertBits(converted, llvmOffset);
        llvmOffset += llvmFieldWidth;
      }
      return adjustAPIntWidth(result, llvmWidth);
    }
  }

  if (auto hwArrayType = dyn_cast<hw::ArrayType>(hwType)) {
    if (auto llvmArrayType = dyn_cast<LLVM::LLVMArrayType>(llvmType)) {
      unsigned hwElemWidth = getTypeWidth(hwArrayType.getElementType());
      unsigned llvmElemWidth = getTypeWidth(llvmArrayType.getElementType());
      unsigned numElements = hwArrayType.getNumElements();
      unsigned llvmElements = llvmArrayType.getNumElements();
      size_t count = std::min(numElements, llvmElements);

      APInt result = APInt::getZero(llvmWidth);
      for (size_t i = 0; i < count; ++i) {
        unsigned hwOffset = (numElements - 1 - i) * hwElemWidth;
        unsigned llvmOffset = i * llvmElemWidth;
        APInt fieldBits = value.extractBits(hwElemWidth, hwOffset);
        APInt converted = convertHWToLLVMLayout(
            fieldBits, hwArrayType.getElementType(),
            llvmArrayType.getElementType());
        converted = adjustAPIntWidth(converted, llvmElemWidth);
        result.insertBits(converted, llvmOffset);
      }
      return adjustAPIntWidth(result, llvmWidth);
    }
  }

  return adjustAPIntWidth(value, llvmWidth);
}

std::optional<llvm::APInt>
LLHDProcessInterpreter::getEncodedUnknownForType(Type type) const {
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return getEncodedUnknownForType(refType.getNestedType());

  if (auto structType = dyn_cast<hw::StructType>(type)) {
    auto elements = structType.getElements();
    unsigned totalWidth = getTypeWidth(structType);

    if (elements.size() == 2 &&
        elements[0].name.getValue() == "value" &&
        elements[1].name.getValue() == "unknown") {
      unsigned unknownWidth = getTypeWidth(elements[1].type);
      APInt result(totalWidth, 0);
      if (unknownWidth == 0)
        return result;
      APInt unknownMask = APInt::getLowBitsSet(totalWidth, unknownWidth);
      result |= unknownMask;
      return result;
    }

    APInt result(totalWidth, 0);
    unsigned bitOffset = totalWidth;
    for (auto element : elements) {
      auto fieldOpt = getEncodedUnknownForType(element.type);
      if (!fieldOpt)
        return std::nullopt;
      unsigned fieldWidth = getTypeWidth(element.type);
      bitOffset -= fieldWidth;
      APInt fieldBits = fieldOpt->zextOrTrunc(fieldWidth);
      result.insertBits(fieldBits, bitOffset);
    }
    return result;
  }

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    unsigned elemWidth = getTypeWidth(arrayType.getElementType());
    unsigned count = arrayType.getNumElements();
    unsigned totalWidth = elemWidth * count;
    auto elemOpt = getEncodedUnknownForType(arrayType.getElementType());
    if (!elemOpt)
      return std::nullopt;
    APInt elemBits = elemOpt->zextOrTrunc(elemWidth);
    APInt result(totalWidth, 0);
    for (unsigned i = 0; i < count; ++i) {
      unsigned offset = (count - 1 - i) * elemWidth;
      result.insertBits(elemBits, offset);
    }
    return result;
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Signal Registry Bridge
//===----------------------------------------------------------------------===//

namespace {
/// Static pointer to the current interpreter for callback access.
/// This is needed because the MooreRuntime callbacks are C-style function
/// pointers that don't support closures.
LLHDProcessInterpreter *currentInterpreter = nullptr;

/// Callback for reading a signal value from the ProcessScheduler.
int64_t signalReadCallback(MooreSignalHandle handle, void *userData) {
  auto *scheduler = static_cast<ProcessScheduler *>(userData);
  if (!scheduler)
    return 0;

  SignalId sigId = static_cast<SignalId>(handle);
  const SignalValue &value = scheduler->getSignalValue(sigId);
  return static_cast<int64_t>(value.getValue());
}

/// Callback for writing (depositing) a signal value.
int32_t signalWriteCallback(MooreSignalHandle handle, int64_t value,
                            void *userData) {
  auto *scheduler = static_cast<ProcessScheduler *>(userData);
  if (!scheduler)
    return 0;

  SignalId sigId = static_cast<SignalId>(handle);
  // Get the signal's width from the current value
  const SignalValue &currentVal = scheduler->getSignalValue(sigId);
  SignalValue newVal(static_cast<uint64_t>(value), currentVal.getWidth());
  scheduler->updateSignal(sigId, newVal);
  return 1;
}

/// Callback for forcing a signal value.
/// Forces override normal signal updates until released.
int32_t signalForceCallback(MooreSignalHandle handle, int64_t value,
                            void *userData) {
  auto *scheduler = static_cast<ProcessScheduler *>(userData);
  if (!scheduler)
    return 0;

  SignalId sigId = static_cast<SignalId>(handle);
  // Get the signal's width from the current value
  const SignalValue &currentVal = scheduler->getSignalValue(sigId);
  SignalValue newVal(static_cast<uint64_t>(value), currentVal.getWidth());

  // Update the signal value (force tracking is done in MooreRuntime)
  scheduler->updateSignal(sigId, newVal);
  return 1;
}

/// Callback for releasing a forced signal.
int32_t signalReleaseCallback(MooreSignalHandle handle, void *userData) {
  // Release tracking is handled in MooreRuntime forcedSignals map
  // Just return success - the signal will resume normal operation
  (void)handle;
  (void)userData;
  return 1;
}
} // namespace

void LLHDProcessInterpreter::exportSignalsToRegistry() {
  // Clear any existing registrations from previous runs
  __moore_signal_registry_clear();
  __moore_signal_registry_clear_all_forced();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Exporting "
                          << signalIdToName.size()
                          << " signals to MooreRuntime registry\n");

  // Export each signal with its hierarchical path
  for (const auto &entry : signalIdToName) {
    SignalId sigId = entry.first;
    const std::string &signalName = entry.second;

    // Build hierarchical path: moduleName.signalName
    std::string hierarchicalPath;
    if (!moduleName.empty()) {
      hierarchicalPath = moduleName + "." + signalName;
    } else {
      hierarchicalPath = signalName;
    }

    // Get the signal width from the scheduler
    const SignalValue &value = scheduler.getSignalValue(sigId);
    uint32_t width = value.getWidth();

    // Register the signal with both hierarchical and simple paths
    __moore_signal_registry_register(hierarchicalPath.c_str(),
                                     static_cast<MooreSignalHandle>(sigId),
                                     width);

    // Also register with just the signal name for simpler access
    __moore_signal_registry_register(signalName.c_str(),
                                     static_cast<MooreSignalHandle>(sigId),
                                     width);

    LLVM_DEBUG(llvm::dbgs() << "  Exported '" << hierarchicalPath
                            << "' (ID=" << sigId << ", width=" << width
                            << ")\n");
  }

  // Set up the accessor callbacks
  setupRegistryAccessors();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Signal registry now has "
                          << __moore_signal_registry_count() << " entries\n");
}

void LLHDProcessInterpreter::setupRegistryAccessors() {
  // Store current interpreter for static callbacks
  currentInterpreter = this;

  // Set up the accessor callbacks with the ProcessScheduler as user data
  __moore_signal_registry_set_accessor(
      signalReadCallback,   // Read callback
      signalWriteCallback,  // Write/deposit callback
      signalForceCallback,  // Force callback
      signalReleaseCallback, // Release callback
      &scheduler             // User data (ProcessScheduler pointer)
  );

  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: Registry accessors configured, "
             << "connected=" << __moore_signal_registry_is_connected() << "\n");
}

//===----------------------------------------------------------------------===//
// Process Registration
//===----------------------------------------------------------------------===//

LogicalResult
LLHDProcessInterpreter::registerProcesses(const DiscoveredOps &ops) {
  // Register all pre-discovered llhd.process operations (no walk() needed)
  for (llhd::ProcessOp processOp : ops.processes) {
    LLVM_DEBUG(llvm::dbgs() << "  Found llhd.process op (numResults="
                            << processOp.getNumResults() << ")\n");
    registerProcess(processOp);
  }

  // Handle pre-discovered llhd.combinational operations
  for ([[maybe_unused]] llhd::CombinationalOp combOp : ops.combinationals) {
    // TODO: Handle combinational processes in Phase 1B
    LLVM_DEBUG(llvm::dbgs() << "  Found combinational process (TODO)\n");
  }

  // Register all pre-discovered seq.initial operations (no walk() needed)
  for (seq::InitialOp initialOp : ops.initials) {
    registerInitialBlock(initialOp);
  }

  // Handle pre-discovered module-level llhd.drv operations (no walk() needed)
  // Note: The iterative discovery already filters for module-level drives
  for (llhd::DriveOp driveOp : ops.moduleDrives) {
    registerModuleDrive(driveOp, 0, InstanceInputMapping{});
  }

  return success();
}

ProcessId LLHDProcessInterpreter::registerProcess(llhd::ProcessOp processOp) {
  // Generate a process name
  std::string name = "llhd_process_" + std::to_string(processStates.size());

  // Create the execution state for this process
  ProcessExecutionState state(processOp);
  state.cacheable = isProcessCacheableBody(processOp);

  // Register with the scheduler, then bind the callback to the real ID.
  ProcessId procId = scheduler.registerProcess(name, []() {});
  if (auto *process = scheduler.getProcess(procId))
    process->setCallback([this, procId]() { executeProcess(procId); });

  // Store the state
  state.currentBlock = &processOp.getBody().front();
  state.currentOp = state.currentBlock->begin();
  registerProcessState(procId, std::move(state));
  opToProcessId[processOp.getOperation()] = procId;

  LLVM_DEBUG(llvm::dbgs() << "  Registered process '" << name << "' with ID "
                          << procId << "\n");

  // Schedule the process to run at time 0 (initialization)
  scheduler.scheduleProcess(procId, SchedulingRegion::Active);

  return procId;
}

ProcessId LLHDProcessInterpreter::registerInitialBlock(seq::InitialOp initialOp) {
  // Generate a process name for the initial block
  std::string name = "seq_initial_" + std::to_string(processStates.size());

  // Create the execution state for this initial block
  ProcessExecutionState state(initialOp);
  state.cacheable = false;

  // Register with the scheduler, then bind the callback to the real ID.
  ProcessId procId = scheduler.registerProcess(name, []() {});
  if (auto *process = scheduler.getProcess(procId))
    process->setCallback([this, procId]() { executeProcess(procId); });

  // Store the state - initial blocks have a body with a single block
  state.currentBlock = initialOp.getBodyBlock();
  state.currentOp = state.currentBlock->begin();
  registerProcessState(procId, std::move(state));
  opToProcessId[initialOp.getOperation()] = procId;

  LLVM_DEBUG(llvm::dbgs() << "  Registered initial block '" << name
                          << "' with ID " << procId << "\n");

  // Schedule the initial block to run at time 0 (initialization)
  scheduler.scheduleProcess(procId, SchedulingRegion::Active);

  return procId;
}

void LLHDProcessInterpreter::registerModuleDrive(
    llhd::DriveOp driveOp, InstanceId instanceId,
    const InstanceInputMapping &inputMap) {
  ScopedInstanceContext instScope(*this, instanceId);
  ScopedInputValueMap inputScope(*this, inputMap);
  // Module-level drives need special handling:
  // The drive value comes from process results which are populated when
  // the process executes llhd.wait yield or llhd.halt with yield operands.
  //
  // For each module-level drive, we need to:
  // 1. Track the drive operation
  // 2. Identify the source process (if the value comes from a process result)
  // 3. Schedule the drive when the process yields

  Value driveValue = driveOp.getValue();
  InstanceId driveInstance = instanceId;

  // Resolve the drive value through inputValueMap if it's a block argument.
  // This handles the case where a child module's input is mapped to a parent's
  // process result, e.g.: hw.instance @child(in: %proc_val) where %proc_val is
  // the result of an llhd.process in the parent module.
  if (auto arg = dyn_cast<mlir::BlockArgument>(driveValue)) {
    Value mappedValue;
    InstanceId mappedInstance = instanceId;
    if (lookupInputMapping(arg, mappedValue, mappedInstance)) {
      driveValue = mappedValue;
      driveInstance = mappedInstance;
    }
  }

  // Check if the drive value comes from a process result
  if (auto processOp = driveValue.getDefiningOp<llhd::ProcessOp>()) {
    // Find the process ID for this process
    ProcessId procId = InvalidProcessId;
    if (driveInstance != 0) {
      auto ctxIt = instanceOpToProcessId.find(driveInstance);
      if (ctxIt != instanceOpToProcessId.end()) {
        auto procIt = ctxIt->second.find(processOp.getOperation());
        if (procIt != ctxIt->second.end())
          procId = procIt->second;
      }
    }
    if (procId == InvalidProcessId) {
      auto procIt = opToProcessId.find(processOp.getOperation());
      if (procIt != opToProcessId.end())
        procId = procIt->second;
    }
    if (procId != InvalidProcessId) {

      // Store this drive for later execution when the process yields
      moduleDrives.push_back({driveOp, procId, instanceId, inputMap});

      LLVM_DEBUG(llvm::dbgs() << "  Registered module-level drive connected to "
                              << "process " << procId << "\n");
    }
  } else {
    // For non-process-connected drives, schedule them immediately
    // This handles static/constant drives at module level
    LLVM_DEBUG(llvm::dbgs() << "  Found module-level drive (static)\n");
    // These will be handled during initialization
    staticModuleDrives.push_back({driveOp, instanceId, inputMap});
  }
}

void LLHDProcessInterpreter::executeModuleDrives(ProcessId procId) {
  // Recursion guard - prevent re-entrant calls during value evaluation
  static thread_local llvm::DenseSet<ProcessId> inProgress;
  if (!inProgress.insert(procId).second) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping recursive executeModuleDrives for proc "
               << procId << "\n");
    return;
  }
  auto cleanup = llvm::make_scope_exit([&]() { inProgress.erase(procId); });

  auto stateIt = processStates.find(procId);
  if (stateIt == processStates.end())
    return;

  LLVM_DEBUG(llvm::dbgs() << "executeModuleDrives proc " << procId
                          << " moduleDrives=" << moduleDrives.size() << "\n");

  // Execute all module-level drives connected to this process
  size_t drivesProcessed = 0;
  size_t driveIdx = 0;
  for (auto &entry : moduleDrives) {
    ++driveIdx;
    if (entry.procId != procId)
      continue;
    ++drivesProcessed;
    ScopedInstanceContext instScope(*this, entry.instanceId);
    ScopedInputValueMap inputScope(*this, entry.inputMap);
    llhd::DriveOp driveOp = entry.driveOp;

    // Get the signal ID
    SignalId sigId = getSignalId(driveOp.getSignal());
    if (sigId == 0) {
      LLVM_DEBUG(llvm::dbgs() << "  Error: Unknown signal in module drive\n");
      continue;
    }

    // Check enable condition if present
    if (driveOp.getEnable()) {
      InterpretedValue enableVal = getValue(procId, driveOp.getEnable());
      if (enableVal.isX() || enableVal.getUInt64() == 0) {
        LLVM_DEBUG(llvm::dbgs() << "  Module drive disabled\n");
        continue;
      }
    }

    // Get the value to drive from the process result
    InterpretedValue driveVal = getValue(procId, driveOp.getValue());

    // Get the delay time
    SimTime delay = convertTimeValue(procId, driveOp.getTime());

    // Calculate the target time
    SimTime currentTime = scheduler.getCurrentTime();
    SimTime targetTime = currentTime.advanceTime(delay.realTime);
    if (delay.deltaStep > 0) {
      targetTime.deltaStep = delay.deltaStep;
    }

    LLVM_DEBUG(llvm::dbgs() << "  Module drive: scheduling update to signal "
                            << sigId << " value="
                            << (driveVal.isX() ? "X"
                                                : std::to_string(driveVal.getUInt64()))
                            << " at time " << targetTime.realTime << " fs\n");

    // Schedule the signal update
    SignalValue newVal = driveVal.toSignalValue();
    if (scheduler.getSignalValue(sigId) == newVal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Module drive unchanged for signal " << sigId << "\n");
      continue;
    }
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::NBA,
        Event([this, sigId, newVal]() {
          scheduler.updateSignal(sigId, newVal);
        }));
  }
  (void)drivesProcessed;
}

void LLHDProcessInterpreter::executeInstanceOutputUpdates(ProcessId procId) {
  for (const auto &entry : instanceOutputUpdates) {
    if (entry.processIds.empty())
      continue;
    bool dependsOnProc = false;
    for (ProcessId id : entry.processIds) {
      if (id == procId) {
        dependsOnProc = true;
        break;
      }
    }
    if (dependsOnProc)
      scheduleInstanceOutputUpdate(entry.signalId, entry.outputValue,
                                   entry.instanceId,
                                   entry.inputMap.empty()
                                       ? nullptr
                                       : &entry.inputMap);
  }
}

void LLHDProcessInterpreter::registerContinuousAssignments(
    hw::HWModuleOp hwModule, InstanceId instanceId,
    const InstanceInputMapping &inputMap) {
  // For each static module-level drive, we need to:
  // 1. Find which signals the drive value depends on (via llhd.prb)
  // 2. Create a combinational process that re-executes when those signals change
  // 3. The process evaluates the drive value and schedules the signal update

  for (const auto &entry : staticModuleDrives) {
    if (entry.instanceId != instanceId)
      continue;
    llhd::DriveOp driveOp = entry.driveOp;
    if (driveOp->getParentOfType<hw::HWModuleOp>() != hwModule)
      continue;
    // Find the signal being driven
    const InstanceInputMapping &driveInputMap =
        entry.inputMap.empty() ? inputMap : entry.inputMap;
    ScopedInstanceContext instScope(*this, instanceId);
    ScopedInputValueMap inputScope(*this, driveInputMap);

    SignalId targetSigId = getSignalId(driveOp.getSignal());
    if (targetSigId == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: Unknown target signal in continuous assignment\n");
      continue;
    }

    llvm::SmallVector<ProcessId, 4> processIds;
    collectProcessIds(driveOp.getValue(), processIds);
    if (driveOp.getEnable())
      collectProcessIds(driveOp.getEnable(), processIds);

    llvm::SmallVector<SignalId, 4> sourceSignals;
    collectSignalIds(driveOp.getValue(), sourceSignals);
    if (driveOp.getEnable())
      collectSignalIds(driveOp.getEnable(), sourceSignals);

    if (!processIds.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Continuous assignment depends on "
                 << processIds.size() << " process result(s)\n");
      for (ProcessId procId : processIds) {
        bool alreadyRegistered = false;
        for (const auto &entry : moduleDrives) {
          if (entry.driveOp == driveOp && entry.procId == procId &&
              entry.instanceId == instanceId) {
            alreadyRegistered = true;
            break;
          }
        }
        if (!alreadyRegistered)
          moduleDrives.push_back({driveOp, procId, instanceId, driveInputMap});
      }
    }

    if (sourceSignals.empty() && processIds.empty()) {
      // No source signals - this is a constant drive, execute once at init
      LLVM_DEBUG(llvm::dbgs()
                 << "  Constant continuous assignment to signal " << targetSigId
                 << "\n");
      executeContinuousAssignment(driveOp);
      continue;
    }

    if (sourceSignals.empty()) {
      // Process-result-only drive; updates are handled on process yields.
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment to signal "
                            << targetSigId << " depends on " << sourceSignals.size()
                            << " source signals\n");

    // Generate a unique process name
    std::string processName =
        "cont_assign_" + std::to_string(targetSigId);

    // Register a combinational process that executes the drive
    ProcessId procId = scheduler.registerProcess(
        processName,
        [this, driveOp, instanceId, driveInputMap]() {
          ScopedInstanceContext scope(*this, instanceId);
          ScopedInputValueMap inputScope(*this, driveInputMap);
          executeContinuousAssignment(driveOp);
        });

    // Mark as combinational and add sensitivities to all source signals
    auto *process = scheduler.getProcess(procId);
    if (process) {
      process->setCombinational(true);
      for (SignalId srcSigId : sourceSignals) {
        scheduler.addSensitivity(procId, srcSigId);
        LLVM_DEBUG(llvm::dbgs()
                   << "    Added sensitivity to signal " << srcSigId << "\n");
      }
    }

    // Execute once at initialization
    executeContinuousAssignment(driveOp);

    // Schedule the process to run at time 0 to ensure initial state is correct
    scheduler.scheduleProcess(procId, SchedulingRegion::Active);
  }
}

SignalId LLHDProcessInterpreter::resolveSignalId(mlir::Value value) const {
  if (SignalId sigId = getSignalId(value))
    return sigId;
  auto instMapIt = instanceOutputMap.find(activeInstanceId);
  if (instMapIt != instanceOutputMap.end()) {
    auto instIt = instMapIt->second.find(value);
    if (instIt != instMapIt->second.end()) {
      const auto &info = instIt->second;
      if (info.inputMap.empty())
        return resolveSignalId(info.outputValue);
      ScopedInputValueMap scope(
        *const_cast<LLHDProcessInterpreter *>(this), info.inputMap);
      ScopedInstanceContext instScope(
          *const_cast<LLHDProcessInterpreter *>(this), info.instanceId);
      return resolveSignalId(info.outputValue);
    }
  }
  if (auto arg = dyn_cast<mlir::BlockArgument>(value)) {
    Value mappedValue;
    InstanceId mappedInstance = activeInstanceId;
    if (lookupInputMapping(arg, mappedValue, mappedInstance) &&
        mappedValue != value) {
      ScopedInstanceContext scope(
          *const_cast<LLHDProcessInterpreter *>(this), mappedInstance);
      return resolveSignalId(mappedValue);
    }
  }
  // NOTE: We explicitly do NOT trace through llhd::ProbeOp here.
  // The result of a probe is a VALUE (not a signal reference).
  // Operations on probe results should treat them as values, not signals.
  // This is important for cases like:
  //   %ptr = llhd.prb %sig : !llvm.ptr  // %ptr is a VALUE (pointer address)
  //   llvm.store %val, %ptr            // Should write to MEMORY, not drive %sig
  // Trace through UnrealizedConversionCastOp - these are used to convert
  // between !llhd.ref types and LLVM pointer types when passing signals
  // as function arguments.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      return resolveSignalId(castOp.getInputs()[0]);
    }
  }
  // Note: arith.select on ref types is handled dynamically in interpretProbe
  // and interpretDrive since it requires evaluating the condition at runtime.
  // resolveSignalId is a const function that cannot evaluate conditions, so
  // we return 0 here and let the caller handle it.
  return 0;
}

bool LLHDProcessInterpreter::lookupInputMapping(
    mlir::BlockArgument arg, mlir::Value &mappedValue,
    InstanceId &mappedInstance) const {
  auto argIt = inputValueMap.find(arg);
  if (argIt != inputValueMap.end()) {
    mappedValue = argIt->second;
    auto instIt = inputValueInstanceMap.find(arg);
    mappedInstance =
        (instIt != inputValueInstanceMap.end()) ? instIt->second
                                                : activeInstanceId;
    return true;
  }
  auto instanceIt = instanceInputMaps.find(activeInstanceId);
  if (instanceIt == instanceInputMaps.end())
    return false;
  for (const auto &entry : instanceIt->second) {
    if (entry.arg != arg)
      continue;
    mappedValue = entry.value;
    mappedInstance = entry.instanceId;
    return true;
  }
  return false;
}

void LLHDProcessInterpreter::collectSignalIds(
    mlir::Value value, llvm::SmallVectorImpl<SignalId> &signals) const {
  struct WorkItem {
    mlir::Value value;
    const InstanceInputMapping *inputMap = nullptr;
    InstanceId instanceId = 0;
  };

  llvm::SmallVector<WorkItem, 8> worklist;
  llvm::SmallVector<WorkItem, 16> visited;
  worklist.push_back({value, nullptr, activeInstanceId});

  while (!worklist.empty()) {
    WorkItem item = worklist.pop_back_val();
    ScopedInstanceContext instScope(
        *const_cast<LLHDProcessInterpreter *>(this), item.instanceId);
    bool seen = false;
    for (const auto &entry : visited) {
      if (entry.value == item.value && entry.inputMap == item.inputMap &&
          entry.instanceId == item.instanceId) {
        seen = true;
        break;
      }
    }
    if (seen)
      continue;
    visited.push_back(item);

    if (SignalId sigId = getSignalId(item.value)) {
      signals.push_back(sigId);
      continue;
    }

    auto instMapIt = instanceOutputMap.find(item.instanceId);
    if (instMapIt != instanceOutputMap.end()) {
      auto instIt = instMapIt->second.find(item.value);
      if (instIt != instMapIt->second.end()) {
        const auto &info = instIt->second;
        worklist.push_back({info.outputValue, &info.inputMap, info.instanceId});
        continue;
      }
    }

    if (auto arg = dyn_cast<mlir::BlockArgument>(item.value)) {
      if (item.inputMap) {
        for (const auto &entry : *item.inputMap) {
          if (entry.arg == arg) {
            worklist.push_back({entry.value, nullptr, entry.instanceId});
            break;
          }
        }
      }
      Value mappedValue;
      InstanceId mappedInstance = item.instanceId;
      if (lookupInputMapping(arg, mappedValue, mappedInstance)) {
        worklist.push_back({mappedValue, nullptr, mappedInstance});
        continue;
      }
    }

    if (auto toClock = item.value.getDefiningOp<seq::ToClockOp>()) {
      worklist.push_back({toClock.getInput(), item.inputMap, item.instanceId});
      continue;
    }

    if (auto fromClock = item.value.getDefiningOp<seq::FromClockOp>()) {
      worklist.push_back({fromClock.getInput(), item.inputMap, item.instanceId});
      continue;
    }

    if (auto combOp = item.value.getDefiningOp<llhd::CombinationalOp>()) {
      // Add all operands from operations inside the combinational block to the
      // worklist. This avoids recursion through collectSignalIdsFromCombinational
      // which can cause stack overflow on large designs (e.g., OpenTitan IPs).
      combOp.walk([&](Operation *op) {
        for (Value operand : op->getOperands())
          worklist.push_back({operand, item.inputMap, item.instanceId});
      });
      continue;
    }

    if (Operation *defOp = item.value.getDefiningOp()) {
      for (Value operand : defOp->getOperands())
        worklist.push_back({operand, item.inputMap, item.instanceId});
    }
  }
}

void LLHDProcessInterpreter::collectProcessIds(
    mlir::Value value, llvm::SmallVectorImpl<ProcessId> &processIds) const {
  struct WorkItem {
    mlir::Value value;
    const InstanceInputMapping *inputMap = nullptr;
    InstanceId instanceId = 0;
  };

  llvm::SmallVector<WorkItem, 8> worklist;
  llvm::SmallVector<WorkItem, 16> visited;
  llvm::DenseSet<ProcessId> seen;
  worklist.push_back({value, nullptr, activeInstanceId});

  while (!worklist.empty()) {
    WorkItem item = worklist.pop_back_val();
    ScopedInstanceContext instScope(
        *const_cast<LLHDProcessInterpreter *>(this), item.instanceId);
    bool seenValue = false;
    for (const auto &entry : visited) {
      if (entry.value == item.value && entry.inputMap == item.inputMap &&
          entry.instanceId == item.instanceId) {
        seenValue = true;
        break;
      }
    }
    if (seenValue)
      continue;
    visited.push_back(item);

    auto instMapIt = instanceOutputMap.find(item.instanceId);
    if (instMapIt != instanceOutputMap.end()) {
      auto instIt = instMapIt->second.find(item.value);
      if (instIt != instMapIt->second.end()) {
        const auto &info = instIt->second;
        worklist.push_back({info.outputValue, &info.inputMap, info.instanceId});
        continue;
      }
    }

    if (auto arg = dyn_cast<mlir::BlockArgument>(item.value)) {
      if (item.inputMap) {
        for (const auto &entry : *item.inputMap) {
          if (entry.arg == arg) {
            worklist.push_back({entry.value, nullptr, entry.instanceId});
            break;
          }
        }
      }
      Value mappedValue;
      InstanceId mappedInstance = item.instanceId;
      if (lookupInputMapping(arg, mappedValue, mappedInstance)) {
        worklist.push_back({mappedValue, nullptr, mappedInstance});
        continue;
      }
    }

    if (auto result = dyn_cast<OpResult>(item.value)) {
      if (auto processOp = dyn_cast<llhd::ProcessOp>(result.getOwner())) {
        ProcessId procId = InvalidProcessId;
        if (item.instanceId != 0) {
          auto ctxIt = instanceOpToProcessId.find(item.instanceId);
          if (ctxIt != instanceOpToProcessId.end()) {
            auto procIt = ctxIt->second.find(processOp.getOperation());
            if (procIt != ctxIt->second.end())
              procId = procIt->second;
          }
        }
        if (procId == InvalidProcessId) {
          auto procIt = opToProcessId.find(processOp.getOperation());
          if (procIt != opToProcessId.end())
            procId = procIt->second;
        }
        if (procId != InvalidProcessId && seen.insert(procId).second)
          processIds.push_back(procId);
        continue;
      }
    }

    if (Operation *defOp = item.value.getDefiningOp()) {
      for (Value operand : defOp->getOperands())
        worklist.push_back({operand, item.inputMap, item.instanceId});
    }
  }
}

// NOTE: collectSignalIdsFromCombinational has been inlined into collectSignalIds
// to avoid stack overflow from mutual recursion on large designs (e.g., OpenTitan
// hmac_reg_top, rv_timer_reg_top, spi_host_reg_top). The logic now adds operands
// directly to the worklist when encountering a CombinationalOp.

void LLHDProcessInterpreter::registerFirRegs(const DiscoveredOps &ops,
                                             InstanceId instanceId,
                                             const InstanceInputMapping &inputMap) {
  ScopedInstanceContext instScope(*this, instanceId);
  ScopedInputValueMap inputScope(*this, inputMap);
  auto &firRegMap =
      (instanceId == 0) ? firRegStates : instanceFirRegStates[instanceId];
  auto &signalMap =
      (instanceId == 0) ? valueToSignal : instanceValueToSignal[instanceId];
  // Register all pre-discovered seq.firreg operations (no walk() needed)
  for (seq::FirRegOp regOp : ops.firRegs) {
    if (firRegMap.contains(regOp.getOperation()))
      continue;

    std::string baseName;
    if (auto nameAttr = regOp.getNameAttr())
      baseName = nameAttr.str();
    else
      baseName = "firreg_" + std::to_string(firRegMap.size());
    std::string name = baseName;
    if (instanceId != 0)
      name = "inst" + std::to_string(instanceId) + "." + baseName;

    unsigned width = getTypeWidth(regOp.getType());
    SignalId sigId =
        scheduler.registerSignal(name, width, getSignalEncoding(regOp.getType()));
    signalMap[regOp.getResult()] = sigId;
    signalIdToName[sigId] = name;
    signalIdToType[sigId] = unwrapSignalType(regOp.getType());

    bool initSet = false;
    if (regOp.hasReset() && regOp.getIsAsync()) {
      InterpretedValue resetVal = evaluateContinuousValue(regOp.getReset());
      if (!resetVal.isX() && resetVal.getUInt64() != 0) {
        InterpretedValue resetValue =
            evaluateContinuousValue(regOp.getResetValue());
        scheduler.updateSignal(sigId, resetValue.toSignalValue());
        initSet = true;
      }
    }

    if (!initSet && regOp.hasPresetValue()) {
      auto preset = regOp.getPresetAttr();
      if (preset) {
        SignalValue initVal(preset.getValue());
        scheduler.updateSignal(sigId, initVal);
        initSet = true;
      }
    }

    if (!initSet) {
      scheduler.updateSignal(sigId, SignalValue::makeX(width));
    }

    FirRegState state;
    state.signalId = sigId;
    state.instanceId = instanceId;
    state.inputMap = inputMap;
    firRegMap[regOp.getOperation()] = state;

    std::string procName = "firreg_" + name;
    ProcessId procId = scheduler.registerProcess(
        procName, [this, regOp, instanceId, inputMap]() {
          ScopedInstanceContext scope(*this, instanceId);
          ScopedInputValueMap inputScope(*this, inputMap);
          executeFirReg(regOp, instanceId);
        });
    if (auto *process = scheduler.getProcess(procId)) {
      // Schedule firreg updates after combinational propagation in the time slot.
      process->setPreferredRegion(SchedulingRegion::NBA);
    }

    // Track any clock edge so we can update prevClock on negedges too.
    // The actual posedge detection is done inside executeFirReg.
    llvm::SmallVector<SignalId, 4> clkSignals;
    collectSignalIds(regOp.getClk(), clkSignals);
    for (SignalId sig : clkSignals)
      scheduler.addSensitivity(procId, sig, EdgeType::AnyEdge);

    if (regOp.hasReset() && regOp.getIsAsync()) {
      llvm::SmallVector<SignalId, 4> rstSignals;
      collectSignalIds(regOp.getReset(), rstSignals);
      for (SignalId sig : rstSignals)
        scheduler.addSensitivity(procId, sig, EdgeType::AnyEdge);
    }

    scheduler.scheduleProcess(procId, SchedulingRegion::Active);
  }
}

void LLHDProcessInterpreter::executeFirReg(seq::FirRegOp regOp,
                                           InstanceId instanceId) {
  auto &firRegMap =
      (instanceId == 0) ? firRegStates : instanceFirRegStates[instanceId];
  auto it = firRegMap.find(regOp.getOperation());
  if (it == firRegMap.end())
    return;

  FirRegState &state = it->second;
  ScopedInstanceContext instScope(*this, state.instanceId);
  ScopedInputValueMap inputScope(*this, state.inputMap);

  InterpretedValue clkVal = evaluateContinuousValue(regOp.getClk());

  bool clockPosedge = false;
  if (!clkVal.isX()) {
    if (!state.hasPrevClock) {
      state.prevClock = clkVal;
      state.hasPrevClock = true;
    } else if (!state.prevClock.isX()) {
      uint64_t prev = state.prevClock.getUInt64();
      uint64_t curr = clkVal.getUInt64();
      clockPosedge = (prev == 0 && curr != 0);
      state.prevClock = clkVal;
    } else {
      state.prevClock = clkVal;
    }
  }

  bool hasReset = regOp.hasReset();
  bool resetActive = false;
  bool resetUnknown = false;
  InterpretedValue resetVal;
  if (hasReset) {
    resetVal = evaluateContinuousValue(regOp.getReset());
    if (resetVal.isX()) {
      resetUnknown = true;
    } else {
      resetActive = resetVal.getUInt64() != 0;
    }
  }

  InterpretedValue newVal;
  bool doUpdate = false;

  if (hasReset && resetUnknown &&
      (regOp.getIsAsync() || clockPosedge)) {
    newVal = InterpretedValue::makeX(getTypeWidth(regOp.getType()));
    doUpdate = true;
  } else if (hasReset && regOp.getIsAsync() && resetActive) {
    newVal = evaluateContinuousValue(regOp.getResetValue());
    doUpdate = true;
  } else if (clockPosedge) {
    if (hasReset && resetActive && !regOp.getIsAsync()) {
      newVal = evaluateContinuousValue(regOp.getResetValue());
    } else {
      newVal = evaluateContinuousValue(regOp.getNext());
    }
    doUpdate = true;
  }

  if (!doUpdate)
    return;

  scheduler.updateSignal(state.signalId, newVal.toSignalValue());
}

void LLHDProcessInterpreter::executeContinuousAssignment(
    llhd::DriveOp driveOp) {
  // Get the signal being driven
  SignalId targetSigId = getSignalId(driveOp.getSignal());
  if (targetSigId == 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Error: Unknown signal in continuous assignment\n");
    return;
  }

  // Evaluate the drive value by interpreting the defining operation chain
  // We use process ID 0 as a dummy since continuous assignments don't have
  // their own process state - they evaluate values directly from signal state
  if (driveOp.getEnable()) {
    InterpretedValue enableVal = evaluateContinuousValue(driveOp.getEnable());
    if (enableVal.isX() || enableVal.getUInt64() == 0) {
      LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment disabled\n");
      return;
    }
  }
  InterpretedValue driveVal = evaluateContinuousValue(driveOp.getValue());

  // Get the delay time
  SimTime delay;
  if (auto timeOp = driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>()) {
    delay = convertTime(timeOp.getValueAttr());
  } else {
    // Default to epsilon delay
    delay = SimTime(0, 0, 1);
  }

  // Calculate the target time
  SimTime currentTime = scheduler.getCurrentTime();
  SimTime targetTime = currentTime.advanceTime(delay.realTime);
  if (delay.deltaStep > 0) {
    targetTime.deltaStep = currentTime.deltaStep + delay.deltaStep;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Continuous assignment: scheduling update to signal "
                          << targetSigId << " value="
                          << (driveVal.isX() ? "X"
                                             : std::to_string(driveVal.getUInt64()))
                          << " at time " << targetTime.realTime << " fs\n");

  // Schedule the signal update
  SignalValue newVal = driveVal.toSignalValue();
  if (scheduler.getSignalValue(targetSigId) == newVal) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Continuous assignment unchanged for signal "
               << targetSigId << "\n");
    return;
  }
  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::Active,
      Event([this, targetSigId, newVal]() {
        scheduler.updateSignal(targetSigId, newVal);
      }));
}

void LLHDProcessInterpreter::scheduleInstanceOutputUpdate(
    SignalId signalId, mlir::Value outputValue, InstanceId instanceId,
    const InstanceInputMapping *inputMap) {
  ScopedInstanceContext instScope(*this, instanceId);
  if (inputMap && !inputMap->empty()) {
    ScopedInputValueMap scope(*this, *inputMap);
    InterpretedValue driveVal = evaluateContinuousValue(outputValue);
    SignalValue newVal = driveVal.toSignalValue();
    if (scheduler.getSignalValue(signalId) == newVal)
      return;
    scheduler.updateSignal(signalId, newVal);
    return;
  }
  InterpretedValue driveVal = evaluateContinuousValue(outputValue);
  SignalValue newVal = driveVal.toSignalValue();
  if (scheduler.getSignalValue(signalId) == newVal)
    return;

  scheduler.updateSignal(signalId, newVal);
}

bool LLHDProcessInterpreter::evaluateCombinationalOp(
    llhd::CombinationalOp combOp,
    llvm::SmallVectorImpl<InterpretedValue> &results) {
  results.clear();

  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId))
    tempProcId = nextTempProcId++;

  ProcessExecutionState tempState;
  tempState.processOrInitialOp = combOp.getOperation();
  tempState.currentBlock = &combOp.getBody().front();
  tempState.currentOp = tempState.currentBlock->begin();
  processStates[tempProcId] = std::move(tempState);

  bool sawYield = false;
  auto &state = processStates[tempProcId];
  for (Operation &op : *state.currentBlock) {
    if (auto yieldOp = dyn_cast<llhd::YieldOp>(&op)) {
      for (Value operand : yieldOp.getOperands())
        results.push_back(getValue(tempProcId, operand));
      sawYield = true;
      break;
    }

    if (failed(interpretOperation(tempProcId, &op))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: Failed to interpret combinational op\n");
      // Emit diagnostic for failed combinational op
      llvm::errs() << "circt-sim: Failed to interpret combinational op\n";
      llvm::errs() << "  Operation: ";
      op.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
      llvm::errs() << "\n";
      llvm::errs() << "  Location: " << op.getLoc() << "\n";
      break;
    }
  }

  processStates.erase(tempProcId);

  if (!sawYield) {
    results.clear();
    for (Type resultType : combOp.getResultTypes())
      results.push_back(InterpretedValue::makeX(getTypeWidth(resultType)));
    return false;
  }

  return true;
}

/// Evaluate a value for continuous assignments by reading from current signal
/// state.
InterpretedValue LLHDProcessInterpreter::evaluateContinuousValue(
    mlir::Value value) {
  return evaluateContinuousValueImpl(value);
}

InterpretedValue LLHDProcessInterpreter::evaluateContinuousValueImpl(
    mlir::Value value) {
  enum class EvalKind {
    None,
    Forward,
    StructExtract,
    ArrayGet,
    ArrayCreate,
    ArraySlice,
    ArrayConcat,
    StructCreate,
    StructInject,
    StructInjectLegacy,
    Bitcast,
    Xor,
    And,
    Or,
    ICmp,
    Mux,
    Concat,
    Extract,
    Add,
    Sub,
    Replicate,
    Parity,
    Shl,
    ShrU,
    ShrS,
    Mul,
    DivS,
    DivU,
    ModS,
    ModU
  };

  struct EvalFrame {
    mlir::Value value;
    EvalKind kind = EvalKind::None;
    unsigned stage = 0;
    mlir::Value aux;
  };

  // Track how many times each value has been pushed onto the evaluation stack.
  // In a DAG (directed acyclic graph), a value may be shared by multiple
  // consumers. We allow pushing a value up to 2 times: once for the original
  // reference and once for a shared dependency. The duplicate will be a no-op
  // when popped (already cached from the first evaluation). If a value is
  // pushed more than twice, it indicates a true combinational cycle.
  llvm::DenseMap<mlir::Value, unsigned> pushCount;
  llvm::DenseMap<mlir::Value, InterpretedValue> cache;
  llvm::SmallVector<EvalFrame, 64> stack;

  auto makeUnknown = [&](mlir::Value v) -> InterpretedValue {
    return InterpretedValue::makeX(getTypeWidth(v.getType()));
  };

  auto pushValue = [&](mlir::Value v) {
    if (!v)
      return;
    if (cache.find(v) != cache.end())
      return;
    auto &count = pushCount[v];
    if (count >= 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: Cycle detected in evaluateContinuousValue\n");
      return;
    }
    count++;
    stack.push_back(EvalFrame{v});
  };

  pushValue(value);

  while (!stack.empty()) {
    EvalFrame &frame = stack.back();
    mlir::Value current = frame.value;
    if (cache.find(current) != cache.end()) {
      stack.pop_back();
      continue;
    }

    auto finish = [&](InterpretedValue result) {
      cache[current] = result;
      stack.pop_back();
    };

    auto getCached = [&](mlir::Value v) -> InterpretedValue {
      auto it = cache.find(v);
      if (it != cache.end())
        return it->second;
      return makeUnknown(v);
    };

    if (frame.stage == 0) {
      // Check instanceOutputMap BEFORE getSignalId. Instance results are
      // registered as signals (for caching), but the signal value may be stale
      // when multiple combinational processes fire in response to the same
      // source signal change. By evaluating through the instance we always
      // compute the correct combinational value from current inputs.
      auto instMapIt = instanceOutputMap.find(activeInstanceId);
      if (instMapIt != instanceOutputMap.end()) {
        auto instIt = instMapIt->second.find(current);
        if (instIt != instMapIt->second.end()) {
          const auto &info = instIt->second;
          ScopedInstanceContext instScope(*this, info.instanceId);
          if (info.inputMap.empty()) {
            finish(evaluateContinuousValue(info.outputValue));
          } else {
            ScopedInputValueMap scope(*this, info.inputMap);
            finish(evaluateContinuousValue(info.outputValue));
          }
          continue;
        }
      }

      if (SignalId sigId = getSignalId(current)) {
        const SignalValue &sv = scheduler.getSignalValue(sigId);
        if (sv.isUnknown()) {
          if (auto encoded = getEncodedUnknownForType(current.getType()))
            finish(InterpretedValue(*encoded));
          else
            finish(makeUnknown(current));
        } else {
          finish(InterpretedValue::fromSignalValue(sv));
        }
        continue;
      }

      if (auto result = dyn_cast<OpResult>(current)) {
        if (auto processOp = dyn_cast<llhd::ProcessOp>(result.getOwner())) {
          ProcessId procId = InvalidProcessId;
          if (activeInstanceId != 0) {
            auto ctxIt = instanceOpToProcessId.find(activeInstanceId);
            if (ctxIt != instanceOpToProcessId.end()) {
              auto procIt = ctxIt->second.find(processOp.getOperation());
              if (procIt != ctxIt->second.end())
                procId = procIt->second;
            }
          }
          if (procId == InvalidProcessId) {
            auto procIt = opToProcessId.find(processOp.getOperation());
            if (procIt != opToProcessId.end())
              procId = procIt->second;
          }
          if (procId != InvalidProcessId) {
            auto stateIt = processStates.find(procId);
            if (stateIt != processStates.end()) {
              auto valIt = stateIt->second.valueMap.find(current);
              if (valIt != stateIt->second.valueMap.end()) {
                finish(valIt->second);
                continue;
              }
            }
          }
          finish(makeUnknown(current));
          continue;
        }
      }

      if (auto arg = dyn_cast<mlir::BlockArgument>(current)) {
        Value mappedValue;
        InstanceId mappedInstance = activeInstanceId;
        if (lookupInputMapping(arg, mappedValue, mappedInstance) &&
            mappedValue != current) {
          if (mappedInstance != activeInstanceId) {
            ScopedInstanceContext scope(*this, mappedInstance);
            finish(evaluateContinuousValue(mappedValue));
            continue;
          }
          frame.kind = EvalKind::Forward;
          frame.aux = mappedValue;
          frame.stage = 1;
          pushValue(frame.aux);
          continue;
        }
        SignalId sigId = getSignalId(arg);
        if (sigId != 0) {
          const SignalValue &sv = scheduler.getSignalValue(sigId);
          if (sv.isUnknown()) {
            if (auto encoded = getEncodedUnknownForType(arg.getType()))
              finish(InterpretedValue(*encoded));
            else
              finish(makeUnknown(current));
          } else {
            finish(InterpretedValue::fromSignalValue(sv));
          }
          continue;
        }
      }

      if (auto regOp = current.getDefiningOp<seq::FirRegOp>()) {
        SignalId sigId = getSignalId(regOp.getResult());
        if (sigId != 0) {
          const SignalValue &sv = scheduler.getSignalValue(sigId);
          finish(InterpretedValue::fromSignalValue(sv));
        } else {
          finish(makeUnknown(current));
        }
        continue;
      }

      if (auto combOp = current.getDefiningOp<llhd::CombinationalOp>()) {
        llvm::SmallVector<InterpretedValue, 4> results;
        (void)evaluateCombinationalOp(combOp, results);
        auto result = dyn_cast<OpResult>(current);
        if (result && result.getResultNumber() < results.size())
          finish(results[result.getResultNumber()]);
        else
          finish(makeUnknown(current));
        continue;
      }

      if (auto toClockOp = current.getDefiningOp<seq::ToClockOp>()) {
        frame.kind = EvalKind::Forward;
        frame.aux = toClockOp.getInput();
        frame.stage = 1;
        pushValue(frame.aux);
        continue;
      }

      if (auto fromClockOp = current.getDefiningOp<seq::FromClockOp>()) {
        frame.kind = EvalKind::Forward;
        frame.aux = fromClockOp.getInput();
        frame.stage = 1;
        pushValue(frame.aux);
        continue;
      }

      if (auto probeOp = current.getDefiningOp<llhd::ProbeOp>()) {
        SignalId sigId = resolveSignalId(probeOp.getSignal());
        if (sigId != 0) {
          const SignalValue &sv = scheduler.getSignalValue(sigId);
          finish(InterpretedValue::fromSignalValue(sv));
        } else {
          finish(makeUnknown(current));
        }
        continue;
      }

      if (auto constOp = current.getDefiningOp<hw::ConstantOp>()) {
        finish(InterpretedValue(constOp.getValue()));
        continue;
      }

      if (auto aggConstOp = current.getDefiningOp<hw::AggregateConstantOp>()) {
        llvm::APInt flatValue = flattenAggregateConstant(aggConstOp);
        finish(InterpretedValue(flatValue));
        continue;
      }

      if (current.getDefiningOp<hw::StructExtractOp>()) {
        frame.kind = EvalKind::StructExtract;
        frame.stage = 1;
        pushValue(current.getDefiningOp<hw::StructExtractOp>().getInput());
        continue;
      }

      if (current.getDefiningOp<hw::ArrayGetOp>()) {
        frame.kind = EvalKind::ArrayGet;
        frame.stage = 1;
        auto arrayGetOp = current.getDefiningOp<hw::ArrayGetOp>();
        pushValue(arrayGetOp.getInput());
        pushValue(arrayGetOp.getIndex());
        continue;
      }

      if (current.getDefiningOp<hw::ArrayCreateOp>()) {
        frame.kind = EvalKind::ArrayCreate;
        frame.stage = 1;
        auto createOp = current.getDefiningOp<hw::ArrayCreateOp>();
        for (Value input : createOp.getInputs())
          pushValue(input);
        continue;
      }

      if (current.getDefiningOp<hw::ArraySliceOp>()) {
        frame.kind = EvalKind::ArraySlice;
        frame.stage = 1;
        auto sliceOp = current.getDefiningOp<hw::ArraySliceOp>();
        pushValue(sliceOp.getInput());
        pushValue(sliceOp.getLowIndex());
        continue;
      }

      if (current.getDefiningOp<hw::ArrayConcatOp>()) {
        frame.kind = EvalKind::ArrayConcat;
        frame.stage = 1;
        auto concatOp = current.getDefiningOp<hw::ArrayConcatOp>();
        for (Value input : concatOp.getInputs())
          pushValue(input);
        continue;
      }

      if (current.getDefiningOp<hw::StructCreateOp>()) {
        frame.kind = EvalKind::StructCreate;
        frame.stage = 1;
        auto createOp = current.getDefiningOp<hw::StructCreateOp>();
        for (Value input : createOp.getInput())
          pushValue(input);
        continue;
      }

      if (current.getDefiningOp<hw::StructInjectOp>()) {
        frame.kind = EvalKind::StructInject;
        frame.stage = 1;
        auto injectOp = current.getDefiningOp<hw::StructInjectOp>();
        pushValue(injectOp.getInput());
        pushValue(injectOp.getNewValue());
        continue;
      }

      if (auto *defOp = current.getDefiningOp()) {
        if (defOp->getName().getStringRef() == "hw.struct_inject") {
          frame.kind = EvalKind::StructInjectLegacy;
          frame.stage = 1;
          pushValue(defOp->getOperand(0));
          pushValue(defOp->getOperand(1));
          continue;
        }
      }

      if (current.getDefiningOp<hw::BitcastOp>()) {
        frame.kind = EvalKind::Bitcast;
        frame.stage = 1;
        pushValue(current.getDefiningOp<hw::BitcastOp>().getInput());
        continue;
      }

      if (current.getDefiningOp<comb::XorOp>()) {
        frame.kind = EvalKind::Xor;
        frame.stage = 1;
        auto xorOp = current.getDefiningOp<comb::XorOp>();
        for (Value operand : xorOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::AndOp>()) {
        frame.kind = EvalKind::And;
        frame.stage = 1;
        auto andOp = current.getDefiningOp<comb::AndOp>();
        for (Value operand : andOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::OrOp>()) {
        frame.kind = EvalKind::Or;
        frame.stage = 1;
        auto orOp = current.getDefiningOp<comb::OrOp>();
        for (Value operand : orOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::ICmpOp>()) {
        frame.kind = EvalKind::ICmp;
        frame.stage = 1;
        auto icmpOp = current.getDefiningOp<comb::ICmpOp>();
        pushValue(icmpOp.getLhs());
        pushValue(icmpOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::MuxOp>()) {
        frame.kind = EvalKind::Mux;
        frame.stage = 1;
        auto muxOp = current.getDefiningOp<comb::MuxOp>();
        frame.aux = muxOp.getCond();
        pushValue(frame.aux);
        continue;
      }

      if (current.getDefiningOp<comb::ConcatOp>()) {
        frame.kind = EvalKind::Concat;
        frame.stage = 1;
        auto concatOp = current.getDefiningOp<comb::ConcatOp>();
        for (Value operand : concatOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::ExtractOp>()) {
        frame.kind = EvalKind::Extract;
        frame.stage = 1;
        pushValue(current.getDefiningOp<comb::ExtractOp>().getInput());
        continue;
      }

      if (current.getDefiningOp<comb::AddOp>()) {
        frame.kind = EvalKind::Add;
        frame.stage = 1;
        auto addOp = current.getDefiningOp<comb::AddOp>();
        for (Value operand : addOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::SubOp>()) {
        frame.kind = EvalKind::Sub;
        frame.stage = 1;
        auto subOp = current.getDefiningOp<comb::SubOp>();
        pushValue(subOp.getOperand(0));
        pushValue(subOp.getOperand(1));
        continue;
      }

      if (current.getDefiningOp<comb::ReplicateOp>()) {
        frame.kind = EvalKind::Replicate;
        frame.stage = 1;
        auto replOp = current.getDefiningOp<comb::ReplicateOp>();
        pushValue(replOp.getInput());
        continue;
      }

      if (current.getDefiningOp<comb::ParityOp>()) {
        frame.kind = EvalKind::Parity;
        frame.stage = 1;
        auto parityOp = current.getDefiningOp<comb::ParityOp>();
        pushValue(parityOp.getInput());
        continue;
      }

      if (current.getDefiningOp<comb::ShlOp>()) {
        frame.kind = EvalKind::Shl;
        frame.stage = 1;
        auto shlOp = current.getDefiningOp<comb::ShlOp>();
        pushValue(shlOp.getLhs());
        pushValue(shlOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::ShrUOp>()) {
        frame.kind = EvalKind::ShrU;
        frame.stage = 1;
        auto shrOp = current.getDefiningOp<comb::ShrUOp>();
        pushValue(shrOp.getLhs());
        pushValue(shrOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::ShrSOp>()) {
        frame.kind = EvalKind::ShrS;
        frame.stage = 1;
        auto shrOp = current.getDefiningOp<comb::ShrSOp>();
        pushValue(shrOp.getLhs());
        pushValue(shrOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::MulOp>()) {
        frame.kind = EvalKind::Mul;
        frame.stage = 1;
        auto mulOp = current.getDefiningOp<comb::MulOp>();
        for (Value operand : mulOp.getOperands())
          pushValue(operand);
        continue;
      }

      if (current.getDefiningOp<comb::DivSOp>()) {
        frame.kind = EvalKind::DivS;
        frame.stage = 1;
        auto divOp = current.getDefiningOp<comb::DivSOp>();
        pushValue(divOp.getLhs());
        pushValue(divOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::DivUOp>()) {
        frame.kind = EvalKind::DivU;
        frame.stage = 1;
        auto divOp = current.getDefiningOp<comb::DivUOp>();
        pushValue(divOp.getLhs());
        pushValue(divOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::ModSOp>()) {
        frame.kind = EvalKind::ModS;
        frame.stage = 1;
        auto modOp = current.getDefiningOp<comb::ModSOp>();
        pushValue(modOp.getLhs());
        pushValue(modOp.getRhs());
        continue;
      }

      if (current.getDefiningOp<comb::ModUOp>()) {
        frame.kind = EvalKind::ModU;
        frame.stage = 1;
        auto modOp = current.getDefiningOp<comb::ModUOp>();
        pushValue(modOp.getLhs());
        pushValue(modOp.getRhs());
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: Cannot evaluate continuous value for "
                 << *current.getDefiningOp() << "\n");
      finish(makeUnknown(current));
      continue;
    }

    switch (frame.kind) {
    case EvalKind::Forward: {
      InterpretedValue forwarded = getCached(frame.aux);
      finish(forwarded);
      break;
    }
    case EvalKind::StructExtract: {
      auto extractOp = current.getDefiningOp<hw::StructExtractOp>();
      InterpretedValue inputVal = getCached(extractOp.getInput());
      if (inputVal.isX()) {
        finish(makeUnknown(current));
        break;
      }
      auto structType =
          hw::type_cast<hw::StructType>(extractOp.getInput().getType());
      StringRef fieldName = extractOp.getFieldName();
      unsigned bitOffset = 0;
      unsigned fieldWidth = 0;
      auto elements = structType.getElements();
      unsigned totalWidth = getTypeWidth(structType);
      for (auto &element : elements) {
        unsigned elemWidth = getTypeWidth(element.type);
        if (element.name == fieldName) {
          fieldWidth = elemWidth;
          break;
        }
        bitOffset += elemWidth;
      }
      unsigned shiftAmount = totalWidth - bitOffset - fieldWidth;
      llvm::APInt fieldVal =
          inputVal.getAPInt().extractBits(fieldWidth, shiftAmount);
      finish(InterpretedValue(fieldVal));
      break;
    }
    case EvalKind::ArrayGet: {
      auto arrayGetOp = current.getDefiningOp<hw::ArrayGetOp>();
      InterpretedValue arrayVal = getCached(arrayGetOp.getInput());
      InterpretedValue indexVal = getCached(arrayGetOp.getIndex());
      if (arrayVal.isX() || indexVal.isX()) {
        finish(makeUnknown(current));
        break;
      }
      auto arrayType =
          hw::type_cast<hw::ArrayType>(arrayGetOp.getInput().getType());
      unsigned elementWidth = getTypeWidth(arrayType.getElementType());
      unsigned numElements = arrayType.getNumElements();
      uint64_t index = indexVal.getAPInt().getZExtValue();
      if (index >= numElements) {
        finish(InterpretedValue::makeX(elementWidth));
        break;
      }
      // Array element 0 is at LSB (offset 0), matching CIRCT hw dialect convention.
      unsigned bitOffset = index * elementWidth;
      APInt result = arrayVal.getAPInt().extractBits(elementWidth, bitOffset);
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::ArrayCreate: {
      auto createOp = current.getDefiningOp<hw::ArrayCreateOp>();
      auto arrayType = hw::type_cast<hw::ArrayType>(createOp.getType());
      unsigned elementWidth = getTypeWidth(arrayType.getElementType());
      unsigned numElements = arrayType.getNumElements();
      unsigned totalWidth = elementWidth * numElements;

      APInt result(totalWidth, 0);
      bool hasX = false;

      for (size_t i = 0; i < createOp.getInputs().size(); ++i) {
        InterpretedValue elem = getCached(createOp.getInputs()[i]);
        APInt elemVal(elementWidth, 0);
        if (elem.isX()) {
          if (auto encoded =
                  getEncodedUnknownForType(arrayType.getElementType())) {
            elemVal = encoded->zextOrTrunc(elementWidth);
          } else {
            hasX = true;
            break;
          }
        } else {
          elemVal = elem.getAPInt();
        }
        if (elemVal.getBitWidth() < elementWidth)
          elemVal = elemVal.zext(elementWidth);
        else if (elemVal.getBitWidth() > elementWidth)
          elemVal = elemVal.trunc(elementWidth);
        unsigned offset = (numElements - 1 - i) * elementWidth;
        result.insertBits(elemVal, offset);
      }

      if (hasX) {
        finish(InterpretedValue::makeX(totalWidth));
      } else {
        finish(InterpretedValue(result));
      }
      break;
    }
    case EvalKind::ArraySlice: {
      auto sliceOp = current.getDefiningOp<hw::ArraySliceOp>();
      InterpretedValue arrayVal = getCached(sliceOp.getInput());
      InterpretedValue indexVal = getCached(sliceOp.getLowIndex());
      auto resultType = hw::type_cast<hw::ArrayType>(sliceOp.getType());
      if (arrayVal.isX() || indexVal.isX()) {
        unsigned resultWidth = getTypeWidth(resultType.getElementType()) *
                               resultType.getNumElements();
        finish(InterpretedValue::makeX(resultWidth));
        break;
      }

      auto inputType =
          hw::type_cast<hw::ArrayType>(sliceOp.getInput().getType());
      unsigned elementWidth = getTypeWidth(inputType.getElementType());
      unsigned inputElements = inputType.getNumElements();
      unsigned resultElements = resultType.getNumElements();
      uint64_t lowIdx = indexVal.getAPInt().getZExtValue();

      if (lowIdx + resultElements > inputElements) {
        unsigned resultWidth = elementWidth * resultElements;
        finish(InterpretedValue::makeX(resultWidth));
        break;
      }

      // Array element 0 is at LSB (offset 0), matching CIRCT hw dialect convention.
      unsigned offset = lowIdx * elementWidth;
      unsigned sliceWidth = resultElements * elementWidth;
      APInt slice = arrayVal.getAPInt().extractBits(sliceWidth, offset);
      finish(InterpretedValue(slice));
      break;
    }
    case EvalKind::ArrayConcat: {
      auto concatOp = current.getDefiningOp<hw::ArrayConcatOp>();
      auto resultType = hw::type_cast<hw::ArrayType>(concatOp.getType());
      unsigned resultWidth = getTypeWidth(resultType.getElementType()) *
                             resultType.getNumElements();

      APInt result(resultWidth, 0);
      bool hasX = false;
      unsigned bitOffset = resultWidth;

      for (Value input : concatOp.getInputs()) {
        InterpretedValue val = getCached(input);
        if (val.isX()) {
          hasX = true;
          break;
        }
        unsigned inputWidth = val.getWidth();
        bitOffset -= inputWidth;
        result.insertBits(val.getAPInt(), bitOffset);
      }

      if (hasX) {
        finish(InterpretedValue::makeX(resultWidth));
      } else {
        finish(InterpretedValue(result));
      }
      break;
    }
    case EvalKind::StructCreate: {
      auto createOp = current.getDefiningOp<hw::StructCreateOp>();
      auto structType = hw::type_cast<hw::StructType>(createOp.getType());
      unsigned totalWidth = getTypeWidth(structType);
      llvm::APInt result(totalWidth, 0);
      auto elements = structType.getElements();
      unsigned bitOffset = totalWidth;
      for (size_t i = 0; i < createOp.getInput().size(); ++i) {
        InterpretedValue fieldVal = getCached(createOp.getInput()[i]);
        unsigned fieldWidth = getTypeWidth(elements[i].type);
        bitOffset -= fieldWidth;
        if (!fieldVal.isX()) {
          APInt fieldBits = fieldVal.getAPInt();
          if (fieldBits.getBitWidth() < fieldWidth)
            fieldBits = fieldBits.zext(fieldWidth);
          else if (fieldBits.getBitWidth() > fieldWidth)
            fieldBits = fieldBits.trunc(fieldWidth);
          result.insertBits(fieldBits, bitOffset);
        }
      }
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::StructInject: {
      auto injectOp = current.getDefiningOp<hw::StructInjectOp>();
      InterpretedValue structVal = getCached(injectOp.getInput());
      InterpretedValue newVal = getCached(injectOp.getNewValue());
      unsigned totalWidth = getTypeWidth(injectOp.getType());
      if (structVal.isX() || newVal.isX()) {
        finish(InterpretedValue::makeX(totalWidth));
        break;
      }
      auto structType = cast<hw::StructType>(injectOp.getInput().getType());
      auto fieldIndexOpt = structType.getFieldIndex(injectOp.getFieldName());
      if (!fieldIndexOpt) {
        finish(InterpretedValue::makeX(totalWidth));
        break;
      }
      unsigned fieldIndex = *fieldIndexOpt;
      auto elements = structType.getElements();
      unsigned fieldOffset = 0;
      for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
        fieldOffset += getTypeWidth(elements[i].type);
      unsigned fieldWidth = getTypeWidth(elements[fieldIndex].type);
      llvm::APInt result = structVal.getAPInt();
      llvm::APInt fieldValue = newVal.getAPInt();
      if (fieldValue.getBitWidth() < fieldWidth)
        fieldValue = fieldValue.zext(fieldWidth);
      else if (fieldValue.getBitWidth() > fieldWidth)
        fieldValue = fieldValue.trunc(fieldWidth);
      result.insertBits(fieldValue, fieldOffset);
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::StructInjectLegacy: {
      auto *defOp = current.getDefiningOp();
      Value input = defOp->getOperand(0);
      Value newValue = defOp->getOperand(1);
      auto structType = cast<hw::StructType>(input.getType());
      unsigned totalWidth = getTypeWidth(structType);
      auto fieldIndexAttr = defOp->getAttrOfType<IntegerAttr>("fieldIndex");
      if (!fieldIndexAttr) {
        finish(InterpretedValue::makeX(totalWidth));
        break;
      }
      unsigned fieldIndex = fieldIndexAttr.getValue().getZExtValue();
      auto elements = structType.getElements();
      InterpretedValue structVal = getCached(input);
      InterpretedValue newVal = getCached(newValue);
      if (structVal.isX() || newVal.isX()) {
        finish(InterpretedValue::makeX(totalWidth));
        break;
      }
      unsigned fieldOffset = 0;
      for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
        fieldOffset += getTypeWidth(elements[i].type);
      unsigned fieldWidth = getTypeWidth(elements[fieldIndex].type);
      llvm::APInt result = structVal.getAPInt();
      llvm::APInt fieldValue = newVal.getAPInt();
      if (fieldValue.getBitWidth() < fieldWidth)
        fieldValue = fieldValue.zext(fieldWidth);
      else if (fieldValue.getBitWidth() > fieldWidth)
        fieldValue = fieldValue.trunc(fieldWidth);
      result.insertBits(fieldValue, fieldOffset);
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Bitcast: {
      auto bitcastOp = current.getDefiningOp<hw::BitcastOp>();
      InterpretedValue inputVal = getCached(bitcastOp.getInput());
      unsigned outputWidth = getTypeWidth(bitcastOp.getType());
      if (inputVal.isX()) {
        finish(InterpretedValue::makeX(outputWidth));
        break;
      }
      llvm::APInt result = inputVal.getAPInt();
      if (result.getBitWidth() < outputWidth)
        result = result.zext(outputWidth);
      else if (result.getBitWidth() > outputWidth)
        result = result.trunc(outputWidth);
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Xor: {
      auto xorOp = current.getDefiningOp<comb::XorOp>();
      unsigned width = getTypeWidth(current.getType());
      llvm::APInt result = APInt::getZero(width);
      bool sawX = false;
      for (Value operand : xorOp.getOperands()) {
        InterpretedValue opVal = getCached(operand);
        if (opVal.isX()) { sawX = true; break; }
        llvm::APInt v = opVal.getAPInt();
        if (v.getBitWidth() != width)
          v = v.zextOrTrunc(width);
        result ^= v;
      }
      if (sawX)
        finish(makeUnknown(current));
      else
        finish(InterpretedValue(result));
      break;
    }
    case EvalKind::And: {
      auto andOp = current.getDefiningOp<comb::AndOp>();
      unsigned width = getTypeWidth(current.getType());
      llvm::APInt result(width, 0);
      result.setAllBits();
      bool sawX = false;
      for (Value operand : andOp.getOperands()) {
        InterpretedValue opVal = getCached(operand);
        if (opVal.isX()) {
          sawX = true;
          break;
        }
        llvm::APInt opBits = opVal.getAPInt();
        if (opBits.getBitWidth() < width)
          opBits = opBits.zext(width);
        else if (opBits.getBitWidth() > width)
          opBits = opBits.trunc(width);
        result &= opBits;
      }
      if (sawX)
        finish(InterpretedValue::makeX(width));
      else
        finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Or: {
      auto orOp = current.getDefiningOp<comb::OrOp>();
      unsigned width = getTypeWidth(current.getType());
      llvm::APInt result(width, 0);
      bool sawX = false;
      for (Value operand : orOp.getOperands()) {
        InterpretedValue opVal = getCached(operand);
        if (opVal.isX()) {
          sawX = true;
          break;
        }
        llvm::APInt opBits = opVal.getAPInt();
        if (opBits.getBitWidth() < width)
          opBits = opBits.zext(width);
        else if (opBits.getBitWidth() > width)
          opBits = opBits.trunc(width);
        result |= opBits;
      }
      if (sawX)
        finish(InterpretedValue::makeX(width));
      else
        finish(InterpretedValue(result));
      break;
    }
    case EvalKind::ICmp: {
      auto icmpOp = current.getDefiningOp<comb::ICmpOp>();
      InterpretedValue lhs = getCached(icmpOp.getLhs());
      InterpretedValue rhs = getCached(icmpOp.getRhs());
      if (lhs.isX() || rhs.isX()) {
        finish(InterpretedValue::makeX(1));
        break;
      }
      bool result = false;
      llvm::APInt lVal = lhs.getAPInt();
      llvm::APInt rVal = rhs.getAPInt();
      unsigned compareWidth = std::max(lVal.getBitWidth(), rVal.getBitWidth());
      normalizeWidths(lVal, rVal, compareWidth);
      switch (icmpOp.getPredicate()) {
      case comb::ICmpPredicate::eq:
        result = (lVal == rVal);
        break;
      case comb::ICmpPredicate::ne:
        result = (lVal != rVal);
        break;
      case comb::ICmpPredicate::ult:
        result = lVal.ult(rVal);
        break;
      case comb::ICmpPredicate::ule:
        result = lVal.ule(rVal);
        break;
      case comb::ICmpPredicate::ugt:
        result = lVal.ugt(rVal);
        break;
      case comb::ICmpPredicate::uge:
        result = lVal.uge(rVal);
        break;
      default:
        result = false;
        break;
      }
      finish(InterpretedValue(result ? 1ULL : 0ULL, 1));
      break;
    }
    case EvalKind::Mux: {
      auto muxOp = current.getDefiningOp<comb::MuxOp>();
      if (frame.stage == 1) {
        InterpretedValue cond = getCached(frame.aux);
        if (cond.isX()) {
          finish(makeUnknown(current));
          break;
        }
        Value selected = cond.getUInt64() != 0 ? muxOp.getTrueValue()
                                               : muxOp.getFalseValue();
        frame.aux = selected;
        frame.stage = 2;
        pushValue(selected);
        continue;
      }
      InterpretedValue selectedVal = getCached(frame.aux);
      finish(selectedVal);
      break;
    }
    case EvalKind::Concat: {
      auto concatOp = current.getDefiningOp<comb::ConcatOp>();
      unsigned totalWidth = getTypeWidth(current.getType());
      llvm::APInt result(totalWidth, 0);
      unsigned bitOffset = totalWidth;
      for (Value operand : concatOp.getOperands()) {
        InterpretedValue opVal = getCached(operand);
        unsigned width = getTypeWidth(operand.getType());
        bitOffset -= width;
        if (opVal.isX())
          continue;
        llvm::APInt bits = opVal.getAPInt();
        if (bits.getBitWidth() < width)
          bits = bits.zext(width);
        else if (bits.getBitWidth() > width)
          bits = bits.trunc(width);
        result.insertBits(bits, bitOffset);
      }
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Extract: {
      auto extractOp = current.getDefiningOp<comb::ExtractOp>();
      InterpretedValue inputVal = getCached(extractOp.getInput());
      unsigned width = getTypeWidth(current.getType());
      if (inputVal.isX()) {
        finish(InterpretedValue::makeX(width));
        break;
      }
      unsigned lowBit = extractOp.getLowBit();
      llvm::APInt input = inputVal.getAPInt();
      llvm::APInt sliced = input.extractBits(width, lowBit);
      finish(InterpretedValue(sliced));
      break;
    }
    case EvalKind::Add: {
      auto addOp = current.getDefiningOp<comb::AddOp>();
      unsigned width = getTypeWidth(current.getType());
      llvm::APInt result = APInt::getZero(width);
      bool sawX = false;
      for (Value operand : addOp.getOperands()) {
        InterpretedValue opVal = getCached(operand);
        if (opVal.isX()) { sawX = true; break; }
        llvm::APInt v = opVal.getAPInt();
        if (v.getBitWidth() != width)
          v = v.zextOrTrunc(width);
        result += v;
      }
      if (sawX)
        finish(makeUnknown(current));
      else
        finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Sub: {
      auto subOp = current.getDefiningOp<comb::SubOp>();
      InterpretedValue lhs = getCached(subOp.getOperand(0));
      InterpretedValue rhs = getCached(subOp.getOperand(1));
      if (lhs.isX() || rhs.isX()) {
        finish(makeUnknown(current));
        break;
      }
      llvm::APInt lhsVal = lhs.getAPInt();
      llvm::APInt rhsVal = rhs.getAPInt();
      unsigned width = getTypeWidth(current.getType());
      normalizeWidths(lhsVal, rhsVal, width);
      finish(InterpretedValue(lhsVal - rhsVal));
      break;
    }
    case EvalKind::Replicate: {
      auto replOp = current.getDefiningOp<comb::ReplicateOp>();
      InterpretedValue input = getCached(replOp.getInput());
      if (input.isX()) {
        finish(InterpretedValue::makeX(getTypeWidth(current.getType())));
        break;
      }
      unsigned inputWidth = input.getWidth();
      unsigned multiple = replOp.getMultiple();
      llvm::APInt result(getTypeWidth(current.getType()), 0);
      for (unsigned i = 0; i < multiple; ++i) {
        llvm::APInt chunk = input.getAPInt().zext(result.getBitWidth());
        result = result.shl(inputWidth) | chunk;
      }
      finish(InterpretedValue(result));
      break;
    }
    case EvalKind::Parity: {
      auto parityOp = current.getDefiningOp<comb::ParityOp>();
      InterpretedValue input = getCached(parityOp.getInput());
      if (input.isX()) {
        finish(InterpretedValue::makeX(1));
        break;
      }
      bool parity = (input.getAPInt().popcount() & 1) != 0;
      finish(InterpretedValue(parity, 1));
      break;
    }
    case EvalKind::Shl: {
      auto shlOp = current.getDefiningOp<comb::ShlOp>();
      InterpretedValue lhs = getCached(shlOp.getLhs());
      InterpretedValue rhs = getCached(shlOp.getRhs());
      if (lhs.isX() || rhs.isX()) {
        finish(InterpretedValue::makeX(getTypeWidth(current.getType())));
        break;
      }
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      finish(InterpretedValue(lhs.getAPInt().shl(shift)));
      break;
    }
    case EvalKind::ShrU: {
      auto shrOp = current.getDefiningOp<comb::ShrUOp>();
      InterpretedValue lhs = getCached(shrOp.getLhs());
      InterpretedValue rhs = getCached(shrOp.getRhs());
      if (lhs.isX() || rhs.isX()) {
        finish(InterpretedValue::makeX(getTypeWidth(current.getType())));
        break;
      }
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      finish(InterpretedValue(lhs.getAPInt().lshr(shift)));
      break;
    }
    case EvalKind::ShrS: {
      auto shrOp = current.getDefiningOp<comb::ShrSOp>();
      InterpretedValue lhs = getCached(shrOp.getLhs());
      InterpretedValue rhs = getCached(shrOp.getRhs());
      if (lhs.isX() || rhs.isX()) {
        finish(InterpretedValue::makeX(getTypeWidth(current.getType())));
        break;
      }
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      finish(InterpretedValue(lhs.getAPInt().ashr(shift)));
      break;
    }
    case EvalKind::Mul: {
      auto mulOp = current.getDefiningOp<comb::MulOp>();
      unsigned targetWidth = getTypeWidth(current.getType());
      llvm::APInt result(targetWidth, 1);
      bool sawX = false;
      for (Value operand : mulOp.getOperands()) {
        InterpretedValue value = getCached(operand);
        if (value.isX()) {
          sawX = true;
          break;
        }
        llvm::APInt operandVal = value.getAPInt();
        if (operandVal.getBitWidth() < targetWidth)
          operandVal = operandVal.zext(targetWidth);
        else if (operandVal.getBitWidth() > targetWidth)
          operandVal = operandVal.trunc(targetWidth);
        result *= operandVal;
      }
      if (sawX)
        finish(InterpretedValue::makeX(targetWidth));
      else
        finish(InterpretedValue(result));
      break;
    }
    case EvalKind::DivS: {
      auto divOp = current.getDefiningOp<comb::DivSOp>();
      InterpretedValue lhs = getCached(divOp.getLhs());
      InterpretedValue rhs = getCached(divOp.getRhs());
      unsigned targetWidth = getTypeWidth(current.getType());
      if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
        finish(InterpretedValue::makeX(targetWidth));
        break;
      }
      llvm::APInt lhsVal = lhs.getAPInt();
      llvm::APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      finish(InterpretedValue(lhsVal.sdiv(rhsVal)));
      break;
    }
    case EvalKind::DivU: {
      auto divOp = current.getDefiningOp<comb::DivUOp>();
      InterpretedValue lhs = getCached(divOp.getLhs());
      InterpretedValue rhs = getCached(divOp.getRhs());
      unsigned targetWidth = getTypeWidth(current.getType());
      if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
        finish(InterpretedValue::makeX(targetWidth));
        break;
      }
      llvm::APInt lhsVal = lhs.getAPInt();
      llvm::APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      finish(InterpretedValue(lhsVal.udiv(rhsVal)));
      break;
    }
    case EvalKind::ModS: {
      auto modOp = current.getDefiningOp<comb::ModSOp>();
      InterpretedValue lhs = getCached(modOp.getLhs());
      InterpretedValue rhs = getCached(modOp.getRhs());
      unsigned targetWidth = getTypeWidth(current.getType());
      if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
        finish(InterpretedValue::makeX(targetWidth));
        break;
      }
      llvm::APInt lhsVal = lhs.getAPInt();
      llvm::APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      finish(InterpretedValue(lhsVal.srem(rhsVal)));
      break;
    }
    case EvalKind::ModU: {
      auto modOp = current.getDefiningOp<comb::ModUOp>();
      InterpretedValue lhs = getCached(modOp.getLhs());
      InterpretedValue rhs = getCached(modOp.getRhs());
      unsigned targetWidth = getTypeWidth(current.getType());
      if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
        finish(InterpretedValue::makeX(targetWidth));
        break;
      }
      llvm::APInt lhsVal = lhs.getAPInt();
      llvm::APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      finish(InterpretedValue(lhsVal.urem(rhsVal)));
      break;
    }
    case EvalKind::None:
      finish(makeUnknown(current));
      break;
    }
  }

  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;
  return makeUnknown(value);
}

//===----------------------------------------------------------------------===//
// Process Execution
//===----------------------------------------------------------------------===//

static void cacheWaitState(ProcessExecutionState &state,
                           const ProcessScheduler &scheduler,
                           const SensitivityList *waitList, bool hadDelay) {
  state.lastWaitHadDelay = hadDelay;
  state.lastWaitHasEdge = false;
  state.lastSensitivityEntries.clear();
  state.lastSensitivityValues.clear();
  state.lastSensitivityValid = false;

  if (!waitList || waitList->empty())
    return;

  state.lastSensitivityEntries = waitList->getEntries();
  state.lastSensitivityValues.reserve(state.lastSensitivityEntries.size());
  for (const auto &entry : state.lastSensitivityEntries) {
    state.lastSensitivityValues.push_back(
        scheduler.getSignalValue(entry.signalId));
    if (entry.edge != EdgeType::AnyEdge)
      state.lastWaitHasEdge = true;
  }
  state.lastSensitivityValid = true;
}

static bool canSkipCachedProcess(const ProcessExecutionState &state,
                                 const ProcessScheduler &scheduler) {
  if (!state.cacheable || !state.lastSensitivityValid ||
      state.lastWaitHadDelay || state.lastWaitHasEdge)
    return false;
  if (state.lastSensitivityEntries.empty())
    return false;
  if (state.lastSensitivityEntries.size() != state.lastSensitivityValues.size())
    return false;

  for (size_t i = 0; i < state.lastSensitivityEntries.size(); ++i) {
    SignalId sigId = state.lastSensitivityEntries[i].signalId;
    if (scheduler.getSignalValue(sigId) != state.lastSensitivityValues[i])
      return false;
  }

  return true;
}

static SensitivityList
buildSensitivityListFromState(const ProcessExecutionState &state) {
  SensitivityList list;
  for (const auto &entry : state.lastSensitivityEntries)
    list.addEdge(entry.signalId, entry.edge);
  return list;
}

void LLHDProcessInterpreter::checkMemoryEventWaiters() {
  if (memoryEventWaiters.empty())
    return;

  // Check each memory event waiter to see if the watched value has changed
  llvm::SmallVector<ProcessId, 4> toWake;

  for (auto &[procId, waiter] : memoryEventWaiters) {
    // Find the memory block containing this address
    MemoryBlock *block = nullptr;
    uint64_t offset = 0;
    uint64_t addr = waiter.address;

    // Check module-level allocas first (by address)
    for (auto &[val, memBlock] : moduleLevelAllocas) {
      auto addrIt = moduleInitValueMap.find(val);
      if (addrIt != moduleInitValueMap.end()) {
        uint64_t blockAddr = addrIt->second.getUInt64();
        if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
          block = &memBlock;
          offset = addr - blockAddr;
          break;
        }
      }
    }

    // Check malloc blocks
    if (!block) {
      for (auto &entry : mallocBlocks) {
        uint64_t mallocBaseAddr = entry.first;
        uint64_t mallocSize = entry.second.size;
        if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
          block = &entry.second;
          offset = addr - mallocBaseAddr;
          break;
        }
      }
    }

    // Check global memory blocks
    if (!block) {
      for (auto &entry : globalAddresses) {
        StringRef globalName = entry.first();
        uint64_t globalBaseAddr = entry.second;
        auto blockIt = globalMemoryBlocks.find(globalName);
        if (blockIt != globalMemoryBlocks.end()) {
          uint64_t globalSize = blockIt->second.size;
          if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
            block = &blockIt->second;
            offset = addr - globalBaseAddr;
            break;
          }
        }
      }
    }

    // Also check process-local memory blocks
    if (!block) {
      auto procStateIt = processStates.find(procId);
      if (procStateIt != processStates.end()) {
        auto &procState = procStateIt->second;
        for (auto &[val, memBlock] : procState.memoryBlocks) {
          auto addrIt = procState.valueMap.find(val);
          if (addrIt != procState.valueMap.end()) {
            uint64_t blockAddr = addrIt->second.getUInt64();
            if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
              block = &memBlock;
              offset = addr - blockAddr;
              break;
            }
          }
        }
      }
    }

    if (!block || !block->initialized) {
      LLVM_DEBUG(llvm::dbgs() << "  Memory event waiter: block not found for "
                                 "address 0x"
                              << llvm::format_hex(addr, 16) << "\n");
      continue;
    }

    // Read current value
    if (offset + waiter.valueSize > block->size)
      continue;

    uint64_t currentValue = 0;
    for (unsigned i = 0; i < waiter.valueSize; ++i) {
      currentValue |=
          static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
    }

    // Check if value changed and matches the expected edge type.
    // For UVM events (!moore.event), we need to detect a "trigger" which is
    // a rising edge (0→1 transition). This is critical for wait_for_objection:
    // if no objection has been raised yet (value=0), we must wait for the
    // NEXT trigger (0→1), not just any change.
    bool shouldWake = false;
    if (currentValue != waiter.lastValue) {
      if (waiter.waitForRisingEdge) {
        // Only wake on rising edge (0→1)
        shouldWake = (waiter.lastValue == 0 && currentValue != 0);
        LLVM_DEBUG(llvm::dbgs()
                   << "  Memory event check for process " << procId
                   << ": address 0x" << llvm::format_hex(addr, 16) << " changed "
                   << waiter.lastValue << " -> " << currentValue
                   << (shouldWake ? " (rising edge - WAKE)" : " (not rising edge - continue waiting)") << "\n");
        // Update lastValue so we can detect the next rising edge
        waiter.lastValue = currentValue;
      } else {
        // Wake on any change
        shouldWake = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "  Memory event triggered for process " << procId
                   << ": address 0x" << llvm::format_hex(addr, 16) << " changed "
                   << waiter.lastValue << " -> " << currentValue << "\n");
      }
    }
    if (shouldWake) {
      toWake.push_back(procId);
    }
  }

  // Wake the processes whose watched memory changed
  for (ProcessId procId : toWake) {
    memoryEventWaiters.erase(procId);
    auto it = processStates.find(procId);
    if (it != processStates.end()) {
      it->second.waiting = false;
      // Schedule the process to run now that the memory event triggered
      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    }
  }
}

void LLHDProcessInterpreter::executeProcess(ProcessId procId) {
  auto it = processStates.find(procId);
  if (it == processStates.end()) {
    LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Unknown process ID "
                            << procId << "\n");
    return;
  }

  ProcessExecutionState &state = it->second;

  // Check if this process is waiting on a memory event.
  // If so, check if the memory value has changed before proceeding.
  auto memWaiterIt = memoryEventWaiters.find(procId);
  if (memWaiterIt != memoryEventWaiters.end()) {
    MemoryEventWaiter &waiter = memWaiterIt->second;
    uint64_t addr = waiter.address;

    // Find the memory block containing this address
    MemoryBlock *block = nullptr;
    uint64_t offset = 0;

    // Check module-level allocas first (by address)
    for (auto &[val, memBlock] : moduleLevelAllocas) {
      auto addrIt = moduleInitValueMap.find(val);
      if (addrIt != moduleInitValueMap.end()) {
        uint64_t blockAddr = addrIt->second.getUInt64();
        if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
          block = &memBlock;
          offset = addr - blockAddr;
          break;
        }
      }
    }

    // Check malloc blocks
    if (!block) {
      for (auto &entry : mallocBlocks) {
        uint64_t mallocBaseAddr = entry.first;
        uint64_t mallocSize = entry.second.size;
        if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
          block = &entry.second;
          offset = addr - mallocBaseAddr;
          break;
        }
      }
    }

    // Check global memory blocks
    if (!block) {
      for (auto &entry : globalAddresses) {
        StringRef globalName = entry.first();
        uint64_t globalBaseAddr = entry.second;
        auto blockIt = globalMemoryBlocks.find(globalName);
        if (blockIt != globalMemoryBlocks.end()) {
          uint64_t globalSize = blockIt->second.size;
          if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
            block = &blockIt->second;
            offset = addr - globalBaseAddr;
            break;
          }
        }
      }
    }

    // Also check process-local memory blocks
    if (!block) {
      auto &procState = processStates[procId];
      for (auto &[val, memBlock] : procState.memoryBlocks) {
        auto addrIt = procState.valueMap.find(val);
        if (addrIt != procState.valueMap.end()) {
          uint64_t blockAddr = addrIt->second.getUInt64();
          if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
            block = &memBlock;
            offset = addr - blockAddr;
            break;
          }
        }
      }
    }

    if (block && block->initialized && offset + waiter.valueSize <= block->size) {
      // Read current value
      uint64_t currentValue = 0;
      for (unsigned i = 0; i < waiter.valueSize; ++i) {
        currentValue |=
            static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
      }

      if (currentValue == waiter.lastValue) {
        // Value hasn't changed - keep waiting
        // Don't re-schedule - process will be checked when memory is written
        LLVM_DEBUG(llvm::dbgs()
                   << "  Memory event: value unchanged at 0x"
                   << llvm::format_hex(addr, 16) << " (value=" << currentValue
                   << "), process " << procId << " remains waiting\n");
        return;
      }

      // Value changed - wake up the process
      LLVM_DEBUG(llvm::dbgs()
                 << "  Memory event triggered for process " << procId
                 << ": address 0x" << llvm::format_hex(addr, 16) << " changed "
                 << waiter.lastValue << " -> " << currentValue << "\n");
      memoryEventWaiters.erase(memWaiterIt);
      state.waiting = false;
    } else {
      // Memory block not accessible - remove waiter and continue
      LLVM_DEBUG(llvm::dbgs()
                 << "  Memory event: block not accessible, removing waiter\n");
      memoryEventWaiters.erase(memWaiterIt);
      state.waiting = false;
    }
  }

  ScopedInstanceContext instScope(*this, state.instanceId);
  ScopedInputValueMap inputScope(*this, state.inputMap);

  if ((state.waiting || state.destBlock) &&
      canSkipCachedProcess(state, scheduler)) {
    SensitivityList waitList = buildSensitivityListFromState(state);
    if (!waitList.empty()) {
      ++state.cacheSkips;
      state.waiting = true;
      scheduler.suspendProcessForEvents(procId, waitList);
      return;
    }
  }

  // If resuming from a wait, set up the destination block
  // Note: waiting flag may already be cleared by resumeProcess, so check destBlock
  if (state.destBlock) {
    LLVM_DEBUG(llvm::dbgs() << "  Resuming to destination block\n");
    state.currentBlock = state.destBlock;

    // If resumeAtCurrentOp is set, keep currentOp as is (used for deferred
    // llhd.halt and sim.terminate). Otherwise, start from block beginning.
    if (!state.resumeAtCurrentOp) {
      state.currentOp = state.currentBlock->begin();
    }

    // Transfer destination operands to block arguments
    for (auto [arg, val] :
         llvm::zip(state.currentBlock->getArguments(), state.destOperands)) {
      state.valueMap[arg] = val;
    }

    state.waiting = false;
    state.destBlock = nullptr;
    state.destOperands.clear();
    state.resumeAtCurrentOp = false;
  } else if (state.waiting) {
    // Handle the case where a process was triggered by an event (via
    // triggerSensitiveProcesses) rather than by the delay callback
    // (resumeProcess). In this case, state.waiting may still be true but
    // destBlock is null because the scheduler directly scheduled the process.
    // This can happen when a process is triggered by a signal change while
    // it was waiting for that signal.
    //
    // IMPORTANT: If waitConditionRestartBlock is set, the process is waiting
    // on a wait(condition) that evaluated to false. In this case, we should
    // NOT resume the process - it should only resume when its poll callback
    // fires (which sets waiting=false before scheduling). This prevents an
    // infinite loop where the process is continuously re-scheduled, evaluates
    // the still-false condition, and sets waiting=true again, all at the same
    // simulation time.
    if (state.waitConditionRestartBlock) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Process triggered while waiting on wait(condition) - "
                    "ignoring spurious trigger, will resume via poll callback\n");
      return;
    }
    //
    // We need to clear the waiting flag so the execution loop will run.
    // The process will resume at its current position (which should be right
    // after the llhd.wait that set the waiting flag originally, or at the
    // wait's destination block if the process was waiting).
    LLVM_DEBUG(llvm::dbgs()
               << "  Warning: Process triggered while waiting but destBlock is "
                  "null. Clearing waiting flag and resuming.\n");
    state.waiting = false;
  }

  // Handle wait_condition restart: if we're resuming after a wait(condition)
  // that was false, we need to restart from the point where the condition
  // computation begins and invalidate only the relevant cached values.
  if (state.waitConditionRestartBlock) {
    LLVM_DEBUG(llvm::dbgs() << "  Restarting for wait_condition re-evaluation "
                            << "(invalidating " << state.waitConditionValuesToInvalidate.size()
                            << " cached values)\n");

    // Set the current block and operation to the restart point
    state.currentBlock = state.waitConditionRestartBlock;
    state.currentOp = state.waitConditionRestartOp;

    // Clear only the cached values that are part of the condition computation.
    // This avoids re-executing side effects like fork creation.
    for (Value v : state.waitConditionValuesToInvalidate) {
      state.valueMap.erase(v);
    }

    // Note: we don't clear waitConditionRestartBlock here - it will be
    // cleared by __moore_wait_condition when the condition becomes true.
  }

  // Handle call stack frames: if we have saved call frames from a wait inside
  // a function, we need to resume execution inside those functions instead of
  // continuing at the process level.
  if (!state.callStack.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Process has " << state.callStack.size()
               << " saved call stack frame(s), resuming from innermost\n");

    // Process call stack frames from innermost to outermost
    // Each frame represents a function that was interrupted by a wait
    while (!state.callStack.empty()) {
      CallStackFrame frame = std::move(state.callStack.back());
      state.callStack.pop_back();

      LLVM_DEBUG(llvm::dbgs() << "    Resuming function '"
                              << frame.funcOp.getName() << "'\n");

      // Resume the function from its saved position
      llvm::SmallVector<InterpretedValue, 4> results;
      ++state.callDepth;
      LogicalResult funcResult =
          interpretFuncBody(procId, frame.funcOp, frame.args, results,
                            frame.callOp, frame.resumeBlock, frame.resumeOp);
      --state.callDepth;

      if (failed(funcResult)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Function '" << frame.funcOp.getName()
                   << "' failed during resume\n");
        finalizeProcess(procId, /*killed=*/false);
        return;
      }

      // Check if the function suspended again
      if (state.waiting) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Function '" << frame.funcOp.getName()
                   << "' suspended again during resume\n");
        // Schedule pending delay if any before returning
        if (state.pendingDelayFs > 0) {
          SimTime currentTime = scheduler.getCurrentTime();
          SimTime targetTime = currentTime.advanceTime(state.pendingDelayFs);
          LLVM_DEBUG(llvm::dbgs()
                     << "    Scheduling delay " << state.pendingDelayFs
                     << " fs from function suspend\n");
          state.pendingDelayFs = 0;
          scheduler.getEventScheduler().schedule(
              targetTime, SchedulingRegion::Active,
              Event([this, procId]() { resumeProcess(procId); }));
        }
        return;
      }

      // Function completed - set its results on the call operation
      if (frame.callOp) {
        if (auto callIndirectOp =
                dyn_cast<mlir::func::CallIndirectOp>(frame.callOp)) {
          for (auto [result, retVal] :
               llvm::zip(callIndirectOp.getResults(), results)) {
            setValue(procId, result, retVal);
          }
        } else if (auto callOp = dyn_cast<mlir::func::CallOp>(frame.callOp)) {
          for (auto [result, retVal] :
               llvm::zip(callOp.getResults(), results)) {
            setValue(procId, result, retVal);
          }
        }
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "    Function '" << frame.funcOp.getName()
                 << "' completed, continuing\n");
    }

    // All call stack frames processed, continue with normal process execution
    LLVM_DEBUG(llvm::dbgs()
               << "  Call stack frames exhausted, continuing process\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executing process "
                          << procId << "\n");

  // Execute operations until we suspend or halt
  constexpr size_t kAbortCheckInterval = 1000;
  size_t localStepCount = 0;
  while (!state.halted && !state.waiting) {
    ++localStepCount;
    // Periodically check for abort requests (e.g., from timeout watchdog)
    if (localStepCount % kAbortCheckInterval == 0 && isAbortRequested()) {
      LLVM_DEBUG(llvm::dbgs() << "  Abort requested, halting process\n");
      finalizeProcess(procId, /*killed=*/false);
      if (abortCallback)
        abortCallback();
      break;
    }
    LLVM_DEBUG({
      if (localStepCount <= 10 || localStepCount % 1000 == 0) {
        llvm::dbgs() << "[executeProcess] proc " << procId << " step "
                     << localStepCount;
        if (state.currentOp != state.currentBlock->end())
          llvm::dbgs() << " op: " << state.currentOp->getName().getStringRef();
        llvm::dbgs() << "\n";
      }
    });
    // Global step limit check (totalSteps includes func body ops)
    if (maxProcessSteps > 0 && state.totalSteps > maxProcessSteps) {
      llvm::errs() << "[circt-sim] ERROR(PROCESS_STEP_OVERFLOW): process "
                   << procId;
      if (auto *proc = scheduler.getProcess(procId))
        llvm::errs() << " '" << proc->getName() << "'";
      llvm::errs() << " exceeded " << maxProcessSteps << " total steps";
      llvm::errs() << " (totalSteps=" << state.totalSteps
                   << ", funcBodySteps=" << state.funcBodySteps << ")";
      if (!state.currentFuncName.empty())
        llvm::errs() << " [in " << state.currentFuncName << "]";
      if (state.lastOp)
        llvm::errs() << " (lastOp=" << state.lastOp->getName().getStringRef()
                     << ")";
      llvm::errs() << "\n";
      finalizeProcess(procId, /*killed=*/false);
      break;
    }
    if (!executeStep(procId))
      break;
  }

  // After the loop exits, check if we have pending delay from __moore_delay.
  // If so, schedule the resumption event with the accumulated delay.
  if (state.waiting && state.pendingDelayFs > 0) {
    SimTime currentTime = scheduler.getCurrentTime();
    SimTime targetTime = currentTime.advanceTime(state.pendingDelayFs);

    LLVM_DEBUG(llvm::dbgs() << "  Scheduling __moore_delay resumption: "
                            << state.pendingDelayFs << " fs from time "
                            << currentTime.realTime << " to "
                            << targetTime.realTime << "\n");

    // Reset the pending delay before scheduling
    state.pendingDelayFs = 0;

    // Schedule resumption at the target time
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::Active,
        Event([this, procId]() { resumeProcess(procId); }));
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
  state.lastOp = op;
  ++state.totalSteps;
  if (collectOpStats)
    ++opStats[op->getName().getStringRef()];

  LLVM_DEBUG(llvm::dbgs() << "  Executing: " << *op << "\n");

  // Interpret the operation
  if (failed(interpretOperation(procId, op))) {
    LLVM_DEBUG(llvm::dbgs() << "  Failed to interpret operation\n");
    // Always emit diagnostic for failed operations (not just in debug mode)
    llvm::errs() << "circt-sim: interpretOperation failed for process "
                 << procId << "\n";
    llvm::errs() << "  Operation: ";
    op->print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
    llvm::errs() << "\n";
    llvm::errs() << "  Location: " << op->getLoc() << "\n";
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

ProcessId LLHDProcessInterpreter::resolveProcessHandle(uint64_t handle) {
  if (handle == 0)
    return InvalidProcessId;

  auto it = processHandleToId.find(handle);
  if (it != processHandleToId.end())
    return it->second;

  for (auto &entry : processStates) {
    if (reinterpret_cast<uint64_t>(&entry.second) == handle) {
      processHandleToId[handle] = entry.first;
      return entry.first;
    }
  }

  return InvalidProcessId;
}

void LLHDProcessInterpreter::registerProcessState(ProcessId procId,
                                                  ProcessExecutionState &&state) {
  auto insertResult = processStates.try_emplace(procId, std::move(state));
  if (!insertResult.second)
    insertResult.first->second = std::move(state);

  insertResult.first->second.randomGenerator.seed(
      static_cast<uint32_t>(procId) ^ 0xC0FFEEu);

  uint64_t handle =
      reinterpret_cast<uint64_t>(&insertResult.first->second);
  processHandleToId[handle] = procId;
}

void LLHDProcessInterpreter::notifyProcessAwaiters(ProcessId procId) {
  auto it = processAwaiters.find(procId);
  if (it == processAwaiters.end())
    return;

  for (ProcessId waiterId : it->second)
    resumeProcess(waiterId);

  processAwaiters.erase(it);
}

void LLHDProcessInterpreter::finalizeProcess(ProcessId procId, bool killed) {
  if (auto *proc = scheduler.getProcess(procId)) {
    if (proc->getState() == ProcessState::Terminated) {
      notifyProcessAwaiters(procId);
      return;
    }
  }

  auto it = processStates.find(procId);
  if (it != processStates.end()) {
    it->second.halted = true;
    if (killed)
      it->second.killed = true;
  }

  if (forkJoinManager.getForkGroupForChild(procId))
    forkJoinManager.markChildComplete(procId);

  scheduler.terminateProcess(procId);
  notifyProcessAwaiters(procId);
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

  if (auto waitOp = dyn_cast<llhd::WaitOp>(op)) {
    return interpretWait(procId, waitOp);
  }

  if (auto haltOp = dyn_cast<llhd::HaltOp>(op))
    return interpretHalt(procId, haltOp);

  if (auto constTimeOp = dyn_cast<llhd::ConstantTimeOp>(op))
    return interpretConstantTime(procId, constTimeOp);

  // Handle llhd.sig.extract - extracts a bit range from a signal/ref.
  // For signal-backed refs, propagate the signal mapping.
  // For alloca-backed refs, just succeed - llhd.drv/llhd.prb trace the def chain.
  if (auto sigExtractOp = dyn_cast<llhd::SigExtractOp>(op)) {
    // Try to propagate signal mapping from input to result
    SignalId inputSigId = getSignalId(sigExtractOp.getInput());
    if (inputSigId != 0) {
      valueToSignal[sigExtractOp.getResult()] = inputSigId;
    }
    // For alloca-backed refs, the result is just a narrowed ref.
    // The drv/prb handlers will trace back through this op to find
    // the alloca and compute the bit offset.
    return success();
  }

  // Handle llhd.sig (runtime signal creation for local variables in initial blocks)
  // When global constructors or initial blocks execute, local variable declarations
  // produce llhd.sig operations at runtime. We need to dynamically register these
  // signals so that subsequent probe/drive operations can access them.
  if (auto sigOp = dyn_cast<llhd::SignalOp>(op)) {
    // Get the initial value from the process's value map
    InterpretedValue initVal = getValue(procId, sigOp.getInit());

    // Generate a unique name for this runtime signal
    std::string name = sigOp.getName().value_or("").str();
    if (name.empty()) {
      name = "runtime_sig_" + std::to_string(valueToSignal.size());
    }

    // Get the type width
    Type innerType = sigOp.getInit().getType();
    unsigned width = getTypeWidth(innerType);

    // Register with the scheduler
    SignalId sigId =
        scheduler.registerSignal(name, width, getSignalEncoding(innerType));

    // Store the mapping from the signal result to the signal ID
    valueToSignal[sigOp.getResult()] = sigId;
    signalIdToName[sigId] = name;
    signalIdToType[sigId] = innerType;

    // Set the initial value
    if (!initVal.isX()) {
      SignalValue sv = initVal.toSignalValue();
      scheduler.updateSignal(sigId, sv);
    }

    LLVM_DEBUG(llvm::dbgs() << "  Runtime signal '" << name << "' registered with ID "
                            << sigId << " (width=" << width << ", init="
                            << (initVal.isX() ? "X" : std::to_string(initVal.getUInt64()))
                            << ")\n");

    return success();
  }

  // Sim dialect operations - for $display support
  if (auto printOp = dyn_cast<sim::PrintFormattedProcOp>(op))
    return interpretProcPrint(procId, printOp);

  // Sim dialect operations - for $finish support
  if (auto terminateOp = dyn_cast<sim::TerminateOp>(op))
    return interpretTerminate(procId, terminateOp);

  // Sim dialect operations - for fork/join support
  if (auto forkOp = dyn_cast<sim::SimForkOp>(op))
    return interpretSimFork(procId, forkOp);

  if (auto forkTermOp = dyn_cast<sim::SimForkTerminatorOp>(op))
    return interpretSimForkTerminator(procId, forkTermOp);

  if (auto joinOp = dyn_cast<sim::SimJoinOp>(op))
    return interpretSimJoin(procId, joinOp);

  if (auto joinAnyOp = dyn_cast<sim::SimJoinAnyOp>(op))
    return interpretSimJoinAny(procId, joinAnyOp);

  if (auto waitForkOp = dyn_cast<sim::SimWaitForkOp>(op))
    return interpretSimWaitFork(procId, waitForkOp);

  if (auto disableForkOp = dyn_cast<sim::SimDisableForkOp>(op))
    return interpretSimDisableFork(procId, disableForkOp);

  // Seq dialect operations - seq.yield terminates seq.initial blocks
  if (auto yieldOp = dyn_cast<seq::YieldOp>(op))
    return interpretSeqYield(procId, yieldOp);

  // Moore dialect operations - wait_event suspends until signal change
  // These operations should have been converted to llhd.wait by the
  // MooreToCore pass, but when they appear in function bodies that haven't
  // been inlined, we need to handle them directly.
  if (auto waitEventOp = dyn_cast<moore::WaitEventOp>(op))
    return interpretMooreWaitEvent(procId, waitEventOp);

  // Moore detect_event ops inside wait_event bodies - handled by wait_event
  if (isa<moore::DetectEventOp>(op)) {
    // DetectEventOp is only meaningful inside WaitEventOp - it sets up
    // edge detection. When executed standalone, just skip it.
    return success();
  }

  // Format string operations are consumed by interpretProcPrint - just return
  // success as they don't need individual interpretation
  if (isa<sim::FormatLiteralOp, sim::FormatHexOp, sim::FormatDecOp,
          sim::FormatBinOp, sim::FormatOctOp, sim::FormatCharOp,
          sim::FormatStringConcatOp, sim::FormatDynStringOp>(op)) {
    // These ops are evaluated lazily when their results are used
    return success();
  }

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
      if (isa<llhd::RefType>(arg.getType())) {
        if (SignalId sigId = resolveSignalId(operand))
          valueToSignal[arg] = sigId;
        else
          valueToSignal.erase(arg);
      }
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
        if (isa<llhd::RefType>(arg.getType())) {
          if (SignalId sigId = resolveSignalId(operand))
            valueToSignal[arg] = sigId;
          else
            valueToSignal.erase(arg);
        }
      }
    } else {
      // False branch (or X treated as false)
      state.currentBlock = condBranchOp.getFalseDest();
      state.currentOp = state.currentBlock->begin();
      for (auto [arg, operand] :
           llvm::zip(state.currentBlock->getArguments(),
                     condBranchOp.getFalseDestOperands())) {
        state.valueMap[arg] = getValue(procId, operand);
        if (isa<llhd::RefType>(arg.getType())) {
          if (SignalId sigId = resolveSignalId(operand))
            valueToSignal[arg] = sigId;
          else
            valueToSignal.erase(arg);
        }
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
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    // Normalize widths for comparison - use the larger of the two widths
    unsigned compareWidth = std::max(lhsVal.getBitWidth(), rhsVal.getBitWidth());
    normalizeWidths(lhsVal, rhsVal, compareWidth);
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
    unsigned targetWidth = getTypeWidth(andOp.getType());
    if (targetWidth <= 64) {
      uint64_t result = ~0ULL;
      uint64_t mask =
          (targetWidth == 64) ? ~0ULL : ((1ULL << targetWidth) - 1);
      for (Value operand : andOp.getOperands()) {
        InterpretedValue value = getValue(procId, operand);
        uint64_t operandVal = 0;
        if (!getMaskedUInt64(value, targetWidth, operandVal)) {
          setValue(procId, andOp.getResult(),
                   InterpretedValue::makeX(targetWidth));
          return success();
        }
        if ((operandVal & mask) == mask)
          continue;
        result &= operandVal;
        if (result == 0)
          break;
      }
      if (targetWidth < 64)
        result &= mask;
      setValue(procId, andOp.getResult(),
               InterpretedValue(result, targetWidth));
      return success();
    }
    llvm::APInt result(targetWidth, 0);
    result.setAllBits(); // Start with all 1s for AND
    for (Value operand : andOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, andOp.getResult(),
                 InterpretedValue::makeX(targetWidth));
        return success();
      }
      APInt operandVal = value.getAPInt();
      // Normalize to target width
      if (operandVal.getBitWidth() < targetWidth)
        operandVal = operandVal.zext(targetWidth);
      else if (operandVal.getBitWidth() > targetWidth)
        operandVal = operandVal.trunc(targetWidth);
      if (operandVal.isAllOnes())
        continue;
      result &= operandVal;
      if (result.isZero())
        break;
    }
    setValue(procId, andOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    unsigned targetWidth = getTypeWidth(orOp.getType());
    if (targetWidth <= 64) {
      uint64_t result = 0;
      uint64_t mask =
          (targetWidth == 64) ? ~0ULL : ((1ULL << targetWidth) - 1);
      for (Value operand : orOp.getOperands()) {
        InterpretedValue value = getValue(procId, operand);
        uint64_t operandVal = 0;
        if (!getMaskedUInt64(value, targetWidth, operandVal)) {
          setValue(procId, orOp.getResult(),
                   InterpretedValue::makeX(targetWidth));
          return success();
        }
        if (operandVal == 0)
          continue;
        result |= operandVal;
        if ((result & mask) == mask)
          break;
      }
      if (targetWidth < 64)
        result &= mask;
      setValue(procId, orOp.getResult(),
               InterpretedValue(result, targetWidth));
      return success();
    }
    llvm::APInt result(targetWidth, 0); // Start with all 0s for OR
    for (Value operand : orOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, orOp.getResult(),
                 InterpretedValue::makeX(targetWidth));
        return success();
      }
      APInt operandVal = value.getAPInt();
      // Normalize to target width
      if (operandVal.getBitWidth() < targetWidth)
        operandVal = operandVal.zext(targetWidth);
      else if (operandVal.getBitWidth() > targetWidth)
        operandVal = operandVal.trunc(targetWidth);
      if (operandVal.isZero())
        continue;
      result |= operandVal;
      if (result.isAllOnes())
        break;
    }
    setValue(procId, orOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    unsigned targetWidth = getTypeWidth(xorOp.getType());
    if (targetWidth <= 64) {
      uint64_t result = 0;
      uint64_t mask =
          (targetWidth == 64) ? ~0ULL : ((1ULL << targetWidth) - 1);
      bool invert = false;
      for (Value operand : xorOp.getOperands()) {
        InterpretedValue value = getValue(procId, operand);
        uint64_t operandVal = 0;
        if (!getMaskedUInt64(value, targetWidth, operandVal)) {
          setValue(procId, xorOp.getResult(),
                   InterpretedValue::makeX(targetWidth));
          return success();
        }
        operandVal &= mask;
        if (operandVal == 0)
          continue;
        if (operandVal == mask) {
          invert = !invert;
          continue;
        }
        result ^= operandVal;
      }
      if (invert)
        result ^= mask;
      if (targetWidth < 64)
        result &= mask;
      setValue(procId, xorOp.getResult(),
               InterpretedValue(result, targetWidth));
      return success();
    }
    llvm::APInt result(targetWidth, 0); // Start with all 0s for XOR
    bool invert = false;
    for (Value operand : xorOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, xorOp.getResult(),
                 InterpretedValue::makeX(targetWidth));
        return success();
      }
      APInt operandVal = value.getAPInt();
      // Normalize to target width
      if (operandVal.getBitWidth() < targetWidth)
        operandVal = operandVal.zext(targetWidth);
      else if (operandVal.getBitWidth() > targetWidth)
        operandVal = operandVal.trunc(targetWidth);
      if (operandVal.isZero())
        continue;
      if (operandVal.isAllOnes()) {
        invert = !invert;
        continue;
      }
      result ^= operandVal;
    }
    if (invert)
      result ^= APInt::getAllOnes(targetWidth);
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
    unsigned targetWidth = getTypeWidth(subOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, subOp.getResult(),
               InterpretedValue::makeX(targetWidth));
      return success();
    }
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    normalizeWidths(lhsVal, rhsVal, targetWidth);
    setValue(procId, subOp.getResult(),
             InterpretedValue(lhsVal - rhsVal));
    return success();
  }

  if (auto mulOp = dyn_cast<comb::MulOp>(op)) {
    unsigned targetWidth = getTypeWidth(mulOp.getType());
    llvm::APInt result(targetWidth, 1); // Start with 1 for multiplication
    for (Value operand : mulOp.getOperands()) {
      InterpretedValue value = getValue(procId, operand);
      if (value.isX()) {
        setValue(procId, mulOp.getResult(),
                 InterpretedValue::makeX(targetWidth));
        return success();
      }
      APInt operandVal = value.getAPInt();
      // Normalize to target width
      if (operandVal.getBitWidth() < targetWidth)
        operandVal = operandVal.zext(targetWidth);
      else if (operandVal.getBitWidth() > targetWidth)
        operandVal = operandVal.trunc(targetWidth);
      result *= operandVal;
    }
    setValue(procId, mulOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto divsOp = dyn_cast<comb::DivSOp>(op)) {
    InterpretedValue lhs = getValue(procId, divsOp.getLhs());
    InterpretedValue rhs = getValue(procId, divsOp.getRhs());
    unsigned targetWidth = getTypeWidth(divsOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, divsOp.getResult(),
               InterpretedValue::makeX(targetWidth));
      return success();
    }
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    normalizeWidths(lhsVal, rhsVal, targetWidth);
    setValue(procId, divsOp.getResult(),
             InterpretedValue(lhsVal.sdiv(rhsVal)));
    return success();
  }

  if (auto divuOp = dyn_cast<comb::DivUOp>(op)) {
    InterpretedValue lhs = getValue(procId, divuOp.getLhs());
    InterpretedValue rhs = getValue(procId, divuOp.getRhs());
    unsigned targetWidth = getTypeWidth(divuOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, divuOp.getResult(),
               InterpretedValue::makeX(targetWidth));
      return success();
    }
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    normalizeWidths(lhsVal, rhsVal, targetWidth);
    setValue(procId, divuOp.getResult(),
             InterpretedValue(lhsVal.udiv(rhsVal)));
    return success();
  }

  if (auto modsOp = dyn_cast<comb::ModSOp>(op)) {
    InterpretedValue lhs = getValue(procId, modsOp.getLhs());
    InterpretedValue rhs = getValue(procId, modsOp.getRhs());
    unsigned targetWidth = getTypeWidth(modsOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, modsOp.getResult(),
               InterpretedValue::makeX(targetWidth));
      return success();
    }
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    normalizeWidths(lhsVal, rhsVal, targetWidth);
    setValue(procId, modsOp.getResult(),
             InterpretedValue(lhsVal.srem(rhsVal)));
    return success();
  }

  if (auto moduOp = dyn_cast<comb::ModUOp>(op)) {
    InterpretedValue lhs = getValue(procId, moduOp.getLhs());
    InterpretedValue rhs = getValue(procId, moduOp.getRhs());
    unsigned targetWidth = getTypeWidth(moduOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, moduOp.getResult(),
               InterpretedValue::makeX(targetWidth));
      return success();
    }
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    normalizeWidths(lhsVal, rhsVal, targetWidth);
    setValue(procId, moduOp.getResult(),
             InterpretedValue(lhsVal.urem(rhsVal)));
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
    if (width <= 64 && lowBit + width <= 64) {
      uint64_t inputVal = input.getUInt64();
      uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
      uint64_t result = (inputVal >> lowBit) & mask;
      setValue(procId, extractOp.getResult(),
               InterpretedValue(result, width));
      return success();
    }
    if (width == 1 && input.getWidth() > 64) {
      bool bit = input.getAPInt()[lowBit];
      setValue(procId, extractOp.getResult(), InterpretedValue(bit, 1));
      return success();
    }
    if (width <= 64 && input.getWidth() <= 64) {
      uint64_t inputVal = input.getUInt64();
      uint64_t mask = (width == 64) ? ~0ULL : ((1ULL << width) - 1);
      uint64_t result = (inputVal >> lowBit) & mask;
      setValue(procId, extractOp.getResult(),
               InterpretedValue(result, width));
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
    unsigned targetWidth = getTypeWidth(addOp.getType());
    // comb.add is variadic - handle N operands (canonicalization merges nested adds)
    APInt result = APInt::getZero(targetWidth);
    for (Value operand : addOp.getOperands()) {
      InterpretedValue val = getValue(procId, operand);
      if (val.isX()) {
        setValue(procId, addOp.getResult(),
                 InterpretedValue::makeX(targetWidth));
        return success();
      }
      APInt v = val.getAPInt();
      if (v.getBitWidth() != targetWidth)
        v = v.zextOrTrunc(targetWidth);
      result += v;
    }
    setValue(procId, addOp.getResult(), InterpretedValue(result));
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
    unsigned targetWidth = getTypeWidth(arithAddIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithAddIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithAddIOp.getResult(),
               InterpretedValue(lhsVal + rhsVal));
    }
    return success();
  }

  if (auto arithSubIOp = dyn_cast<mlir::arith::SubIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithSubIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithSubIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithSubIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithSubIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithSubIOp.getResult(),
               InterpretedValue(lhsVal - rhsVal));
    }
    return success();
  }

  if (auto arithMulIOp = dyn_cast<mlir::arith::MulIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithMulIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithMulIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithMulIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithMulIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithMulIOp.getResult(),
               InterpretedValue(lhsVal * rhsVal));
    }
    return success();
  }

  if (auto arithDivSIOp = dyn_cast<mlir::arith::DivSIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithDivSIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithDivSIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithDivSIOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithDivSIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithDivSIOp.getResult(),
               InterpretedValue(lhsVal.sdiv(rhsVal)));
    }
    return success();
  }

  if (auto arithDivUIOp = dyn_cast<mlir::arith::DivUIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithDivUIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithDivUIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithDivUIOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithDivUIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithDivUIOp.getResult(),
               InterpretedValue(lhsVal.udiv(rhsVal)));
    }
    return success();
  }

  if (auto arithRemSIOp = dyn_cast<mlir::arith::RemSIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithRemSIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithRemSIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithRemSIOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithRemSIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithRemSIOp.getResult(),
               InterpretedValue(lhsVal.srem(rhsVal)));
    }
    return success();
  }

  if (auto arithRemUIOp = dyn_cast<mlir::arith::RemUIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithRemUIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithRemUIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithRemUIOp.getType());
    if (lhs.isX() || rhs.isX() || rhs.getAPInt().isZero()) {
      setValue(procId, arithRemUIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithRemUIOp.getResult(),
               InterpretedValue(lhsVal.urem(rhsVal)));
    }
    return success();
  }

  if (auto arithAndIOp = dyn_cast<mlir::arith::AndIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithAndIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithAndIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithAndIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithAndIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithAndIOp.getResult(),
               InterpretedValue(lhsVal & rhsVal));
    }
    return success();
  }

  if (auto arithOrIOp = dyn_cast<mlir::arith::OrIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithOrIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithOrIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithOrIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithOrIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithOrIOp.getResult(),
               InterpretedValue(lhsVal | rhsVal));
    }
    return success();
  }

  if (auto arithXOrIOp = dyn_cast<mlir::arith::XOrIOp>(op)) {
    InterpretedValue lhs = getValue(procId, arithXOrIOp.getLhs());
    InterpretedValue rhs = getValue(procId, arithXOrIOp.getRhs());
    unsigned targetWidth = getTypeWidth(arithXOrIOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, arithXOrIOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      setValue(procId, arithXOrIOp.getResult(),
               InterpretedValue(lhsVal ^ rhsVal));
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
    APInt lhsVal = lhs.getAPInt();
    APInt rhsVal = rhs.getAPInt();
    // Normalize widths for comparison - use the larger of the two widths
    unsigned compareWidth = std::max(lhsVal.getBitWidth(), rhsVal.getBitWidth());
    normalizeWidths(lhsVal, rhsVal, compareWidth);
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

  // Handle func.call_indirect for virtual method calls
  if (auto callIndirectOp = dyn_cast<mlir::func::CallIndirectOp>(op)) {
    // The callee is the first operand (function pointer)
    Value calleeValue = callIndirectOp.getCallee();
    InterpretedValue funcPtrVal = getValue(procId, calleeValue);

    // Throttle vtable dispatch warnings to prevent flooding stderr during
    // UVM initialization. Both X function pointers and unmapped addresses
    // share a single counter.
    auto emitVtableWarning = [&](StringRef reason) {
      static unsigned vtableWarnCount = 0;
      if (vtableWarnCount < 30) {
        ++vtableWarnCount;
        llvm::errs() << "[circt-sim] WARNING: virtual method call "
                     << "(func.call_indirect) failed: " << reason
                     << ". Callee operand: ";
        calleeValue.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
        llvm::errs() << " (type: " << calleeValue.getType() << ")\n";
      } else if (vtableWarnCount == 30) {
        ++vtableWarnCount;
        llvm::errs() << "[circt-sim] (suppressing further vtable warnings)\n";
      }
    };

    if (funcPtrVal.isX()) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: callee is X "
                              << "(uninitialized vtable pointer)\n");

      // Fallback: try to resolve the virtual method statically by tracing
      // the SSA chain back to the vtable GEP pattern:
      //   calleeValue = unrealized_conversion_cast(llvm.load(llvm.getelementptr(
      //                     vtablePtr, [0, methodIndex])))
      // where vtablePtr = llvm.load(llvm.getelementptr(objPtr, [0, ..., 1]))
      // From the outer GEP's struct type we get the class name, construct
      // "ClassName::__vtable__", and read the function address at methodIndex.
      bool resolved = false;
      do {
        // Step 1: trace calleeValue -> unrealized_conversion_cast input
        auto castOp =
            calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          break;
        Value rawPtr = castOp.getInputs()[0];

        // Step 2: rawPtr should come from llvm.load (loads func ptr from vtable)
        auto funcPtrLoad = rawPtr.getDefiningOp<LLVM::LoadOp>();
        if (!funcPtrLoad)
          break;

        // Step 3: the load address comes from a GEP into the vtable array
        auto vtableGEP = funcPtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!vtableGEP)
          break;

        // Extract the method index from the GEP indices (last index)
        auto vtableIndices = vtableGEP.getIndices();
        if (vtableIndices.size() < 2)
          break;
        auto lastIdx = vtableIndices[vtableIndices.size() - 1];
        int64_t methodIndex = -1;
        if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(lastIdx))
          methodIndex = intAttr.getInt();
        else if (auto dynIdx = llvm::dyn_cast_if_present<Value>(lastIdx)) {
          InterpretedValue dynVal = getValue(procId, dynIdx);
          if (!dynVal.isX())
            methodIndex = static_cast<int64_t>(dynVal.getUInt64());
        }
        if (methodIndex < 0)
          break;

        // Step 4: vtableGEP base = vtable pointer from llvm.load
        auto vtablePtrLoad =
            vtableGEP.getBase().getDefiningOp<LLVM::LoadOp>();
        if (!vtablePtrLoad)
          break;

        // Step 5: the vtable pointer load address comes from GEP on the object
        auto objGEP =
            vtablePtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!objGEP)
          break;

        // Extract the struct type name from the outer GEP's element type
        std::string vtableGlobalName;
        if (auto structTy =
                dyn_cast<LLVM::LLVMStructType>(objGEP.getElemType())) {
          if (structTy.isIdentified()) {
            vtableGlobalName = structTy.getName().str() + "::__vtable__";
          }
        }
        if (vtableGlobalName.empty())
          break;

        // Step 6: find the vtable global and read the function address
        auto globalIt = globalMemoryBlocks.find(vtableGlobalName);
        if (globalIt == globalMemoryBlocks.end())
          break;

        auto &vtableBlock = globalIt->second;
        unsigned slotOffset = methodIndex * 8;
        if (slotOffset + 8 > vtableBlock.size)
          break;

        // Read 8-byte function address (little-endian) from vtable memory
        uint64_t resolvedFuncAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          resolvedFuncAddr |=
              static_cast<uint64_t>(vtableBlock.data[slotOffset + i]) << (i * 8);

        if (resolvedFuncAddr == 0)
          break; // Slot is empty (no function registered)

        auto funcIt = addressToFunction.find(resolvedFuncAddr);
        if (funcIt == addressToFunction.end())
          break;

        StringRef resolvedName = funcIt->second;
        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: fallback vtable resolution: "
                   << vtableGlobalName << "[" << methodIndex << "] -> "
                   << resolvedName << "\n");

        // Look up the function
        auto &state = processStates[procId];
        Operation *parent = state.processOrInitialOp;
        while (parent && !isa<ModuleOp>(parent))
          parent = parent->getParentOp();
        ModuleOp moduleOp = parent ? cast<ModuleOp>(parent) : rootModule;
        auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(resolvedName);
        if (!funcOp)
          break;

        // Gather arguments
        SmallVector<InterpretedValue, 4> args;
        for (Value arg : callIndirectOp.getArgOperands())
          args.push_back(getValue(procId, arg));

        // Dispatch the call
        auto &callState = processStates[procId];
        ++callState.callDepth;
        SmallVector<InterpretedValue, 4> results;
        auto callResult = interpretFuncBody(procId, funcOp, args, results,
                                            callIndirectOp);
        --callState.callDepth;

        if (failed(callResult))
          break;

        // Set return values
        for (auto [result, val] :
             llvm::zip(callIndirectOp.getResults(), results))
          setValue(procId, result, val);

        resolved = true;
      } while (false);

      if (!resolved) {
        emitVtableWarning("function pointer is X (uninitialized)");
        // Return zero/null instead of X to prevent cascading X-propagation
        // in UVM code paths that check return values for null.
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
      }
      return success();
    }

    // Look up the function name from the vtable
    uint64_t funcAddr = funcPtrVal.getUInt64();
    auto it = addressToFunction.find(funcAddr);
    if (it == addressToFunction.end()) {
      // Runtime vtable pointer is corrupt or unmapped. Try static resolution
      // by tracing the SSA chain back to the vtable global, which is always
      // correct regardless of runtime memory corruption.
      bool staticResolved = false;
      do {
        auto castOp =
            calleeValue.getDefiningOp<mlir::UnrealizedConversionCastOp>();
        if (!castOp || castOp.getInputs().size() != 1)
          break;
        Value rawPtr = castOp.getInputs()[0];
        auto funcPtrLoad = rawPtr.getDefiningOp<LLVM::LoadOp>();
        if (!funcPtrLoad)
          break;
        auto vtableGEP =
            funcPtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!vtableGEP)
          break;
        auto vtableIndices = vtableGEP.getIndices();
        if (vtableIndices.size() < 2)
          break;
        auto lastIdx = vtableIndices[vtableIndices.size() - 1];
        int64_t methodIndex = -1;
        if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(lastIdx))
          methodIndex = intAttr.getInt();
        else if (auto dynIdx = llvm::dyn_cast_if_present<Value>(lastIdx)) {
          InterpretedValue dynVal = getValue(procId, dynIdx);
          if (!dynVal.isX())
            methodIndex = static_cast<int64_t>(dynVal.getUInt64());
        }
        if (methodIndex < 0)
          break;
        auto vtablePtrLoad =
            vtableGEP.getBase().getDefiningOp<LLVM::LoadOp>();
        if (!vtablePtrLoad)
          break;
        auto objGEP =
            vtablePtrLoad.getAddr().getDefiningOp<LLVM::GEPOp>();
        if (!objGEP)
          break;
        std::string vtableGlobalName;
        if (auto structTy =
                dyn_cast<LLVM::LLVMStructType>(objGEP.getElemType())) {
          if (structTy.isIdentified())
            vtableGlobalName = structTy.getName().str() + "::__vtable__";
        }
        if (vtableGlobalName.empty())
          break;
        auto globalIt = globalMemoryBlocks.find(vtableGlobalName);
        if (globalIt == globalMemoryBlocks.end())
          break;
        auto &vtableBlock = globalIt->second;
        unsigned slotOffset = methodIndex * 8;
        if (slotOffset + 8 > vtableBlock.size)
          break;
        uint64_t resolvedFuncAddr = 0;
        for (unsigned i = 0; i < 8; ++i)
          resolvedFuncAddr |=
              static_cast<uint64_t>(vtableBlock.data[slotOffset + i])
              << (i * 8);
        if (resolvedFuncAddr == 0)
          break;
        auto funcIt2 = addressToFunction.find(resolvedFuncAddr);
        if (funcIt2 == addressToFunction.end())
          break;
        StringRef resolvedName = funcIt2->second;
        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: static fallback: "
                   << vtableGlobalName << "[" << methodIndex << "] -> "
                   << resolvedName << "\n");
        auto &st2 = processStates[procId];
        Operation *par = st2.processOrInitialOp;
        while (par && !isa<ModuleOp>(par))
          par = par->getParentOp();
        ModuleOp modOp = par ? cast<ModuleOp>(par) : rootModule;
        auto fOp = modOp.lookupSymbol<func::FuncOp>(resolvedName);
        if (!fOp)
          break;
        SmallVector<InterpretedValue, 4> sArgs;
        for (Value arg : callIndirectOp.getArgOperands())
          sArgs.push_back(getValue(procId, arg));
        auto &cs2 = processStates[procId];
        ++cs2.callDepth;
        SmallVector<InterpretedValue, 4> sResults;
        auto callRes = interpretFuncBody(procId, fOp, sArgs, sResults,
                                         callIndirectOp);
        --cs2.callDepth;
        if (failed(callRes))
          break;
        for (auto [result, val] :
             llvm::zip(callIndirectOp.getResults(), sResults))
          setValue(procId, result, val);
        staticResolved = true;
      } while (false);
      if (!staticResolved) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  func.call_indirect: address 0x"
                   << llvm::format_hex(funcAddr, 16)
                   << " not in vtable map\n");
        std::string reason = "address " +
            llvm::utohexstr(funcAddr) + " not found in vtable map";
        emitVtableWarning(reason);
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
      }
      return success();
    }

    StringRef calleeName = it->second;
    LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: resolved 0x"
                            << llvm::format_hex(funcAddr, 16)
                            << " -> " << calleeName << "\n");

    // Look up the function
    auto &state = processStates[procId];
    Operation *parent = state.processOrInitialOp;
    while (parent && !isa<ModuleOp>(parent))
      parent = parent->getParentOp();

    // Use rootModule as fallback for global constructors
    ModuleOp moduleOp = parent ? cast<ModuleOp>(parent) : rootModule;
    auto funcOp = moduleOp.lookupSymbol<func::FuncOp>(calleeName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: function '" << calleeName
                              << "' not found\n");
      for (Value result : callIndirectOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
      return success();
    }

    // Gather argument values (use getArgOperands to get just the arguments, not callee)
    SmallVector<InterpretedValue, 4> args;
    for (Value arg : callIndirectOp.getArgOperands()) {
      args.push_back(getValue(procId, arg));
    }

    // Check call depth to prevent stack overflow from deep recursion (UVM patterns)
    auto &callState = processStates[procId];
    constexpr size_t maxCallDepth = 200;
    if (callState.callDepth >= maxCallDepth) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: max call depth ("
                              << maxCallDepth
                              << ") exceeded, returning zero\n");
      for (Value result : callIndirectOp.getResults()) {
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
      }
      return success();
    }

    // Recursive DFS depth detection (same as func.call handler)
    Operation *indFuncKey = funcOp.getOperation();
    uint64_t indArg0Val = 0;
    bool indHasArg0 = !args.empty() && !args[0].isX();
    if (indHasArg0)
      indArg0Val = args[0].getUInt64();
    constexpr unsigned maxRecursionDepth = 20;
    auto &indDepthMap = callState.recursionVisited[indFuncKey];
    if (indHasArg0 && callState.callDepth > 0) {
      unsigned &depth = indDepthMap[indArg0Val];
      if (depth >= maxRecursionDepth) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  call_indirect: recursion depth " << depth
                   << " exceeded for '" << calleeName << "' with arg0=0x"
                   << llvm::format_hex(indArg0Val, 16) << "\n");
        for (Value result : callIndirectOp.getResults()) {
          unsigned width = getTypeWidth(result.getType());
          setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
        }
        return success();
      }
    }
    bool indAddedToVisited = indHasArg0;
    if (indHasArg0)
      ++indDepthMap[indArg0Val];

    // Call the function with depth tracking
    ++callState.callDepth;
    SmallVector<InterpretedValue, 2> results;
    // Pass the call operation so it can be saved in call stack frames
    LogicalResult funcResult =
        interpretFuncBody(procId, funcOp, args, results, callIndirectOp);
    --callState.callDepth;

    // Decrement depth counter after returning
    if (indAddedToVisited) {
      auto &depthRef = processStates[procId].recursionVisited[indFuncKey][indArg0Val];
      if (depthRef > 0)
        --depthRef;
    }

    if (failed(funcResult))
      return failure();

    // Check if process suspended during function execution (e.g., due to wait)
    // If so, return early without setting results - the function didn't complete
    auto &postCallState = processStates[procId];
    if (postCallState.waiting) {
      LLVM_DEBUG(llvm::dbgs() << "  call_indirect: process suspended during call to '"
                              << calleeName << "'\n");
      return success();
    }

    // Set results
    for (auto [result, retVal] : llvm::zip(callIndirectOp.getResults(), results)) {
      setValue(procId, result, retVal);
    }

    return success();
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
      APInt elemVal(elementWidth, 0);
      if (elem.isX()) {
        if (auto encoded = getEncodedUnknownForType(arrayType.getElementType())) {
          elemVal = encoded->zextOrTrunc(elementWidth);
        } else {
          hasX = true;
          break;
        }
      } else {
        elemVal = elem.getAPInt();
      }
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

    if (indexVal.isX()) {
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

    if (arrayVal.isX()) {
      if (auto encoded = getEncodedUnknownForType(arrayType.getElementType())) {
        setValue(procId, arrayGetOp.getResult(),
                 InterpretedValue(encoded->zextOrTrunc(elementWidth)));
      } else {
        setValue(procId, arrayGetOp.getResult(),
                 InterpretedValue::makeX(elementWidth));
      }
      return success();
    }

    // Array element 0 is at LSB (offset 0), matching CIRCT hw dialect convention.
    unsigned offset = idx * elementWidth;
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

    // Array element 0 is at LSB (offset 0), matching CIRCT hw dialect convention.
    unsigned offset = lowIdx * elementWidth;
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
  // HW Struct Operations
  //===--------------------------------------------------------------------===//

  if (auto structExtractOp = dyn_cast<hw::StructExtractOp>(op)) {
    InterpretedValue structVal = getValue(procId, structExtractOp.getInput());
    if (structVal.isX()) {
      if (auto encoded = getEncodedUnknownForType(structExtractOp.getType())) {
        setValue(procId, structExtractOp.getResult(),
                 InterpretedValue(*encoded));
      } else {
        setValue(procId, structExtractOp.getResult(),
                 InterpretedValue::makeX(getTypeWidth(structExtractOp.getType())));
      }
      return success();
    }

    // Get the struct type and field info
    auto structType = cast<hw::StructType>(structExtractOp.getInput().getType());
    auto elements = structType.getElements();
    uint32_t fieldIndex = structExtractOp.getFieldIndex();

    // Calculate the bit offset for the field
    // Fields are laid out from high bits to low bits in order
    unsigned fieldOffset = 0;
    for (size_t i = fieldIndex + 1; i < elements.size(); ++i) {
      fieldOffset += getTypeWidth(elements[i].type);
    }

    unsigned fieldWidth = getTypeWidth(elements[fieldIndex].type);
    APInt fieldValue = structVal.getAPInt().extractBits(fieldWidth, fieldOffset);
    setValue(procId, structExtractOp.getResult(), InterpretedValue(fieldValue));
    return success();
  }

  if (auto structCreateOp = dyn_cast<hw::StructCreateOp>(op)) {
    auto structType = cast<hw::StructType>(structCreateOp.getType());
    unsigned totalWidth = getTypeWidth(structType);

    APInt result(totalWidth, 0);
    bool hasX = false;
    auto elements = structType.getElements();
    unsigned bitOffset = totalWidth;

    for (size_t i = 0; i < structCreateOp.getInput().size(); ++i) {
      InterpretedValue val = getValue(procId, structCreateOp.getInput()[i]);
      unsigned fieldWidth = getTypeWidth(elements[i].type);
      bitOffset -= fieldWidth;
      if (val.isX()) {
        if (auto encoded = getEncodedUnknownForType(elements[i].type)) {
          result.insertBits(encoded->zextOrTrunc(fieldWidth), bitOffset);
          continue;
        }
        hasX = true;
        break;
      }
      result.insertBits(val.getAPInt(), bitOffset);
    }

    if (hasX) {
      setValue(procId, structCreateOp.getResult(),
               InterpretedValue::makeX(totalWidth));
    } else {
      setValue(procId, structCreateOp.getResult(), InterpretedValue(result));
    }
    return success();
  }

  if (auto structInjectOp = dyn_cast<hw::StructInjectOp>(op)) {
    InterpretedValue structVal = getValue(procId, structInjectOp.getInput());
    InterpretedValue newVal = getValue(procId, structInjectOp.getNewValue());
    auto structType = cast<hw::StructType>(structInjectOp.getInput().getType());
    unsigned totalWidth = getTypeWidth(structType);
    if (structVal.isX() || newVal.isX()) {
      setValue(procId, structInjectOp.getResult(),
               InterpretedValue::makeX(totalWidth));
      return success();
    }

    auto fieldIndexOpt = structType.getFieldIndex(structInjectOp.getFieldName());
    if (!fieldIndexOpt) {
      setValue(procId, structInjectOp.getResult(),
               InterpretedValue::makeX(totalWidth));
      return success();
    }
    unsigned fieldIndex = *fieldIndexOpt;
    auto elements = structType.getElements();

    unsigned fieldOffset = 0;
    for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
      fieldOffset += getTypeWidth(elements[i].type);

    unsigned fieldWidth = getTypeWidth(elements[fieldIndex].type);
    APInt result = structVal.getAPInt();
    APInt fieldValue = newVal.getAPInt();
    if (fieldValue.getBitWidth() < fieldWidth)
      fieldValue = fieldValue.zext(fieldWidth);
    else if (fieldValue.getBitWidth() > fieldWidth)
      fieldValue = fieldValue.trunc(fieldWidth);
    result.insertBits(fieldValue, fieldOffset);

    setValue(procId, structInjectOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (op->getName().getStringRef() == "hw.struct_inject") {
    Value input = op->getOperand(0);
    Value newValue = op->getOperand(1);
    auto structType = cast<hw::StructType>(input.getType());
    unsigned totalWidth = getTypeWidth(structType);
    auto fieldIndexAttr = op->getAttrOfType<IntegerAttr>("fieldIndex");
    if (!fieldIndexAttr) {
      for (Value result : op->getResults())
        setValue(procId, result, InterpretedValue::makeX(totalWidth));
      return success();
    }
    unsigned fieldIndex = fieldIndexAttr.getValue().getZExtValue();
    auto elements = structType.getElements();
    InterpretedValue structVal = getValue(procId, input);
    InterpretedValue newVal = getValue(procId, newValue);
    if (structVal.isX() || newVal.isX()) {
      for (Value result : op->getResults())
        setValue(procId, result, InterpretedValue::makeX(totalWidth));
      return success();
    }

    unsigned fieldOffset = 0;
    for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
      fieldOffset += getTypeWidth(elements[i].type);
    unsigned fieldWidth = getTypeWidth(elements[fieldIndex].type);
    APInt result = structVal.getAPInt();
    APInt fieldValue = newVal.getAPInt();
    if (fieldValue.getBitWidth() < fieldWidth)
      fieldValue = fieldValue.zext(fieldWidth);
    else if (fieldValue.getBitWidth() > fieldWidth)
      fieldValue = fieldValue.trunc(fieldWidth);
    result.insertBits(fieldValue, fieldOffset);

    for (Value resultVal : op->getResults())
      setValue(procId, resultVal, InterpretedValue(result));
    return success();
  }

  if (auto aggConstOp = dyn_cast<hw::AggregateConstantOp>(op)) {
    APInt value = flattenAggregateConstant(aggConstOp);
    setValue(procId, aggConstOp.getResult(), InterpretedValue(value));
    return success();
  }

  if (auto bitcastOp = dyn_cast<hw::BitcastOp>(op)) {
    InterpretedValue inputVal = getValue(procId, bitcastOp.getInput());
    // Bitcast preserves the raw bits, just reinterprets the type
    unsigned outputWidth = getTypeWidth(bitcastOp.getType());
    if (inputVal.isX()) {
      setValue(procId, bitcastOp.getResult(),
               InterpretedValue::makeX(outputWidth));
    } else {
      // Extend or truncate to match output width if necessary
      APInt result = inputVal.getAPInt();
      if (result.getBitWidth() < outputWidth)
        result = result.zext(outputWidth);
      else if (result.getBitWidth() > outputWidth)
        result = result.trunc(outputWidth);
      setValue(procId, bitcastOp.getResult(), InterpretedValue(result));
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

  //===--------------------------------------------------------------------===//
  // LLVM Dialect Operations
  //===--------------------------------------------------------------------===//

  // LLVM constant operation (llvm.mlir.constant)
  if (auto llvmConstOp = dyn_cast<LLVM::ConstantOp>(op)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(llvmConstOp.getValue())) {
      setValue(procId, llvmConstOp.getResult(),
               InterpretedValue(intAttr.getValue()));
    } else {
      // For non-integer constants, create an X value
      setValue(procId, llvmConstOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(llvmConstOp.getType())));
    }
    return success();
  }

  if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(op))
    return interpretLLVMAlloca(procId, allocaOp);

  if (auto loadOp = dyn_cast<LLVM::LoadOp>(op))
    return interpretLLVMLoad(procId, loadOp);

  if (auto storeOp = dyn_cast<LLVM::StoreOp>(op))
    return interpretLLVMStore(procId, storeOp);

  if (auto gepOp = dyn_cast<LLVM::GEPOp>(op))
    return interpretLLVMGEP(procId, gepOp);

  if (auto addrOfOp = dyn_cast<LLVM::AddressOfOp>(op))
    return interpretLLVMAddressOf(procId, addrOfOp);

  if (auto llvmCallOp = dyn_cast<LLVM::CallOp>(op))
    return interpretLLVMCall(procId, llvmCallOp);

  // LLVM return op is handled by call interpreter
  if (isa<LLVM::ReturnOp>(op))
    return success();

  // LLVM unreachable indicates control should not reach here (e.g., after $finish)
  // Halt the process when this is reached
  if (isa<LLVM::UnreachableOp>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.unreachable reached - halting process\n");
    finalizeProcess(procId, /*killed=*/false);
    return success();
  }

  // LLVM undef creates an undefined value
  if (auto undefOp = dyn_cast<LLVM::UndefOp>(op)) {
    setValue(procId, undefOp.getResult(),
             InterpretedValue::makeX(getTypeWidth(undefOp.getType())));
    return success();
  }

  // LLVM null pointer constant
  if (auto nullOp = dyn_cast<LLVM::ZeroOp>(op)) {
    setValue(procId, nullOp.getResult(), InterpretedValue(0, 64));
    return success();
  }

  // LLVM inttoptr
  if (auto intToPtrOp = dyn_cast<LLVM::IntToPtrOp>(op)) {
    InterpretedValue input = getValue(procId, intToPtrOp.getArg());
    setValue(procId, intToPtrOp.getResult(), input);
    return success();
  }

  // LLVM ptrtoint
  if (auto ptrToIntOp = dyn_cast<LLVM::PtrToIntOp>(op)) {
    InterpretedValue input = getValue(procId, ptrToIntOp.getArg());
    unsigned width = getTypeWidth(ptrToIntOp.getType());
    if (input.isX()) {
      setValue(procId, ptrToIntOp.getResult(), InterpretedValue::makeX(width));
    } else {
      APInt val = input.getAPInt();
      if (val.getBitWidth() < width)
        val = val.zext(width);
      else if (val.getBitWidth() > width)
        val = val.trunc(width);
      setValue(procId, ptrToIntOp.getResult(), InterpretedValue(val));
    }
    return success();
  }

  // LLVM bitcast
  if (auto bitcastOp = dyn_cast<LLVM::BitcastOp>(op)) {
    InterpretedValue input = getValue(procId, bitcastOp.getArg());
    setValue(procId, bitcastOp.getResult(), input);
    return success();
  }

  // LLVM trunc
  if (auto truncOp = dyn_cast<LLVM::TruncOp>(op)) {
    InterpretedValue input = getValue(procId, truncOp.getArg());
    unsigned width = getTypeWidth(truncOp.getType());
    if (input.isX()) {
      setValue(procId, truncOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, truncOp.getResult(),
               InterpretedValue(input.getAPInt().trunc(width)));
    }
    return success();
  }

  // LLVM zext
  if (auto zextOp = dyn_cast<LLVM::ZExtOp>(op)) {
    InterpretedValue input = getValue(procId, zextOp.getArg());
    unsigned width = getTypeWidth(zextOp.getType());
    if (input.isX()) {
      setValue(procId, zextOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, zextOp.getResult(),
               InterpretedValue(input.getAPInt().zext(width)));
    }
    return success();
  }

  // LLVM sext
  if (auto sextOp = dyn_cast<LLVM::SExtOp>(op)) {
    InterpretedValue input = getValue(procId, sextOp.getArg());
    unsigned width = getTypeWidth(sextOp.getType());
    if (input.isX()) {
      setValue(procId, sextOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, sextOp.getResult(),
               InterpretedValue(input.getAPInt().sext(width)));
    }
    return success();
  }

  // LLVM add
  if (auto addOp = dyn_cast<LLVM::AddOp>(op)) {
    InterpretedValue lhs = getValue(procId, addOp.getLhs());
    InterpretedValue rhs = getValue(procId, addOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, addOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(addOp.getType())));
    } else {
      setValue(procId, addOp.getResult(),
               InterpretedValue(lhs.getAPInt() + rhs.getAPInt()));
    }
    return success();
  }

  // LLVM sub
  if (auto subOp = dyn_cast<LLVM::SubOp>(op)) {
    InterpretedValue lhs = getValue(procId, subOp.getLhs());
    InterpretedValue rhs = getValue(procId, subOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, subOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(subOp.getType())));
    } else {
      setValue(procId, subOp.getResult(),
               InterpretedValue(lhs.getAPInt() - rhs.getAPInt()));
    }
    return success();
  }

  // LLVM mul
  if (auto mulOp = dyn_cast<LLVM::MulOp>(op)) {
    InterpretedValue lhs = getValue(procId, mulOp.getLhs());
    InterpretedValue rhs = getValue(procId, mulOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, mulOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(mulOp.getType())));
    } else {
      setValue(procId, mulOp.getResult(),
               InterpretedValue(lhs.getAPInt() * rhs.getAPInt()));
    }
    return success();
  }

  // LLVM icmp
  if (auto icmpOp = dyn_cast<LLVM::ICmpOp>(op)) {
    InterpretedValue lhs = getValue(procId, icmpOp.getLhs());
    InterpretedValue rhs = getValue(procId, icmpOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, icmpOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }
    bool result = false;
    const APInt &lhsVal = lhs.getAPInt();
    const APInt &rhsVal = rhs.getAPInt();
    switch (icmpOp.getPredicate()) {
    case LLVM::ICmpPredicate::eq:
      result = lhsVal == rhsVal;
      break;
    case LLVM::ICmpPredicate::ne:
      result = lhsVal != rhsVal;
      break;
    case LLVM::ICmpPredicate::slt:
      result = lhsVal.slt(rhsVal);
      break;
    case LLVM::ICmpPredicate::sle:
      result = lhsVal.sle(rhsVal);
      break;
    case LLVM::ICmpPredicate::sgt:
      result = lhsVal.sgt(rhsVal);
      break;
    case LLVM::ICmpPredicate::sge:
      result = lhsVal.sge(rhsVal);
      break;
    case LLVM::ICmpPredicate::ult:
      result = lhsVal.ult(rhsVal);
      break;
    case LLVM::ICmpPredicate::ule:
      result = lhsVal.ule(rhsVal);
      break;
    case LLVM::ICmpPredicate::ugt:
      result = lhsVal.ugt(rhsVal);
      break;
    case LLVM::ICmpPredicate::uge:
      result = lhsVal.uge(rhsVal);
      break;
    }
    setValue(procId, icmpOp.getResult(), InterpretedValue(result ? 1 : 0, 1));
    return success();
  }

  // LLVM and
  if (auto andOp = dyn_cast<LLVM::AndOp>(op)) {
    InterpretedValue lhs = getValue(procId, andOp.getLhs());
    InterpretedValue rhs = getValue(procId, andOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, andOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(andOp.getType())));
    } else {
      setValue(procId, andOp.getResult(),
               InterpretedValue(lhs.getAPInt() & rhs.getAPInt()));
    }
    return success();
  }

  // LLVM or
  if (auto orOp = dyn_cast<LLVM::OrOp>(op)) {
    InterpretedValue lhs = getValue(procId, orOp.getLhs());
    InterpretedValue rhs = getValue(procId, orOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, orOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(orOp.getType())));
    } else {
      setValue(procId, orOp.getResult(),
               InterpretedValue(lhs.getAPInt() | rhs.getAPInt()));
    }
    return success();
  }

  // LLVM xor
  if (auto xorOp = dyn_cast<LLVM::XOrOp>(op)) {
    InterpretedValue lhs = getValue(procId, xorOp.getLhs());
    InterpretedValue rhs = getValue(procId, xorOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, xorOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(xorOp.getType())));
    } else {
      setValue(procId, xorOp.getResult(),
               InterpretedValue(lhs.getAPInt() ^ rhs.getAPInt()));
    }
    return success();
  }

  // LLVM shl
  if (auto shlOp = dyn_cast<LLVM::ShlOp>(op)) {
    InterpretedValue lhs = getValue(procId, shlOp.getLhs());
    InterpretedValue rhs = getValue(procId, shlOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, shlOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(shlOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, shlOp.getResult(),
               InterpretedValue(lhs.getAPInt().shl(shift)));
    }
    return success();
  }

  // LLVM lshr
  if (auto lshrOp = dyn_cast<LLVM::LShrOp>(op)) {
    InterpretedValue lhs = getValue(procId, lshrOp.getLhs());
    InterpretedValue rhs = getValue(procId, lshrOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, lshrOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(lshrOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, lshrOp.getResult(),
               InterpretedValue(lhs.getAPInt().lshr(shift)));
    }
    return success();
  }

  // LLVM ashr
  if (auto ashrOp = dyn_cast<LLVM::AShrOp>(op)) {
    InterpretedValue lhs = getValue(procId, ashrOp.getLhs());
    InterpretedValue rhs = getValue(procId, ashrOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, ashrOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(ashrOp.getType())));
    } else {
      uint64_t shift = rhs.getAPInt().getLimitedValue();
      setValue(procId, ashrOp.getResult(),
               InterpretedValue(lhs.getAPInt().ashr(shift)));
    }
    return success();
  }

  // LLVM select - conditional value selection
  if (auto selectOp = dyn_cast<LLVM::SelectOp>(op)) {
    InterpretedValue cond = getValue(procId, selectOp.getCondition());
    unsigned width = getTypeWidth(selectOp.getType());
    if (cond.isX()) {
      // X condition propagates to X result
      setValue(procId, selectOp.getResult(), InterpretedValue::makeX(width));
    } else {
      bool condVal = cond.getUInt64() != 0;
      InterpretedValue selected =
          condVal ? getValue(procId, selectOp.getTrueValue())
                  : getValue(procId, selectOp.getFalseValue());
      setValue(procId, selectOp.getResult(), selected);
    }
    return success();
  }

  // LLVM freeze - freeze undefined values to a deterministic value
  if (auto freezeOp = dyn_cast<LLVM::FreezeOp>(op)) {
    InterpretedValue input = getValue(procId, freezeOp.getVal());
    unsigned width = getTypeWidth(freezeOp.getType());
    if (input.isX()) {
      // Freeze X to 0 (a deterministic but arbitrary value)
      setValue(procId, freezeOp.getResult(), InterpretedValue(0, width));
    } else {
      // Pass through known values
      setValue(procId, freezeOp.getResult(), input);
    }
    return success();
  }

  // LLVM sdiv - signed integer division
  if (auto sdivOp = dyn_cast<LLVM::SDivOp>(op)) {
    InterpretedValue lhs = getValue(procId, sdivOp.getLhs());
    InterpretedValue rhs = getValue(procId, sdivOp.getRhs());
    unsigned width = getTypeWidth(sdivOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, sdivOp.getResult(), InterpretedValue::makeX(width));
    } else if (rhs.getAPInt().isZero()) {
      // Division by zero returns X
      setValue(procId, sdivOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, sdivOp.getResult(),
               InterpretedValue(lhs.getAPInt().sdiv(rhs.getAPInt())));
    }
    return success();
  }

  // LLVM udiv - unsigned integer division
  if (auto udivOp = dyn_cast<LLVM::UDivOp>(op)) {
    InterpretedValue lhs = getValue(procId, udivOp.getLhs());
    InterpretedValue rhs = getValue(procId, udivOp.getRhs());
    unsigned width = getTypeWidth(udivOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, udivOp.getResult(), InterpretedValue::makeX(width));
    } else if (rhs.getAPInt().isZero()) {
      // Division by zero returns X
      setValue(procId, udivOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, udivOp.getResult(),
               InterpretedValue(lhs.getAPInt().udiv(rhs.getAPInt())));
    }
    return success();
  }

  // LLVM srem - signed integer remainder
  if (auto sremOp = dyn_cast<LLVM::SRemOp>(op)) {
    InterpretedValue lhs = getValue(procId, sremOp.getLhs());
    InterpretedValue rhs = getValue(procId, sremOp.getRhs());
    unsigned width = getTypeWidth(sremOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, sremOp.getResult(), InterpretedValue::makeX(width));
    } else if (rhs.getAPInt().isZero()) {
      // Remainder by zero returns X
      setValue(procId, sremOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, sremOp.getResult(),
               InterpretedValue(lhs.getAPInt().srem(rhs.getAPInt())));
    }
    return success();
  }

  // LLVM urem - unsigned integer remainder
  if (auto uremOp = dyn_cast<LLVM::URemOp>(op)) {
    InterpretedValue lhs = getValue(procId, uremOp.getLhs());
    InterpretedValue rhs = getValue(procId, uremOp.getRhs());
    unsigned width = getTypeWidth(uremOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, uremOp.getResult(), InterpretedValue::makeX(width));
    } else if (rhs.getAPInt().isZero()) {
      // Remainder by zero returns X
      setValue(procId, uremOp.getResult(), InterpretedValue::makeX(width));
    } else {
      setValue(procId, uremOp.getResult(),
               InterpretedValue(lhs.getAPInt().urem(rhs.getAPInt())));
    }
    return success();
  }

  // LLVM fadd - floating point addition
  if (auto faddOp = dyn_cast<LLVM::FAddOp>(op)) {
    InterpretedValue lhs = getValue(procId, faddOp.getLhs());
    InterpretedValue rhs = getValue(procId, faddOp.getRhs());
    unsigned width = getTypeWidth(faddOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, faddOp.getResult(), InterpretedValue::makeX(width));
    } else {
      // Convert APInt to floating point, perform operation, convert back
      APInt lhsInt = lhs.getAPInt();
      APInt rhsInt = rhs.getAPInt();
      if (width == 32) {
        uint32_t lhsBits = static_cast<uint32_t>(lhsInt.getZExtValue());
        uint32_t rhsBits = static_cast<uint32_t>(rhsInt.getZExtValue());
        float lhsFloat = llvm::bit_cast<float>(lhsBits);
        float rhsFloat = llvm::bit_cast<float>(rhsBits);
        float result = lhsFloat + rhsFloat;
        setValue(procId, faddOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint32_t>(result), 32));
      } else if (width == 64) {
        double lhsDouble = llvm::bit_cast<double>(lhsInt.getZExtValue());
        double rhsDouble = llvm::bit_cast<double>(rhsInt.getZExtValue());
        double result = lhsDouble + rhsDouble;
        setValue(procId, faddOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint64_t>(result), 64));
      } else {
        setValue(procId, faddOp.getResult(), InterpretedValue::makeX(width));
      }
    }
    return success();
  }

  // LLVM fsub - floating point subtraction
  if (auto fsubOp = dyn_cast<LLVM::FSubOp>(op)) {
    InterpretedValue lhs = getValue(procId, fsubOp.getLhs());
    InterpretedValue rhs = getValue(procId, fsubOp.getRhs());
    unsigned width = getTypeWidth(fsubOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, fsubOp.getResult(), InterpretedValue::makeX(width));
    } else {
      APInt lhsInt = lhs.getAPInt();
      APInt rhsInt = rhs.getAPInt();
      if (width == 32) {
        uint32_t lhsBits = static_cast<uint32_t>(lhsInt.getZExtValue());
        uint32_t rhsBits = static_cast<uint32_t>(rhsInt.getZExtValue());
        float lhsFloat = llvm::bit_cast<float>(lhsBits);
        float rhsFloat = llvm::bit_cast<float>(rhsBits);
        float result = lhsFloat - rhsFloat;
        setValue(procId, fsubOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint32_t>(result), 32));
      } else if (width == 64) {
        double lhsDouble = llvm::bit_cast<double>(lhsInt.getZExtValue());
        double rhsDouble = llvm::bit_cast<double>(rhsInt.getZExtValue());
        double result = lhsDouble - rhsDouble;
        setValue(procId, fsubOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint64_t>(result), 64));
      } else {
        setValue(procId, fsubOp.getResult(), InterpretedValue::makeX(width));
      }
    }
    return success();
  }

  // LLVM fmul - floating point multiplication
  if (auto fmulOp = dyn_cast<LLVM::FMulOp>(op)) {
    InterpretedValue lhs = getValue(procId, fmulOp.getLhs());
    InterpretedValue rhs = getValue(procId, fmulOp.getRhs());
    unsigned width = getTypeWidth(fmulOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, fmulOp.getResult(), InterpretedValue::makeX(width));
    } else {
      APInt lhsInt = lhs.getAPInt();
      APInt rhsInt = rhs.getAPInt();
      if (width == 32) {
        uint32_t lhsBits = static_cast<uint32_t>(lhsInt.getZExtValue());
        uint32_t rhsBits = static_cast<uint32_t>(rhsInt.getZExtValue());
        float lhsFloat = llvm::bit_cast<float>(lhsBits);
        float rhsFloat = llvm::bit_cast<float>(rhsBits);
        float result = lhsFloat * rhsFloat;
        setValue(procId, fmulOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint32_t>(result), 32));
      } else if (width == 64) {
        double lhsDouble = llvm::bit_cast<double>(lhsInt.getZExtValue());
        double rhsDouble = llvm::bit_cast<double>(rhsInt.getZExtValue());
        double result = lhsDouble * rhsDouble;
        setValue(procId, fmulOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint64_t>(result), 64));
      } else {
        setValue(procId, fmulOp.getResult(), InterpretedValue::makeX(width));
      }
    }
    return success();
  }

  // LLVM fdiv - floating point division
  if (auto fdivOp = dyn_cast<LLVM::FDivOp>(op)) {
    InterpretedValue lhs = getValue(procId, fdivOp.getLhs());
    InterpretedValue rhs = getValue(procId, fdivOp.getRhs());
    unsigned width = getTypeWidth(fdivOp.getType());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, fdivOp.getResult(), InterpretedValue::makeX(width));
    } else {
      APInt lhsInt = lhs.getAPInt();
      APInt rhsInt = rhs.getAPInt();
      if (width == 32) {
        uint32_t lhsBits = static_cast<uint32_t>(lhsInt.getZExtValue());
        uint32_t rhsBits = static_cast<uint32_t>(rhsInt.getZExtValue());
        float lhsFloat = llvm::bit_cast<float>(lhsBits);
        float rhsFloat = llvm::bit_cast<float>(rhsBits);
        float result = lhsFloat / rhsFloat;
        setValue(procId, fdivOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint32_t>(result), 32));
      } else if (width == 64) {
        double lhsDouble = llvm::bit_cast<double>(lhsInt.getZExtValue());
        double rhsDouble = llvm::bit_cast<double>(rhsInt.getZExtValue());
        double result = lhsDouble / rhsDouble;
        setValue(procId, fdivOp.getResult(),
                 InterpretedValue(llvm::bit_cast<uint64_t>(result), 64));
      } else {
        setValue(procId, fdivOp.getResult(), InterpretedValue::makeX(width));
      }
    }
    return success();
  }

  // LLVM fcmp - floating point comparison
  if (auto fcmpOp = dyn_cast<LLVM::FCmpOp>(op)) {
    InterpretedValue lhs = getValue(procId, fcmpOp.getLhs());
    InterpretedValue rhs = getValue(procId, fcmpOp.getRhs());
    if (lhs.isX() || rhs.isX()) {
      setValue(procId, fcmpOp.getResult(), InterpretedValue::makeX(1));
      return success();
    }
    unsigned width = getTypeWidth(fcmpOp.getLhs().getType());
    bool result = false;

    // Convert to appropriate floating point type and compare
    if (width == 32) {
      uint32_t lhsBits = static_cast<uint32_t>(lhs.getAPInt().getZExtValue());
      uint32_t rhsBits = static_cast<uint32_t>(rhs.getAPInt().getZExtValue());
      float lhsFloat = llvm::bit_cast<float>(lhsBits);
      float rhsFloat = llvm::bit_cast<float>(rhsBits);
      switch (fcmpOp.getPredicate()) {
      case LLVM::FCmpPredicate::_false: result = false; break;
      case LLVM::FCmpPredicate::oeq: result = lhsFloat == rhsFloat; break;
      case LLVM::FCmpPredicate::ogt: result = lhsFloat > rhsFloat; break;
      case LLVM::FCmpPredicate::oge: result = lhsFloat >= rhsFloat; break;
      case LLVM::FCmpPredicate::olt: result = lhsFloat < rhsFloat; break;
      case LLVM::FCmpPredicate::ole: result = lhsFloat <= rhsFloat; break;
      case LLVM::FCmpPredicate::one: result = lhsFloat != rhsFloat && !std::isnan(lhsFloat) && !std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::ord: result = !std::isnan(lhsFloat) && !std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::ueq: result = lhsFloat == rhsFloat || std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::ugt: result = lhsFloat > rhsFloat || std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::uge: result = lhsFloat >= rhsFloat || std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::ult: result = lhsFloat < rhsFloat || std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::ule: result = lhsFloat <= rhsFloat || std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::une: result = lhsFloat != rhsFloat; break;
      case LLVM::FCmpPredicate::uno: result = std::isnan(lhsFloat) || std::isnan(rhsFloat); break;
      case LLVM::FCmpPredicate::_true: result = true; break;
      }
    } else if (width == 64) {
      double lhsDouble = llvm::bit_cast<double>(lhs.getAPInt().getZExtValue());
      double rhsDouble = llvm::bit_cast<double>(rhs.getAPInt().getZExtValue());
      switch (fcmpOp.getPredicate()) {
      case LLVM::FCmpPredicate::_false: result = false; break;
      case LLVM::FCmpPredicate::oeq: result = lhsDouble == rhsDouble; break;
      case LLVM::FCmpPredicate::ogt: result = lhsDouble > rhsDouble; break;
      case LLVM::FCmpPredicate::oge: result = lhsDouble >= rhsDouble; break;
      case LLVM::FCmpPredicate::olt: result = lhsDouble < rhsDouble; break;
      case LLVM::FCmpPredicate::ole: result = lhsDouble <= rhsDouble; break;
      case LLVM::FCmpPredicate::one: result = lhsDouble != rhsDouble && !std::isnan(lhsDouble) && !std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::ord: result = !std::isnan(lhsDouble) && !std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::ueq: result = lhsDouble == rhsDouble || std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::ugt: result = lhsDouble > rhsDouble || std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::uge: result = lhsDouble >= rhsDouble || std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::ult: result = lhsDouble < rhsDouble || std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::ule: result = lhsDouble <= rhsDouble || std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::une: result = lhsDouble != rhsDouble; break;
      case LLVM::FCmpPredicate::uno: result = std::isnan(lhsDouble) || std::isnan(rhsDouble); break;
      case LLVM::FCmpPredicate::_true: result = true; break;
      }
    }
    setValue(procId, fcmpOp.getResult(), InterpretedValue(result ? 1 : 0, 1));
    return success();
  }

  // LLVM extractvalue - extract a value from an aggregate (struct/array)
  if (auto extractValueOp = dyn_cast<LLVM::ExtractValueOp>(op)) {
    InterpretedValue container = getValue(procId, extractValueOp.getContainer());
    unsigned resultWidth = getTypeWidth(extractValueOp.getType());
    if (container.isX()) {
      setValue(procId, extractValueOp.getResult(),
               InterpretedValue::makeX(resultWidth));
      return success();
    }

    // Calculate the bit offset for the indexed position
    // LLVM aggregates are laid out from low bits to high bits (opposite of HW)
    Type currentType = extractValueOp.getContainer().getType();
    unsigned bitOffset = 0;
    for (int64_t idx : extractValueOp.getPosition()) {
      if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        // For struct, accumulate offsets of preceding fields
        auto body = structType.getBody();
        for (int64_t i = 0; i < idx; ++i) {
          bitOffset += getTypeWidth(body[i]);
        }
        currentType = body[idx];
      } else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
        // For array, compute offset based on element size
        unsigned elemWidth = getTypeWidth(arrayType.getElementType());
        bitOffset += elemWidth * idx;
        currentType = arrayType.getElementType();
      }
    }

    APInt extractedValue =
        container.getAPInt().extractBits(resultWidth, bitOffset);
    setValue(procId, extractValueOp.getResult(),
             InterpretedValue(extractedValue));
    return success();
  }

  // LLVM insertvalue - insert a value into an aggregate (struct/array)
  if (auto insertValueOp = dyn_cast<LLVM::InsertValueOp>(op)) {
    InterpretedValue container = getValue(procId, insertValueOp.getContainer());
    InterpretedValue value = getValue(procId, insertValueOp.getValue());
    unsigned totalWidth = getTypeWidth(insertValueOp.getType());

    // If only the value being inserted is X, propagate X
    if (value.isX()) {
      setValue(procId, insertValueOp.getResult(),
               InterpretedValue::makeX(totalWidth));
      return success();
    }

    // If the container is X (e.g., from llvm.mlir.undef), treat it as zeros
    // to allow building up structs incrementally. This is the common pattern
    // for constructing structs: start with undef, then insertvalue fields.
    if (container.isX()) {
      container = InterpretedValue(APInt::getZero(totalWidth));
    }

    // Calculate the bit offset for the indexed position
    // LLVM aggregates are laid out from low bits to high bits (opposite of HW)
    Type currentType = insertValueOp.getContainer().getType();
    unsigned bitOffset = 0;
    unsigned fieldWidth = 0;
    for (int64_t idx : insertValueOp.getPosition()) {
      if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        // For struct, accumulate offsets of preceding fields
        auto body = structType.getBody();
        for (int64_t i = 0; i < idx; ++i) {
          bitOffset += getTypeWidth(body[i]);
        }
        currentType = body[idx];
      } else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
        // For array, compute offset based on element size
        unsigned elemWidth = getTypeWidth(arrayType.getElementType());
        bitOffset += elemWidth * idx;
        currentType = arrayType.getElementType();
      }
    }
    fieldWidth = getTypeWidth(currentType);

    APInt result = container.getAPInt();
    APInt fieldValue = value.getAPInt();
    if (fieldValue.getBitWidth() < fieldWidth)
      fieldValue = fieldValue.zext(fieldWidth);
    else if (fieldValue.getBitWidth() > fieldWidth)
      fieldValue = fieldValue.trunc(fieldWidth);
    result.insertBits(fieldValue, bitOffset);

    setValue(procId, insertValueOp.getResult(), InterpretedValue(result));
    return success();
  }

  // Handle builtin.unrealized_conversion_cast - propagate values through
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    // For casts, propagate the input values to the output values
    if (castOp.getNumOperands() == castOp.getNumResults()) {
      // Simple 1:1 mapping
      for (auto [input, output] : llvm::zip(castOp.getInputs(), castOp.getOutputs())) {
        // For !llvm.ptr to !llhd.ref casts, propagate the pointer address.
        // The probe/drive handlers use SSA tracing (getDefiningOp) first,
        // but the address value is needed as a fallback when the ref is
        // passed through function arguments (where SSA tracing fails).
        Type inputType = input.getType();
        Type outputType = output.getType();
        if (isa<LLVM::LLVMPointerType>(inputType) &&
            isa<llhd::RefType>(outputType)) {
          InterpretedValue ptrVal = getValue(procId, input);
          setValue(procId, output, ptrVal);
          LLVM_DEBUG(llvm::dbgs() << "  builtin.unrealized_conversion_cast: "
                                  << "ptr->ref cast, propagating address "
                                  << (ptrVal.isX() ? "X" : std::to_string(ptrVal.getUInt64()))
                                  << "\n");
          continue;
        }

        InterpretedValue val = getValue(procId, input);

        // Handle layout conversion between LLVM struct and HW struct types
        Type inputType2 = input.getType();
        Type outputType2 = output.getType();
        if (!val.isX()) {
          if ((isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(inputType2)) &&
              (isa<hw::StructType, hw::ArrayType>(outputType2))) {
            // LLVM -> HW layout conversion
            APInt converted = convertLLVMToHWLayout(val.getAPInt(), inputType2, outputType2);
            val = InterpretedValue(converted);
          } else if ((isa<hw::StructType, hw::ArrayType>(inputType2)) &&
                     (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(outputType2))) {
            // HW -> LLVM layout conversion
            APInt converted = convertHWToLLVMLayout(val.getAPInt(), inputType2, outputType2);
            val = InterpretedValue(converted);
          }
        }

        // Adjust width if needed
        unsigned outputWidth = getTypeWidth(output.getType());
        if (!val.isX() && val.getWidth() != outputWidth) {
          if (outputWidth > 64) {
            APInt apVal = val.getAPInt();
            if (apVal.getBitWidth() < outputWidth) {
              apVal = apVal.zext(outputWidth);
            } else if (apVal.getBitWidth() > outputWidth) {
              apVal = apVal.trunc(outputWidth);
            }
            val = InterpretedValue(apVal);
          } else {
            // Mask the value to fit in outputWidth bits to avoid APInt assertion
            uint64_t maskedVal = val.getUInt64();
            if (outputWidth < 64)
              maskedVal &= ((1ULL << outputWidth) - 1);
            val = InterpretedValue(maskedVal, outputWidth);
          }
        }
        setValue(procId, output, val);
      }
    } else if (castOp.getNumOperands() == 1 && castOp.getNumResults() > 0) {
      // Single input to multiple outputs (common for function types)
      InterpretedValue val = getValue(procId, castOp.getInputs()[0]);
      for (Value output : castOp.getOutputs()) {
        unsigned outputWidth = getTypeWidth(output.getType());
        // Mask the value to fit in outputWidth bits to avoid APInt assertion
        uint64_t maskedVal = val.isX() ? 0 : val.getUInt64();
        if (outputWidth < 64)
          maskedVal &= ((1ULL << outputWidth) - 1);
        setValue(procId, output, InterpretedValue(maskedVal, outputWidth));
      }
    } else {
      // Just propagate input values for non-standard patterns
      unsigned numToCopy = std::min(castOp.getNumOperands(), castOp.getNumResults());
      for (unsigned i = 0; i < numToCopy; ++i) {
        setValue(procId, castOp.getResult(i), getValue(procId, castOp.getOperand(i)));
      }
      // Set remaining results to X
      for (unsigned i = numToCopy; i < castOp.getNumResults(); ++i) {
        setValue(procId, castOp.getResult(i),
                 InterpretedValue::makeX(getTypeWidth(castOp.getResult(i).getType())));
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "  builtin.unrealized_conversion_cast: propagated "
                            << castOp.getNumOperands() << " inputs to "
                            << castOp.getNumResults() << " outputs\n");
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
  Value signal = probeOp.getSignal();
  while (auto muxOp = signal.getDefiningOp<comb::MuxOp>()) {
    InterpretedValue cond = getValue(procId, muxOp.getCond());
    if (cond.isX()) {
      setValue(procId, probeOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(probeOp.getResult().getType())));
      return success();
    }
    signal = cond.getUInt64() != 0 ? muxOp.getTrueValue() : muxOp.getFalseValue();
  }

  // Handle arith.select on ref types (e.g., !llhd.ref<!hw.struct<...>>)
  while (auto selectOp = signal.getDefiningOp<arith::SelectOp>()) {
    InterpretedValue cond = getValue(procId, selectOp.getCondition());
    if (cond.isX()) {
      setValue(procId, probeOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(probeOp.getResult().getType())));
      return success();
    }
    signal = cond.getUInt64() != 0 ? selectOp.getTrueValue() : selectOp.getFalseValue();
  }

  // Get the signal ID for the probed signal
  SignalId sigId = resolveSignalId(signal);
  if (sigId == 0) {
    // Check if this is a global variable access via UnrealizedConversionCastOp
    // This happens when static class properties are accessed - they're stored
    // in LLVM globals, not LLHD signals.
    if (auto castOp =
            signal.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1) {
        Value input = castOp.getInputs()[0];
        if (auto addrOfOp = input.getDefiningOp<LLVM::AddressOfOp>()) {
          StringRef globalName = addrOfOp.getGlobalName();
          LLVM_DEBUG(llvm::dbgs()
                     << "  Probe of global variable: " << globalName << "\n");

          // Read from global memory block
          auto blockIt = globalMemoryBlocks.find(globalName);
          if (blockIt != globalMemoryBlocks.end()) {
            MemoryBlock &block = blockIt->second;
            // Read the pointer value from global memory
            uint64_t ptrValue = 0;
            bool hasUnknown = !block.initialized;

            if (!hasUnknown) {
              // Read 8 bytes (pointer size) from the memory block
              unsigned readSize = std::min(8u, static_cast<unsigned>(block.size));
              for (unsigned i = 0; i < readSize; ++i) {
                ptrValue |= (static_cast<uint64_t>(block.data[i]) << (i * 8));
              }
            }

            InterpretedValue val;
            if (hasUnknown) {
              val = InterpretedValue::makeX(64);
            } else {
              val = InterpretedValue(ptrValue, 64);
            }
            setValue(procId, probeOp.getResult(), val);
            LLVM_DEBUG(llvm::dbgs()
                       << "  Read global " << globalName << " = "
                       << (hasUnknown ? "X" : std::to_string(ptrValue)) << "\n");
            return success();
          }
        }
        // Handle GEP-based memory access (e.g., class member access)
        // This happens when class properties are accessed - the property ref
        // creates a GEP to the field, which is then cast to !llhd.ref.
        if (auto gepOp = input.getDefiningOp<LLVM::GEPOp>()) {
          // Get the pointer value computed by the GEP
          InterpretedValue ptrVal = getValue(procId, gepOp.getResult());
          unsigned width = getTypeWidth(probeOp.getResult().getType());
          unsigned loadSize = (width + 7) / 8;

          if (ptrVal.isX()) {
            // Uninitialized pointer - return X
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe of GEP pointer: X (uninitialized)\n");
            return success();
          }

          // Find the memory block - first try local, then global, then malloc
          MemoryBlock *block = findMemoryBlock(procId, gepOp);
          uint64_t offset = 0;

          if (block) {
            // Calculate offset for local memory
            InterpretedValue baseVal = getValue(procId, gepOp.getBase());
            if (!baseVal.isX()) {
              offset = ptrVal.getUInt64() - baseVal.getUInt64();
            }
          } else {
            // Check global and malloc memory by address
            uint64_t addr = ptrVal.getUInt64();

            // Check globals
            for (auto &entry : globalAddresses) {
              StringRef globalName = entry.first();
              uint64_t globalBaseAddr = entry.second;
              auto blockIt = globalMemoryBlocks.find(globalName);
              if (blockIt != globalMemoryBlocks.end()) {
                uint64_t globalSize = blockIt->second.size;
                if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
                  block = &blockIt->second;
                  offset = addr - globalBaseAddr;
                  LLVM_DEBUG(llvm::dbgs() << "  Probe: found global '" << globalName
                                          << "' at offset " << offset << "\n");
                  break;
                }
              }
            }

            // Check malloc blocks
            if (!block) {
              for (auto &entry : mallocBlocks) {
                uint64_t mallocBaseAddr = entry.first;
                uint64_t mallocSize = entry.second.size;
                if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
                  block = &entry.second;
                  offset = addr - mallocBaseAddr;
                  LLVM_DEBUG(llvm::dbgs() << "  Probe: found malloc block at 0x"
                                          << llvm::format_hex(mallocBaseAddr, 16)
                                          << " offset " << offset << "\n");
                  break;
                }
              }
            }
          }

          if (!block) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe of GEP pointer 0x"
                       << llvm::format_hex(ptrVal.getUInt64(), 0)
                       << " failed - memory not found\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          if (offset + loadSize > block->size) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe: out of bounds (offset=" << offset
                       << " size=" << loadSize << " block=" << block->size << ")\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          if (!block->initialized) {
            LLVM_DEBUG(llvm::dbgs() << "  Probe: reading uninitialized memory\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          // Read the value from memory
          uint64_t value = 0;
          for (unsigned i = 0; i < loadSize; ++i) {
            value |= (static_cast<uint64_t>(block->data[offset + i]) << (i * 8));
          }
          // Mask to the exact bit width
          if (width < 64)
            value &= (1ULL << width) - 1;

          setValue(procId, probeOp.getResult(), InterpretedValue(value, width));
          LLVM_DEBUG(llvm::dbgs()
                     << "  Probe of GEP pointer 0x"
                     << llvm::format_hex(ptrVal.getUInt64(), 0)
                     << " = " << value << " (width=" << width << ")\n");
          return success();
        }
        // Handle AllocaOp - local variables in functions backed by llvm.alloca
        // Pattern: %alloca = llvm.alloca -> unrealized_cast to !llhd.ref -> llhd.prb
        // This happens when local variables inside class methods are cast to refs.
        if (auto allocaOp = input.getDefiningOp<LLVM::AllocaOp>()) {
          unsigned width = getTypeWidth(probeOp.getResult().getType());
          unsigned loadSize = (width + 7) / 8;

          // Find the memory block for this alloca
          MemoryBlock *block = findMemoryBlock(procId, allocaOp);
          if (!block) {
            // Try finding by alloca result in process-local memory blocks
            auto &state = processStates[procId];
            auto it = state.memoryBlocks.find(allocaOp.getResult());
            if (it != state.memoryBlocks.end()) {
              block = &it->second;
              LLVM_DEBUG(llvm::dbgs() << "  Probe: found local alloca memory\n");
            }
          }

          if (!block) {
            LLVM_DEBUG(llvm::dbgs() << "  Probe of alloca failed - memory not found\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          if (loadSize > block->size) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe: out of bounds (size=" << loadSize
                       << " block=" << block->size << ")\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          if (!block->initialized) {
            LLVM_DEBUG(llvm::dbgs() << "  Probe: reading uninitialized alloca memory\n");
            setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
            return success();
          }

          // Read the value from memory (little-endian byte order)
          APInt memValue = APInt::getZero(width);
          for (unsigned i = 0; i < loadSize && i * 8 < width; ++i) {
            unsigned bitsToInsert = std::min(8u, width - i * 8);
            APInt byteVal(bitsToInsert,
                          block->data[i] & ((1u << bitsToInsert) - 1));
            memValue.insertBits(byteVal, i * 8);
          }

          // Check if we need to convert from LLVM layout to HW layout.
          // LLVM struct fields are at low-to-high bits, while HW struct
          // fields are at high-to-low bits.
          Type resultType = probeOp.getResult().getType();
          Type allocaElemType = allocaOp.getElemType();
          if (isa<hw::StructType, hw::ArrayType>(resultType) &&
              isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(
                  allocaElemType)) {
            // Recursively convert from LLVM layout to HW layout so that
            // nested structs and arrays are also properly reordered.
            APInt hwValue =
                convertLLVMToHWLayout(memValue, allocaElemType, resultType);
            setValue(procId, probeOp.getResult(), InterpretedValue(hwValue));
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe of alloca (LLVM->HW layout conversion) = 0x"
                       << llvm::format_hex(hwValue.getZExtValue(), 0)
                       << " (width=" << width << ")\n");
            return success();
          }

          // No layout conversion needed - use value as-is
          setValue(procId, probeOp.getResult(), InterpretedValue(memValue));
          LLVM_DEBUG(llvm::dbgs()
                     << "  Probe of alloca = 0x"
                     << llvm::format_hex(memValue.getZExtValue(), 0)
                     << " (width=" << width << ")\n");
          return success();
        }
      }
    }

    // Handle llhd.sig.struct_extract - probe a field within a struct signal.
    // We need to read the parent signal and extract the relevant field bits.
    if (auto sigExtractOp = signal.getDefiningOp<llhd::SigStructExtractOp>()) {
      // Find the parent signal ID by tracing through nested extracts
      Value parentSignal = sigExtractOp.getInput();
      SignalId parentSigId = getSignalId(parentSignal);

      // Handle nested struct extracts by tracing to the root signal
      llvm::SmallVector<llhd::SigStructExtractOp, 4> extractChain;
      extractChain.push_back(sigExtractOp);

      while (parentSigId == 0) {
        if (auto nestedExtract =
                parentSignal.getDefiningOp<llhd::SigStructExtractOp>()) {
          extractChain.push_back(nestedExtract);
          parentSignal = nestedExtract.getInput();
          parentSigId = getSignalId(parentSignal);
        } else {
          break;
        }
      }

      // If still not found, try resolveSignalId which handles more cases.
      if (parentSigId == 0) {
        parentSigId = resolveSignalId(parentSignal);
      }

      // Handle memory-backed !llhd.ref (e.g., from llvm.alloca via
      // unrealized_conversion_cast, or passed as function argument).
      // The runtime value contains the alloca address.
      if (parentSigId == 0) {
        InterpretedValue parentPtrVal = getValue(procId, parentSignal);
        if (!parentPtrVal.isX() && parentPtrVal.getUInt64() != 0) {
          uint64_t addr = parentPtrVal.getUInt64();
          uint64_t blockOffset = 0;
          MemoryBlock *block =
              findMemoryBlockByAddress(addr, procId, &blockOffset);
          if (block) {
            // Compute field bit offset using LLVM layout (low-to-high bits,
            // field 0 at bit 0).
            unsigned bitOffset = 0;
            Type currentType = parentSignal.getType();
            if (auto refType = dyn_cast<llhd::RefType>(currentType))
              currentType = refType.getNestedType();

            for (auto it = extractChain.rbegin(); it != extractChain.rend();
                 ++it) {
              auto extractOp = *it;
              auto structType = cast<hw::StructType>(currentType);
              auto elements = structType.getElements();
              StringRef fieldName = extractOp.getField();
              auto fieldIndexOpt = structType.getFieldIndex(fieldName);
              if (!fieldIndexOpt)
                return failure();
              unsigned fieldIndex = *fieldIndexOpt;
              unsigned fieldOff = 0;
              for (size_t i = 0; i < fieldIndex; ++i)
                fieldOff += getTypeWidth(elements[i].type);
              bitOffset += fieldOff;
              currentType = elements[fieldIndex].type;
            }

            unsigned fieldWidth = getTypeWidth(currentType);
            unsigned parentWidth = getTypeWidth(parentSignal.getType());
            unsigned parentStoreSize = (parentWidth + 7) / 8;

            if (blockOffset + parentStoreSize <= block->size) {
              if (!block->initialized) {
                setValue(procId, probeOp.getResult(),
                         InterpretedValue::makeX(fieldWidth));
              } else {
                // Read the parent struct value from memory
                APInt parentBits = APInt::getZero(parentWidth);
                for (unsigned i = 0; i < parentStoreSize; ++i) {
                  unsigned insertPos = i * 8;
                  unsigned bitsToInsert =
                      std::min(8u, parentWidth - insertPos);
                  if (bitsToInsert > 0 && insertPos < parentWidth) {
                    APInt byteVal(bitsToInsert,
                                  block->data[blockOffset + i] &
                                      ((1u << bitsToInsert) - 1));
                    parentBits.insertBits(byteVal, insertPos);
                  }
                }
                // Extract the field
                APInt fieldBits =
                    parentBits.extractBits(fieldWidth, bitOffset);
                setValue(procId, probeOp.getResult(),
                         InterpretedValue(fieldBits));
              }

              LLVM_DEBUG(llvm::dbgs()
                         << "  Probe struct field from memory-backed ref at "
                            "offset "
                         << bitOffset << " width " << fieldWidth << "\n");
              return success();
            }
          }
        }
      }

      if (parentSigId == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Error: Could not find parent signal for struct extract probe\n");
        return failure();
      }

      // Get the current value of the parent signal
      const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
      InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);

      // Compute the bit offset by walking the extract chain in reverse
      // (from root signal to the target field)
      unsigned bitOffset = 0;
      Type currentType = parentSignal.getType();
      if (auto refType = dyn_cast<llhd::RefType>(currentType))
        currentType = refType.getNestedType();

      for (auto it = extractChain.rbegin(); it != extractChain.rend(); ++it) {
        auto extractOp = *it;
        auto structType = cast<hw::StructType>(currentType);
        auto elements = structType.getElements();
        StringRef fieldName = extractOp.getField();

        auto fieldIndexOpt = structType.getFieldIndex(fieldName);
        if (!fieldIndexOpt) {
          LLVM_DEBUG(llvm::dbgs() << "  Error: Field not found: " << fieldName
                                  << "\n");
          return failure();
        }
        unsigned fieldIndex = *fieldIndexOpt;

        // Fields are laid out from high bits to low bits
        // Calculate offset from the low bit of the current struct
        unsigned fieldOffset = 0;
        for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
          fieldOffset += getTypeWidth(elements[i].type);

        bitOffset += fieldOffset;
        currentType = elements[fieldIndex].type;
      }

      unsigned fieldWidth = getTypeWidth(currentType);

      // Extract the field value from the parent signal
      InterpretedValue fieldVal;
      if (parentVal.isX()) {
        fieldVal = InterpretedValue::makeX(fieldWidth);
      } else {
        APInt parentBits = parentVal.getAPInt();
        APInt fieldBits = parentBits.extractBits(fieldWidth, bitOffset);
        fieldVal = InterpretedValue(fieldBits);
      }

      setValue(procId, probeOp.getResult(), fieldVal);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Probe struct field at offset " << bitOffset
                 << " width " << fieldWidth << " from signal " << parentSigId
                 << " = " << (fieldVal.isX() ? "X" : std::to_string(fieldVal.getUInt64()))
                 << "\n");
      return success();
    }

    // Handle llhd.sig.extract - probe a bit range from a signal/ref.
    // Pattern: %alloca -> cast to !llhd.ref<i32> -> sig.extract -> prb
    if (auto bitExtractOp = signal.getDefiningOp<llhd::SigExtractOp>()) {
      InterpretedValue lowBitVal = getValue(procId, bitExtractOp.getLowBit());
      unsigned totalBitOffset = lowBitVal.isX() ? 0 : lowBitVal.getUInt64();

      // Chase through nested SigExtractOps
      Value parentRef = bitExtractOp.getInput();
      while (auto nestedExtract = parentRef.getDefiningOp<llhd::SigExtractOp>()) {
        InterpretedValue nestedLowBit =
            getValue(procId, nestedExtract.getLowBit());
        totalBitOffset += nestedLowBit.isX() ? 0 : nestedLowBit.getUInt64();
        parentRef = nestedExtract.getInput();
      }

      unsigned resultWidth = getTypeWidth(probeOp.getResult().getType());

      // Check for alloca-backed ref
      if (auto castOp =
              parentRef.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          Value input = castOp.getInputs()[0];
          if (auto allocaOp = input.getDefiningOp<LLVM::AllocaOp>()) {
            unsigned parentWidth = getTypeWidth(parentRef.getType());
            unsigned loadSize = (parentWidth + 7) / 8;

            MemoryBlock *block = findMemoryBlock(procId, allocaOp);
            if (!block) {
              auto &state = processStates[procId];
              auto it = state.memoryBlocks.find(allocaOp.getResult());
              if (it != state.memoryBlocks.end())
                block = &it->second;
            }

            if (!block || !block->initialized) {
              setValue(procId, probeOp.getResult(),
                      InterpretedValue::makeX(resultWidth));
              return success();
            }

            // Read the full value from memory
            APInt fullVal = APInt::getZero(parentWidth);
            for (unsigned i = 0; i < loadSize && i < block->data.size(); ++i) {
              unsigned insertPos = i * 8;
              unsigned bitsToInsert = std::min(8u, parentWidth - insertPos);
              if (bitsToInsert > 0 && insertPos < parentWidth) {
                APInt byteVal(bitsToInsert,
                              block->data[i] & ((1u << bitsToInsert) - 1));
                fullVal.insertBits(byteVal, insertPos);
              }
            }

            // Extract the requested bit range
            APInt extractedVal = fullVal.extractBits(resultWidth, totalBitOffset);
            setValue(procId, probeOp.getResult(), InterpretedValue(extractedVal));
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe sig.extract alloca: bits ["
                       << totalBitOffset << ":" << (totalBitOffset + resultWidth)
                       << "] = " << extractedVal.getZExtValue() << "\n");
            return success();
          }
        }
      }

      // Check for signal-backed ref
      SignalId parentSigId = getSignalId(parentRef);
      if (parentSigId == 0)
        parentSigId = resolveSignalId(parentRef);

      if (parentSigId != 0) {
        const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
        InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);

        if (parentVal.isX()) {
          setValue(procId, probeOp.getResult(),
                  InterpretedValue::makeX(resultWidth));
        } else {
          APInt fullVal = parentVal.getAPInt();
          APInt extractedVal = fullVal.extractBits(resultWidth, totalBitOffset);
          setValue(procId, probeOp.getResult(), InterpretedValue(extractedVal));
        }
        return success();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Error: Could not resolve parent for sig.extract probe\n");
      return failure();
    }

    // Handle llhd.sig.array_get - probe an element within an array signal.
    // We need to read the parent signal and extract the relevant element bits.
    if (auto sigArrayGetOp = signal.getDefiningOp<llhd::SigArrayGetOp>()) {
      Value parentSignal = sigArrayGetOp.getInput();
      SignalId parentSigId = getSignalId(parentSignal);

      // If not found directly, try resolveSignalId
      if (parentSigId == 0) {
        parentSigId = resolveSignalId(parentSignal);
      }

      if (parentSigId == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Error: Could not find parent signal for array get probe\n");
        return failure();
      }

      // Get the index value (may be dynamic)
      InterpretedValue indexVal = getValue(procId, sigArrayGetOp.getIndex());
      if (indexVal.isX()) {
        // X index - return X value
        unsigned width = getTypeWidth(probeOp.getResult().getType());
        setValue(procId, probeOp.getResult(), InterpretedValue::makeX(width));
        LLVM_DEBUG(llvm::dbgs() << "  Probe array element with X index\n");
        return success();
      }
      uint64_t index = indexVal.getUInt64();

      // Get the current value of the parent signal
      const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
      InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);

      // Get array element type and width
      auto arrayType = cast<hw::ArrayType>(unwrapSignalType(parentSignal.getType()));
      Type elementType = arrayType.getElementType();
      unsigned elementWidth = getTypeWidth(elementType);
      size_t numElements = arrayType.getNumElements();

      // Bounds check
      if (index >= numElements) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Array index " << index
                                << " out of bounds (size " << numElements << ")\n");
        setValue(procId, probeOp.getResult(), InterpretedValue::makeX(elementWidth));
        return success();
      }

      // hw::ArrayType layout: element 0 at low bits, element N-1 at high bits
      unsigned bitOffset = index * elementWidth;

      // Extract the element value from the parent signal
      InterpretedValue elementVal;
      if (parentVal.isX()) {
        elementVal = InterpretedValue::makeX(elementWidth);
      } else {
        APInt parentBits = parentVal.getAPInt();
        APInt elementBits = parentBits.extractBits(elementWidth, bitOffset);
        elementVal = InterpretedValue(elementBits);
      }

      setValue(procId, probeOp.getResult(), elementVal);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Probe array element[" << index << "] at offset " << bitOffset
                 << " width " << elementWidth << " from signal " << parentSigId
                 << " = " << (elementVal.isX() ? "X" : std::to_string(elementVal.getUInt64()))
                 << "\n");
      return success();
    }

    // Handle memory-backed ref arguments (e.g., !llhd.ref passed through
    // function calls). The interpreted value of the ref is the memory address.
    // This mirrors the same handling in interpretDrive.
    if (isa<BlockArgument>(signal) && isa<llhd::RefType>(signal.getType())) {
      InterpretedValue addrVal = getValue(procId, signal);
      unsigned width = getTypeWidth(probeOp.getResult().getType());
      unsigned loadSize = (width + 7) / 8;

      if (!addrVal.isX() && addrVal.getUInt64() != 0) {
        uint64_t addr = addrVal.getUInt64();
        // Use the comprehensive memory search that checks process-local
        // allocas, module-level allocas, malloc blocks, and globals.
        uint64_t offset = 0;
        MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);

        if (block && offset + loadSize <= block->size) {
          if (!block->initialized) {
            setValue(procId, probeOp.getResult(),
                     InterpretedValue::makeX(width));
            LLVM_DEBUG(llvm::dbgs()
                       << "  Probe of memory-backed ref at 0x"
                       << llvm::format_hex(addr, 16) << ": X (uninitialized)\n");
            return success();
          }

          APInt memValue = APInt::getZero(width);
          for (unsigned i = 0; i < loadSize && i * 8 < width; ++i) {
            unsigned bitsToInsert = std::min(8u, width - i * 8);
            APInt byteVal(bitsToInsert,
                          block->data[offset + i] &
                              ((1u << bitsToInsert) - 1));
            memValue.insertBits(byteVal, i * 8);
          }

          setValue(procId, probeOp.getResult(), InterpretedValue(memValue));
          LLVM_DEBUG(llvm::dbgs()
                     << "  Probe of memory-backed ref at 0x"
                     << llvm::format_hex(addr, 16) << " offset " << offset
                     << " = 0x"
                     << llvm::format_hex(memValue.getZExtValue(), 0)
                     << " (width=" << width << ")\n");
          return success();
        }
      }

      // Address is X or 0 or memory not found - return X
      setValue(procId, probeOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(probeOp.getResult().getType())));
      LLVM_DEBUG(llvm::dbgs()
                 << "  Probe of memory-backed ref: X (address unavailable)\n");
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Error: Unknown signal in probe\n");
    return failure();
  }

  // First check for pending epsilon drives - this enables blocking assignment
  // semantics where a probe sees the value driven earlier in the same process.
  auto pendingIt = pendingEpsilonDrives.find(sigId);
  if (pendingIt != pendingEpsilonDrives.end()) {
    setValue(procId, probeOp.getResult(), pendingIt->second);
    LLVM_DEBUG(llvm::dbgs() << "  Probed signal " << sigId
                            << " = " << (pendingIt->second.isX() ? "X"
                                         : std::to_string(pendingIt->second.getUInt64()))
                            << " (from pending epsilon drive)\n");
    return success();
  }

  // Get the current signal value from the scheduler
  const SignalValue &sigVal = scheduler.getSignalValue(sigId);

  // Convert to InterpretedValue and store
  InterpretedValue val;
  if (sigVal.isUnknown()) {
    if (auto encoded = getEncodedUnknownForType(probeOp.getResult().getType()))
      val = InterpretedValue(*encoded);
    else
      val = InterpretedValue::makeX(
          getTypeWidth(probeOp.getResult().getType()));
  } else {
    val = InterpretedValue::fromSignalValue(sigVal);
  }
  setValue(procId, probeOp.getResult(), val);

  LLVM_DEBUG(llvm::dbgs() << "  Probed signal " << sigId << " = "
                          << (sigVal.isUnknown() ? "X"
                                                  : std::to_string(sigVal.getValue()))
                          << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretDrive(ProcessId procId,
                                                      llhd::DriveOp driveOp) {
  // Handle arith.select on ref types (e.g., !llhd.ref<!hw.struct<...>>)
  // by evaluating the condition and selecting the appropriate ref.
  Value signal = driveOp.getSignal();
  while (auto selectOp = signal.getDefiningOp<arith::SelectOp>()) {
    InterpretedValue cond = getValue(procId, selectOp.getCondition());
    if (cond.isX()) {
      LLVM_DEBUG(llvm::dbgs() << "  Warning: X condition in arith.select for drive\n");
      return success(); // Cannot determine which signal to drive
    }
    signal = cond.getUInt64() != 0 ? selectOp.getTrueValue() : selectOp.getFalseValue();
  }

  // Get the signal ID
  SignalId sigId = getSignalId(signal);
  if (sigId == 0) {
    // Handle llhd.sig.extract on alloca-backed refs.
    // Pattern: %alloca -> cast to !llhd.ref<i32> -> sig.extract -> !llhd.ref<i1>
    // This is used in uvm_oneway_hash to manipulate individual bits of local vars.
    if (auto bitExtractOp = signal.getDefiningOp<llhd::SigExtractOp>()) {
      // Get the lowBit value
      InterpretedValue lowBitVal = getValue(procId, bitExtractOp.getLowBit());
      unsigned lowBit = lowBitVal.isX() ? 0 : lowBitVal.getUInt64();

      // Trace through to find the underlying alloca
      Value parentRef = bitExtractOp.getInput();
      // Chase through nested SigExtractOps
      unsigned totalBitOffset = lowBit;
      while (auto nestedExtract = parentRef.getDefiningOp<llhd::SigExtractOp>()) {
        InterpretedValue nestedLowBit =
            getValue(procId, nestedExtract.getLowBit());
        totalBitOffset += nestedLowBit.isX() ? 0 : nestedLowBit.getUInt64();
        parentRef = nestedExtract.getInput();
      }

      // Now parentRef should be from UnrealizedConversionCastOp -> AllocaOp
      if (auto castOp =
              parentRef.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          Value input = castOp.getInputs()[0];
          if (auto allocaOp = input.getDefiningOp<LLVM::AllocaOp>()) {
            InterpretedValue driveVal = getValue(procId, driveOp.getValue());

            MemoryBlock *block = findMemoryBlock(procId, allocaOp);
            if (!block) {
              auto &state = processStates[procId];
              auto it = state.memoryBlocks.find(allocaOp.getResult());
              if (it != state.memoryBlocks.end())
                block = &it->second;
            }

            if (!block) {
              LLVM_DEBUG(llvm::dbgs()
                         << "  Drive to sig.extract alloca failed - "
                            "memory not found\n");
              return failure();
            }

            // Read-modify-write: read parent value, modify bits, write back
            unsigned parentWidth = getTypeWidth(parentRef.getType());
            unsigned storeSize = (parentWidth + 7) / 8;

            if (storeSize > block->size) {
              LLVM_DEBUG(llvm::dbgs()
                         << "  Drive to sig.extract: out of bounds\n");
              return failure();
            }

            // Read current value from memory
            APInt currentVal = APInt::getZero(parentWidth);
            for (unsigned i = 0; i < storeSize && i < block->data.size();
                 ++i) {
              unsigned insertPos = i * 8;
              unsigned bitsToInsert = std::min(8u, parentWidth - insertPos);
              if (bitsToInsert > 0 && insertPos < parentWidth) {
                APInt byteVal(bitsToInsert,
                              block->data[i] & ((1u << bitsToInsert) - 1));
                currentVal.insertBits(byteVal, insertPos);
              }
            }

            // Insert the drive value at the bit offset
            unsigned extractWidth = driveVal.getWidth();
            if (!driveVal.isX()) {
              APInt insertVal = driveVal.getAPInt();
              if (insertVal.getBitWidth() != extractWidth)
                insertVal = insertVal.zextOrTrunc(extractWidth);
              currentVal.insertBits(insertVal, totalBitOffset);
            }

            // Write back to memory
            for (unsigned i = 0; i < storeSize; ++i) {
              unsigned extractPos = i * 8;
              unsigned bitsToExtract = std::min(8u, parentWidth - extractPos);
              if (bitsToExtract > 0 && extractPos < parentWidth) {
                block->data[i] = static_cast<uint8_t>(
                    currentVal.extractBits(bitsToExtract, extractPos)
                        .getZExtValue());
              }
            }
            block->initialized = true;

            LLVM_DEBUG(llvm::dbgs()
                       << "  Drive to sig.extract alloca: bit " << totalBitOffset
                       << " = "
                       << (driveVal.isX() ? "X"
                                          : std::to_string(driveVal.getUInt64()))
                       << "\n");
            return success();
          }
        }
      }

      // Check if the parent has a signal ID (for actual signal bit extracts)
      SignalId parentSigId = getSignalId(parentRef);
      if (parentSigId == 0)
        parentSigId = resolveSignalId(parentRef);

      if (parentSigId != 0) {
        // Drive to a bit range within an actual signal - use read-modify-write
        InterpretedValue driveVal = getValue(procId, driveOp.getValue());

        const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
        InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);
        unsigned parentWidth = parentVal.getWidth();

        APInt result = parentVal.isX() ? APInt::getZero(parentWidth)
                                       : parentVal.getAPInt();

        unsigned extractWidth = driveVal.getWidth();
        if (!driveVal.isX()) {
          APInt insertVal = driveVal.getAPInt();
          if (insertVal.getBitWidth() != extractWidth)
            insertVal = insertVal.zextOrTrunc(extractWidth);
          result.insertBits(insertVal, totalBitOffset);
        }

        // Schedule the signal update
        SimTime delay = convertTimeValue(procId, driveOp.getTime());
        SimTime currentTime = scheduler.getCurrentTime();
        SimTime targetTime = currentTime.advanceTime(delay.realTime);
        if (delay.deltaStep > 0)
          targetTime.deltaStep = delay.deltaStep;

        uint64_t driverId = (static_cast<uint64_t>(procId) << 32) |
                            static_cast<uint64_t>(parentSigId);

        SignalValue newVal(result);
        scheduler.getEventScheduler().schedule(
            targetTime, SchedulingRegion::NBA,
            Event([this, parentSigId, driverId, newVal]() {
              scheduler.updateSignalWithStrength(parentSigId, driverId, newVal,
                                                 DriveStrength::Strong,
                                                 DriveStrength::Strong);
            }));

        return success();
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Drive to sig.extract failed - could not resolve ref\n");
      return failure();
    }

    // Check if this is a local variable access via UnrealizedConversionCastOp
    // This happens when local variables in functions are accessed - they're
    // backed by llvm.alloca and cast to !llhd.ref, not actual LLHD signals.
    if (auto castOp = signal.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1) {
        Value input = castOp.getInputs()[0];

        // Handle AllocaOp - local variables backed by llvm.alloca
        if (auto allocaOp = input.getDefiningOp<LLVM::AllocaOp>()) {
          // Get the value to drive
          InterpretedValue driveVal = getValue(procId, driveOp.getValue());

          // Find the memory block for this alloca
          MemoryBlock *block = findMemoryBlock(procId, allocaOp);
          if (!block) {
            // Try finding by alloca result in process-local memory blocks
            auto &state = processStates[procId];
            auto it = state.memoryBlocks.find(allocaOp.getResult());
            if (it != state.memoryBlocks.end()) {
              block = &it->second;
              LLVM_DEBUG(llvm::dbgs() << "  Drive: found local alloca memory\n");
            }
          }

          if (!block) {
            LLVM_DEBUG(llvm::dbgs() << "  Drive to alloca failed - memory not found\n");
            return failure();
          }

          unsigned width = driveVal.getWidth();
          unsigned storeSize = (width + 7) / 8;

          if (storeSize > block->size) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Drive: out of bounds (size=" << storeSize
                       << " block=" << block->size << ")\n");
            return failure();
          }

          // Write the value to memory
          if (driveVal.isX()) {
            // Write X pattern (all 1s as marker)
            std::fill(block->data.begin(), block->data.begin() + storeSize, 0xFF);
            block->initialized = false;
          } else {
            uint64_t value = driveVal.getUInt64();
            for (unsigned i = 0; i < storeSize; ++i) {
              block->data[i] = (value >> (i * 8)) & 0xFF;
            }
            block->initialized = true;
          }

          LLVM_DEBUG(llvm::dbgs()
                     << "  Drive to alloca: "
                     << (driveVal.isX() ? "X" : std::to_string(driveVal.getUInt64()))
                     << " (width=" << width << ")\n");
          return success();
        }

        // Handle GEP-based memory access (class member fields).
        // This happens when class properties are driven via
        // unrealized_conversion_cast from a GEP pointer to !llhd.ref.
        // Mirrors the probe handler at interpretProbe.
        if (auto gepOp = input.getDefiningOp<LLVM::GEPOp>()) {
          InterpretedValue ptrVal = getValue(procId, gepOp.getResult());
          InterpretedValue driveVal = getValue(procId, driveOp.getValue());
          auto refType = cast<llhd::RefType>(signal.getType());
          unsigned width = getTypeWidth(refType.getNestedType());
          unsigned storeSize = (width + 7) / 8;

          if (ptrVal.isX()) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Drive to GEP pointer: X (uninitialized)\n");
            return success();
          }

          uint64_t addr = ptrVal.getUInt64();
          uint64_t offset = 0;
          MemoryBlock *block = findMemoryBlock(procId, gepOp);

          if (block) {
            InterpretedValue baseVal = getValue(procId, gepOp.getBase());
            if (!baseVal.isX())
              offset = addr - baseVal.getUInt64();
          } else {
            // Check globals
            for (auto &entry : globalAddresses) {
              StringRef globalName = entry.first();
              uint64_t globalBaseAddr = entry.second;
              auto blockIt = globalMemoryBlocks.find(globalName);
              if (blockIt != globalMemoryBlocks.end()) {
                uint64_t globalSize = blockIt->second.size;
                if (addr >= globalBaseAddr &&
                    addr < globalBaseAddr + globalSize) {
                  block = &blockIt->second;
                  offset = addr - globalBaseAddr;
                  break;
                }
              }
            }
            // Check malloc blocks
            if (!block) {
              for (auto &entry : mallocBlocks) {
                uint64_t mallocBaseAddr = entry.first;
                uint64_t mallocSize = entry.second.size;
                if (addr >= mallocBaseAddr &&
                    addr < mallocBaseAddr + mallocSize) {
                  block = &entry.second;
                  offset = addr - mallocBaseAddr;
                  break;
                }
              }
            }
          }

          if (block && offset + storeSize <= block->size) {
            if (driveVal.isX()) {
              for (unsigned i = 0; i < storeSize; ++i)
                block->data[offset + i] = 0xFF;
            } else {
              APInt val = driveVal.getAPInt();
              if (val.getBitWidth() < width)
                val = val.zext(width);
              else if (val.getBitWidth() > width)
                val = val.trunc(width);
              for (unsigned i = 0; i < storeSize; ++i) {
                unsigned bitPos = i * 8;
                unsigned bitsToWrite = std::min(8u, width - bitPos);
                if (bitsToWrite > 0 && bitPos < width)
                  block->data[offset + i] =
                      val.extractBits(bitsToWrite, bitPos).getZExtValue();
                else
                  block->data[offset + i] = 0;
              }
            }
            block->initialized = !driveVal.isX();

            LLVM_DEBUG(llvm::dbgs()
                       << "  Drive to GEP memory at 0x"
                       << llvm::format_hex(addr, 16) << " offset " << offset
                       << " width " << width << "\n");
            return success();
          }

          LLVM_DEBUG(llvm::dbgs()
                     << "  Drive to GEP pointer 0x"
                     << llvm::format_hex(addr, 0)
                     << " failed - memory not found\n");
          // Fall through to other handlers
        }

        // Handle addressof-based global variable access.
        // This happens when static class properties are driven via
        // unrealized_conversion_cast from addressof to !llhd.ref.
        if (auto addrOfOp = input.getDefiningOp<LLVM::AddressOfOp>()) {
          StringRef globalName = addrOfOp.getGlobalName();
          InterpretedValue driveVal = getValue(procId, driveOp.getValue());
          auto refType = cast<llhd::RefType>(signal.getType());
          unsigned width = getTypeWidth(refType.getNestedType());
          unsigned storeSize = (width + 7) / 8;

          auto blockIt = globalMemoryBlocks.find(globalName);
          if (blockIt != globalMemoryBlocks.end()) {
            MemoryBlock &block = blockIt->second;
            if (storeSize <= block.size) {
              if (driveVal.isX()) {
                for (unsigned i = 0; i < storeSize; ++i)
                  block.data[i] = 0xFF;
                block.initialized = false;
              } else {
                APInt val = driveVal.getAPInt();
                if (val.getBitWidth() < width)
                  val = val.zext(width);
                else if (val.getBitWidth() > width)
                  val = val.trunc(width);
                for (unsigned i = 0; i < storeSize; ++i) {
                  unsigned bitPos = i * 8;
                  unsigned bitsToWrite = std::min(8u, width - bitPos);
                  if (bitsToWrite > 0 && bitPos < width)
                    block.data[i] =
                        val.extractBits(bitsToWrite, bitPos).getZExtValue();
                  else
                    block.data[i] = 0;
                }
                block.initialized = true;
              }

              LLVM_DEBUG(llvm::dbgs()
                         << "  Drive to global '" << globalName << "' width "
                         << width << "\n");
              return success();
            }
          }
          LLVM_DEBUG(llvm::dbgs()
                     << "  Drive to global '" << globalName
                     << "' failed - memory not found\n");
          // Fall through
        }
      }
    }

    // Handle llhd.sig.struct_extract - drive to a field within a struct signal.
    // We need to read-modify-write the parent signal.
    if (auto sigExtractOp = signal.getDefiningOp<llhd::SigStructExtractOp>()) {
      // Find the parent signal ID by tracing through nested extracts
      Value parentSignal = sigExtractOp.getInput();
      SignalId parentSigId = getSignalId(parentSignal);

      // Handle nested struct extracts by tracing to the root signal
      llvm::SmallVector<llhd::SigStructExtractOp, 4> extractChain;
      extractChain.push_back(sigExtractOp);

      while (parentSigId == 0) {
        if (auto nestedExtract =
                parentSignal.getDefiningOp<llhd::SigStructExtractOp>()) {
          extractChain.push_back(nestedExtract);
          parentSignal = nestedExtract.getInput();
          parentSigId = getSignalId(parentSignal);
        } else {
          break;
        }
      }

      // If still not found, try resolveSignalId which handles more cases
      // like tracing through UnrealizedConversionCastOp, block arguments, etc.
      if (parentSigId == 0) {
        parentSigId = resolveSignalId(parentSignal);
      }

      // Check if the parent is actually a memory location (llvm.alloca via cast)
      // rather than an actual LLHD signal. This happens in combinational blocks
      // where structs are built up by driving individual fields.
      if (parentSigId == 0) {
        if (auto castOp =
                parentSignal.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
          if (castOp.getInputs().size() == 1) {
            Value input = castOp.getInputs()[0];
            if (auto allocaOp = input.getDefiningOp<LLVM::AllocaOp>()) {
              // This is a drive to a field within a memory-backed struct.
              // We need to do a read-modify-write of the memory block.

              // Get the value to drive
              InterpretedValue driveVal = getValue(procId, driveOp.getValue());

              // Find the memory block for this alloca
              MemoryBlock *block = findMemoryBlock(procId, allocaOp);
              if (!block) {
                auto &state = processStates[procId];
                auto it = state.memoryBlocks.find(allocaOp.getResult());
                if (it != state.memoryBlocks.end()) {
                  block = &it->second;
                }
              }

              if (!block) {
                LLVM_DEBUG(llvm::dbgs()
                           << "  Drive to struct field in alloca failed - "
                              "memory not found\n");
                return failure();
              }

              // Compute the bit offset by walking the extract chain in reverse.
              // For memory-backed structs (LLVM alloca), we use LLVM layout
              // where fields are at low-to-high bits (field 0 at bit 0).
              unsigned bitOffset = 0;
              Type currentType = parentSignal.getType();
              if (auto refType = dyn_cast<llhd::RefType>(currentType))
                currentType = refType.getNestedType();

              for (auto it = extractChain.rbegin(); it != extractChain.rend();
                   ++it) {
                auto extractOp = *it;
                auto structType = cast<hw::StructType>(currentType);
                auto elements = structType.getElements();
                StringRef fieldName = extractOp.getField();

                auto fieldIndexOpt = structType.getFieldIndex(fieldName);
                if (!fieldIndexOpt) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "  Error: Field not found: " << fieldName
                             << "\n");
                  return failure();
                }
                unsigned fieldIndex = *fieldIndexOpt;

                // For LLVM struct layout: fields are at low-to-high bits.
                // Field 0 starts at bit 0, field 1 starts after field 0, etc.
                unsigned fieldOffset = 0;
                for (size_t i = 0; i < fieldIndex; ++i)
                  fieldOffset += getTypeWidth(elements[i].type);

                bitOffset += fieldOffset;
                currentType = elements[fieldIndex].type;
              }

              unsigned fieldWidth = getTypeWidth(currentType);
              unsigned parentWidth = getTypeWidth(parentSignal.getType());
              unsigned storeSize = (parentWidth + 7) / 8;

              if (storeSize > block->size) {
                LLVM_DEBUG(llvm::dbgs()
                           << "  Drive to struct field in alloca: out of bounds\n");
                return failure();
              }

              // Read the current value from memory
              APInt currentVal = APInt::getZero(parentWidth);
              for (unsigned i = 0; i < storeSize && i < block->data.size();
                   ++i) {
                unsigned insertPos = i * 8;
                unsigned bitsToInsert = std::min(8u, parentWidth - insertPos);
                if (bitsToInsert > 0 && insertPos < parentWidth) {
                  APInt byteVal(bitsToInsert,
                                block->data[i] & ((1u << bitsToInsert) - 1));
                  currentVal.insertBits(byteVal, insertPos);
                }
              }

              // Insert the new field value at the computed bit offset.
              // The drive value is in HW layout, but the alloca stores
              // in LLVM layout. Convert if the field is a struct/array.
              APInt fieldValue = driveVal.isX() ? APInt::getZero(fieldWidth)
                                                : driveVal.getAPInt();
              if (!driveVal.isX() &&
                  isa<hw::StructType, hw::ArrayType>(currentType)) {
                // Find the corresponding LLVM type for conversion
                Type hwFieldType = currentType;
                Type llvmFieldType;
                // Try to get the LLVM type from the alloca's original type
                if (auto allocaPtrType =
                        dyn_cast<LLVM::LLVMPointerType>(allocaOp.getType())) {
                  // The alloca elem type corresponds to the parent signal type
                  // We need to find the LLVM field type at the same position
                  Type allocElemType = allocaOp.getElemType();
                  if (allocElemType) {
                    Type curLLVM = allocElemType;
                    for (auto it2 = extractChain.rbegin();
                         it2 != extractChain.rend(); ++it2) {
                      auto extractOp2 = *it2;
                      Type rawType =
                          it2 == extractChain.rbegin()
                              ? parentSignal.getType()
                              : (*(it2 - 1))->getResult(0).getType();
                      // Unwrap RefType if present
                      Type refInner = rawType;
                      if (auto refT = dyn_cast<llhd::RefType>(rawType))
                        refInner = refT.getNestedType();
                      auto hwST = dyn_cast<hw::StructType>(refInner);
                      if (!hwST) break;
                      auto fidx = hwST.getFieldIndex(extractOp2.getField());
                      if (!fidx) break;
                      if (auto llvmST = dyn_cast<LLVM::LLVMStructType>(curLLVM)) {
                        auto body = llvmST.getBody();
                        if (*fidx < body.size())
                          curLLVM = body[*fidx];
                        else
                          break;
                      } else {
                        break;
                      }
                    }
                    llvmFieldType = curLLVM;
                  }
                }
                if (llvmFieldType &&
                    isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmFieldType)) {
                  fieldValue = convertHWToLLVMLayout(fieldValue, hwFieldType,
                                                     llvmFieldType);
                }
              }
              if (fieldValue.getBitWidth() < fieldWidth)
                fieldValue = fieldValue.zext(fieldWidth);
              else if (fieldValue.getBitWidth() > fieldWidth)
                fieldValue = fieldValue.trunc(fieldWidth);

              currentVal.insertBits(fieldValue, bitOffset);

              // Write the modified value back to memory
              for (unsigned i = 0; i < storeSize; ++i) {
                unsigned extractPos = i * 8;
                unsigned bitsToExtract =
                    std::min(8u, parentWidth - extractPos);
                if (bitsToExtract > 0 && extractPos < parentWidth) {
                  block->data[i] =
                      currentVal.extractBits(bitsToExtract, extractPos)
                          .getZExtValue();
                } else {
                  block->data[i] = 0;
                }
              }
              block->initialized = !driveVal.isX();

              LLVM_DEBUG(llvm::dbgs()
                         << "  Drive to struct field in alloca at offset "
                         << bitOffset << " width " << fieldWidth << "\n");
              return success();
            }
          }
        }
      }

      // Handle memory-backed !llhd.ref passed as function argument or through
      // other indirect paths. The runtime value contains the alloca address.
      if (parentSigId == 0) {
        InterpretedValue parentPtrVal = getValue(procId, parentSignal);
        if (!parentPtrVal.isX() && parentPtrVal.getUInt64() != 0) {
          uint64_t addr = parentPtrVal.getUInt64();
          uint64_t blockOffset = 0;
          MemoryBlock *block =
              findMemoryBlockByAddress(addr, procId, &blockOffset);
          if (block) {
            InterpretedValue driveVal = getValue(procId, driveOp.getValue());

            // Compute field bit offset using LLVM layout (low-to-high bits).
            unsigned bitOffset = 0;
            Type currentType = parentSignal.getType();
            if (auto refType = dyn_cast<llhd::RefType>(currentType))
              currentType = refType.getNestedType();

            for (auto it = extractChain.rbegin(); it != extractChain.rend();
                 ++it) {
              auto extractOp = *it;
              auto structType = cast<hw::StructType>(currentType);
              auto elements = structType.getElements();
              StringRef fieldName = extractOp.getField();
              auto fieldIndexOpt = structType.getFieldIndex(fieldName);
              if (!fieldIndexOpt)
                return failure();
              unsigned fieldIndex = *fieldIndexOpt;
              unsigned fieldOff = 0;
              for (size_t i = 0; i < fieldIndex; ++i)
                fieldOff += getTypeWidth(elements[i].type);
              bitOffset += fieldOff;
              currentType = elements[fieldIndex].type;
            }

            unsigned fieldWidth = getTypeWidth(currentType);
            unsigned parentWidth = getTypeWidth(parentSignal.getType());
            unsigned storeSize = (parentWidth + 7) / 8;

            if (blockOffset + storeSize <= block->size) {
              // Read current parent value from memory
              APInt currentVal = APInt::getZero(parentWidth);
              for (unsigned i = 0; i < storeSize; ++i) {
                unsigned insertPos = i * 8;
                unsigned bitsToInsert =
                    std::min(8u, parentWidth - insertPos);
                if (bitsToInsert > 0 && insertPos < parentWidth) {
                  APInt byteVal(bitsToInsert,
                                block->data[blockOffset + i] &
                                    ((1u << bitsToInsert) - 1));
                  currentVal.insertBits(byteVal, insertPos);
                }
              }

              // Insert the new field value
              APInt fieldValue = driveVal.isX()
                                     ? APInt::getZero(fieldWidth)
                                     : driveVal.getAPInt();
              if (fieldValue.getBitWidth() < fieldWidth)
                fieldValue = fieldValue.zext(fieldWidth);
              else if (fieldValue.getBitWidth() > fieldWidth)
                fieldValue = fieldValue.trunc(fieldWidth);

              currentVal.insertBits(fieldValue, bitOffset);

              // Write back to memory
              for (unsigned i = 0; i < storeSize; ++i) {
                unsigned extractPos = i * 8;
                unsigned bitsToExtract =
                    std::min(8u, parentWidth - extractPos);
                if (bitsToExtract > 0 && extractPos < parentWidth) {
                  block->data[blockOffset + i] =
                      currentVal.extractBits(bitsToExtract, extractPos)
                          .getZExtValue();
                } else {
                  block->data[blockOffset + i] = 0;
                }
              }
              block->initialized = !driveVal.isX();

              LLVM_DEBUG(llvm::dbgs()
                         << "  Drive to struct field in memory-backed ref at "
                            "offset "
                         << bitOffset << " width " << fieldWidth << "\n");
              return success();
            }
          }
        }
      }

      if (parentSigId == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Error: Could not find parent signal for struct extract\n");
        return failure();
      }

      // Get the value to drive
      InterpretedValue driveVal = getValue(procId, driveOp.getValue());

      // Get the current value of the parent signal
      const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
      InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);

      // If parent is X, we can still drive a field (result will be X except for
      // the driven field)
      unsigned parentWidth = parentVal.getWidth();
      APInt result = parentVal.isX() ? APInt::getZero(parentWidth)
                                     : parentVal.getAPInt();

      // Compute the bit offset by walking the extract chain in reverse
      // (from root signal to the target field)
      unsigned bitOffset = 0;
      Type currentType = parentSignal.getType();
      if (auto refType = dyn_cast<llhd::RefType>(currentType))
        currentType = refType.getNestedType();

      for (auto it = extractChain.rbegin(); it != extractChain.rend(); ++it) {
        auto extractOp = *it;
        auto structType = cast<hw::StructType>(currentType);
        auto elements = structType.getElements();
        StringRef fieldName = extractOp.getField();

        auto fieldIndexOpt = structType.getFieldIndex(fieldName);
        if (!fieldIndexOpt) {
          LLVM_DEBUG(llvm::dbgs() << "  Error: Field not found: " << fieldName
                                  << "\n");
          return failure();
        }
        unsigned fieldIndex = *fieldIndexOpt;

        // Fields are laid out from high bits to low bits
        // Calculate offset from the low bit of the current struct
        unsigned fieldOffset = 0;
        for (size_t i = fieldIndex + 1; i < elements.size(); ++i)
          fieldOffset += getTypeWidth(elements[i].type);

        bitOffset += fieldOffset;
        currentType = elements[fieldIndex].type;
      }

      unsigned fieldWidth = getTypeWidth(currentType);

      // Insert the new value at the computed bit offset
      APInt fieldValue = driveVal.isX() ? APInt::getZero(fieldWidth)
                                        : driveVal.getAPInt();
      if (fieldValue.getBitWidth() < fieldWidth)
        fieldValue = fieldValue.zext(fieldWidth);
      else if (fieldValue.getBitWidth() > fieldWidth)
        fieldValue = fieldValue.trunc(fieldWidth);

      result.insertBits(fieldValue, bitOffset);

      // Get the delay time
      SimTime delay = convertTimeValue(procId, driveOp.getTime());
      SimTime currentTime = scheduler.getCurrentTime();
      SimTime targetTime = currentTime.advanceTime(delay.realTime);
      if (delay.deltaStep > 0)
        targetTime.deltaStep = delay.deltaStep;

      // Use the same driver ID scheme
      uint64_t driverId = (static_cast<uint64_t>(procId) << 32) |
                          static_cast<uint64_t>(parentSigId);

      LLVM_DEBUG(llvm::dbgs()
                 << "  Drive to struct field at offset " << bitOffset
                 << " width " << fieldWidth << " in signal " << parentSigId
                 << "\n");

      // Schedule the signal update
      SignalValue newVal(result);
      scheduler.getEventScheduler().schedule(
          targetTime, SchedulingRegion::NBA,
          Event([this, parentSigId, driverId, newVal]() {
            scheduler.updateSignalWithStrength(parentSigId, driverId, newVal,
                                               DriveStrength::Strong,
                                               DriveStrength::Strong);
          }));

      return success();
    }

    // Handle llhd.sig.array_get - drive to an element within an array signal.
    // We need to read-modify-write the parent signal.
    if (auto sigArrayGetOp = signal.getDefiningOp<llhd::SigArrayGetOp>()) {
      Value parentSignal = sigArrayGetOp.getInput();
      SignalId parentSigId = getSignalId(parentSignal);

      // If not found directly, try resolveSignalId
      if (parentSigId == 0) {
        parentSigId = resolveSignalId(parentSignal);
      }

      // Check if the parent is a memory-backed array (llvm.alloca or malloc via cast)
      if (parentSigId == 0) {
        if (auto castOp =
                parentSignal.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
          if (castOp.getInputs().size() == 1) {
            Value input = castOp.getInputs()[0];

            // Get the index value (may be dynamic)
            InterpretedValue indexVal = getValue(procId, sigArrayGetOp.getIndex());
            if (indexVal.isX()) {
              LLVM_DEBUG(llvm::dbgs() << "  Warning: X index in array drive\n");
              return success(); // Don't drive with X index
            }
            uint64_t index = indexVal.getUInt64();

            // Get array element type and width
            auto arrayType = cast<hw::ArrayType>(unwrapSignalType(parentSignal.getType()));
            Type elementType = arrayType.getElementType();
            unsigned elementWidth = getTypeWidth(elementType);
            size_t numElements = arrayType.getNumElements();

            // Bounds check
            if (index >= numElements) {
              LLVM_DEBUG(llvm::dbgs() << "  Warning: Array index " << index
                                      << " out of bounds (size " << numElements << ")\n");
              return success();
            }

            // Get the value to drive
            InterpretedValue driveVal = getValue(procId, driveOp.getValue());

            // Try to find the memory block - could be from pointer value
            InterpretedValue ptrVal = getValue(procId, input);
            if (!ptrVal.isX()) {
              uint64_t baseAddr = ptrVal.getUInt64();
              MemoryBlock *block = nullptr;
              uint64_t baseOffset = 0;

              // Check malloc blocks first
              for (auto &entry : mallocBlocks) {
                uint64_t mallocBaseAddr = entry.first;
                uint64_t mallocSize = entry.second.size;
                if (baseAddr >= mallocBaseAddr && baseAddr < mallocBaseAddr + mallocSize) {
                  block = &entry.second;
                  baseOffset = baseAddr - mallocBaseAddr;
                  LLVM_DEBUG(llvm::dbgs() << "  Array drive: found malloc block at 0x"
                                          << llvm::format_hex(mallocBaseAddr, 16) << "\n");
                  break;
                }
              }

              // Check global blocks
              if (!block) {
                for (auto &entry : globalAddresses) {
                  StringRef globalName = entry.first();
                  uint64_t globalBaseAddr = entry.second;
                  auto blockIt = globalMemoryBlocks.find(globalName);
                  if (blockIt != globalMemoryBlocks.end()) {
                    uint64_t globalSize = blockIt->second.size;
                    if (baseAddr >= globalBaseAddr && baseAddr < globalBaseAddr + globalSize) {
                      block = &blockIt->second;
                      baseOffset = baseAddr - globalBaseAddr;
                      LLVM_DEBUG(llvm::dbgs() << "  Array drive: found global '" << globalName
                                              << "'\n");
                      break;
                    }
                  }
                }
              }

              if (block) {
                // Calculate byte offset for the element
                unsigned elementByteWidth = (elementWidth + 7) / 8;
                uint64_t elementOffset = baseOffset + index * elementByteWidth;

                if (elementOffset + elementByteWidth <= block->size) {
                  // Write the element value to memory (little-endian)
                  if (driveVal.isX()) {
                    // Write zeros for X
                    for (unsigned i = 0; i < elementByteWidth; ++i) {
                      block->data[elementOffset + i] = 0;
                    }
                  } else {
                    APInt val = driveVal.getAPInt();
                    if (val.getBitWidth() < elementWidth)
                      val = val.zext(elementWidth);
                    else if (val.getBitWidth() > elementWidth)
                      val = val.trunc(elementWidth);
                    for (unsigned i = 0; i < elementByteWidth; ++i) {
                      block->data[elementOffset + i] =
                          val.extractBits(std::min(8u, elementWidth - i * 8), i * 8)
                              .getZExtValue();
                    }
                  }
                  block->initialized = !driveVal.isX();

                  LLVM_DEBUG(llvm::dbgs()
                             << "  Drive to memory-backed array[" << index
                             << "] at offset " << elementOffset
                             << " width " << elementWidth << "\n");
                  return success();
                }
              }
            }
          }
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "  Error: Could not find parent signal for array get\n");
        return failure();
      }

      // Get the index value (may be dynamic)
      InterpretedValue indexVal = getValue(procId, sigArrayGetOp.getIndex());
      if (indexVal.isX()) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: X index in array drive\n");
        return success(); // Don't drive with X index
      }
      uint64_t index = indexVal.getUInt64();

      // Get the value to drive
      InterpretedValue driveVal = getValue(procId, driveOp.getValue());

      // Get the current value of the parent signal
      const SignalValue &parentSV = scheduler.getSignalValue(parentSigId);
      InterpretedValue parentVal = InterpretedValue::fromSignalValue(parentSV);

      unsigned parentWidth = parentVal.getWidth();
      APInt result = parentVal.isX() ? APInt::getZero(parentWidth)
                                     : parentVal.getAPInt();

      // Get array element type and width
      auto arrayType = cast<hw::ArrayType>(unwrapSignalType(parentSignal.getType()));
      Type elementType = arrayType.getElementType();
      unsigned elementWidth = getTypeWidth(elementType);
      size_t numElements = arrayType.getNumElements();

      // Bounds check
      if (index >= numElements) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Array index " << index
                                << " out of bounds (size " << numElements << ")\n");
        return success(); // Out of bounds - don't drive
      }

      // hw::ArrayType layout: element 0 at low bits, element N-1 at high bits
      unsigned bitOffset = index * elementWidth;

      // Insert the new value at the computed bit offset
      APInt elementValue = driveVal.isX() ? APInt::getZero(elementWidth)
                                          : driveVal.getAPInt();
      if (elementValue.getBitWidth() < elementWidth)
        elementValue = elementValue.zext(elementWidth);
      else if (elementValue.getBitWidth() > elementWidth)
        elementValue = elementValue.trunc(elementWidth);

      result.insertBits(elementValue, bitOffset);

      // Get the delay time
      SimTime delay = convertTimeValue(procId, driveOp.getTime());
      SimTime currentTime = scheduler.getCurrentTime();
      SimTime targetTime = currentTime.advanceTime(delay.realTime);
      if (delay.deltaStep > 0)
        targetTime.deltaStep = delay.deltaStep;

      // Use the same driver ID scheme
      uint64_t driverId = (static_cast<uint64_t>(procId) << 32) |
                          static_cast<uint64_t>(parentSigId);

      LLVM_DEBUG(llvm::dbgs()
                 << "  Drive to array element[" << index << "] at offset "
                 << bitOffset << " width " << elementWidth << " in signal "
                 << parentSigId << "\n");

      // Schedule the signal update
      SignalValue newVal(result);
      scheduler.getEventScheduler().schedule(
          targetTime, SchedulingRegion::NBA,
          Event([this, parentSigId, driverId, newVal]() {
            scheduler.updateSignalWithStrength(parentSigId, driverId, newVal,
                                               DriveStrength::Strong,
                                               DriveStrength::Strong);
          }));

      return success();
    }

    // Handle memory-backed ref arguments (e.g., !llhd.ref passed through
    // function calls from UnrealizedConversionCastOp of GEP/addressof).
    // These refs are backed by LLVM memory locations (class fields, globals),
    // not LLHD signals. The interpreted value of the ref is the memory address.
    // This is critical for uvm_config_db::set which stores values into
    // associative arrays via ref arguments passed through multiple call layers.
    if (isa<BlockArgument>(signal) && isa<llhd::RefType>(signal.getType())) {
      InterpretedValue addrVal = getValue(procId, signal);
      if (!addrVal.isX() && addrVal.getUInt64() != 0) {
        uint64_t addr = addrVal.getUInt64();
        InterpretedValue driveVal = getValue(procId, driveOp.getValue());

        // Get the type being driven
        auto refType = cast<llhd::RefType>(signal.getType());
        unsigned width = getTypeWidth(refType.getNestedType());
        unsigned storeSize = (width + 7) / 8;

        // Find the memory block at this address using the comprehensive
        // search that checks process-local allocas, module-level allocas,
        // malloc blocks, and global memory blocks.  The previous manual
        // search only checked globals and mallocs, missing process-local
        // allocas (e.g., automatic variables passed by ref through
        // func.call chains like uvm_resource_debug::init_access_record).
        uint64_t offset = 0;
        MemoryBlock *block = findMemoryBlockByAddress(addr, procId, &offset);

        if (block && offset + storeSize <= block->size) {
          if (driveVal.isX()) {
            // Write X pattern
            for (unsigned i = 0; i < storeSize; ++i)
              block->data[offset + i] = 0xFF;
          } else {
            APInt val = driveVal.getAPInt();
            if (val.getBitWidth() < width)
              val = val.zext(width);
            else if (val.getBitWidth() > width)
              val = val.trunc(width);
            for (unsigned i = 0; i < storeSize; ++i) {
              unsigned bitPos = i * 8;
              unsigned bitsToWrite = std::min(8u, width - bitPos);
              if (bitsToWrite > 0 && bitPos < width)
                block->data[offset + i] =
                    val.extractBits(bitsToWrite, bitPos).getZExtValue();
              else
                block->data[offset + i] = 0;
            }
          }
          block->initialized = !driveVal.isX();

          LLVM_DEBUG(llvm::dbgs()
                     << "  Drive to memory-backed ref at 0x"
                     << llvm::format_hex(addr, 16) << " offset " << offset
                     << " width " << width << "\n");
          return success();
        }

        LLVM_DEBUG(llvm::dbgs()
                   << "  Drive to memory-backed ref at 0x"
                   << llvm::format_hex(addr, 16)
                   << " failed - memory not found\n");
        // Fall through to resolveSignalId as last resort
      }
    }

    // Try resolveSignalId which handles more cases like tracing through
    // UnrealizedConversionCastOp, block arguments, instance outputs, etc.
    // This is needed for direct struct drives where the signal reference
    // might not be in the direct valueToSignal map.
    sigId = resolveSignalId(signal);
    if (sigId != 0) {
      LLVM_DEBUG(llvm::dbgs() << "  Resolved signal via resolveSignalId: "
                              << sigId << "\n");
      // Fall through to the normal drive handling below
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  Error: Unknown signal in drive\n");
      return failure();
    }
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

  // Extract strength attributes if present
  DriveStrength strength0 = DriveStrength::Strong; // Default
  DriveStrength strength1 = DriveStrength::Strong; // Default

  if (auto s0Attr = driveOp.getStrength0Attr()) {
    // Convert LLHD DriveStrength to sim DriveStrength
    strength0 = static_cast<DriveStrength>(
        static_cast<uint8_t>(s0Attr.getValue()));
  }
  if (auto s1Attr = driveOp.getStrength1Attr()) {
    strength1 = static_cast<DriveStrength>(
        static_cast<uint8_t>(s1Attr.getValue()));
  }

  // Use a per-process, per-signal driver ID so multiple llhd.drv ops in the
  // same process model a single driver rather than conflicting drivers.
  uint64_t driverId =
      (static_cast<uint64_t>(procId) << 32) | static_cast<uint64_t>(sigId);

  LLVM_DEBUG(llvm::dbgs() << "  Scheduling drive to signal " << sigId
                          << " at time " << targetTime.realTime << " fs"
                          << " (delay " << delay.realTime << " fs)"
                          << " strength(" << getDriveStrengthName(strength0)
                          << ", " << getDriveStrengthName(strength1) << ")\n");

  // For epsilon/zero delays, also store in pending drives for immediate reads.
  // This enables blocking assignment semantics where a subsequent probe in the
  // same process sees the value immediately rather than waiting for the event.
  if (delay.realTime == 0 && delay.deltaStep <= 1) {
    pendingEpsilonDrives[sigId] = driveVal;
  }

  // Schedule the signal update with strength information
  SignalValue newVal = driveVal.toSignalValue();
  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::NBA,
      Event([this, sigId, driverId, newVal, strength0, strength1]() {
        scheduler.updateSignalWithStrength(sigId, driverId, newVal, strength0,
                                           strength1);
      }));

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretWait(ProcessId procId,
                                                     llhd::WaitOp waitOp) {
  auto &state = processStates[procId];

  // Handle yield operands - these become the process result values
  // The yield values are immediately available to the llhd.drv operations
  // that reference the process results.
  auto yieldOperands = waitOp.getYieldOperands();
  if (!yieldOperands.empty()) {
    // Get the parent process operation
    if (auto processOp = state.getProcessOp()) {
      auto results = processOp.getResults();
      for (auto [result, yieldOp] : llvm::zip(results, yieldOperands)) {
        InterpretedValue yieldVal = getValue(procId, yieldOp);
        // Store the yielded value so it can be accessed when evaluating
        // the process results outside the process
        state.valueMap[result] = yieldVal;
        LLVM_DEBUG(llvm::dbgs() << "  Yield value for process result: "
                                << (yieldVal.isX() ? "X"
                                                    : std::to_string(yieldVal.getUInt64()))
                                << "\n");
      }
    }

    // Execute module-level drives that depend on this process's results
    executeModuleDrives(procId);
    executeInstanceOutputUpdates(procId);
  }

  // Get the destination block
  state.destBlock = waitOp.getDest();

  // Store destination operands
  state.destOperands.clear();
  for (Value operand : waitOp.getDestOperands()) {
    state.destOperands.push_back(getValue(procId, operand));
  }

  // Mark as waiting
  state.waiting = true;
  bool hadDelay = static_cast<bool>(waitOp.getDelay());

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
  auto applySelfDrivenFilter = [&](SensitivityList &list) {
    if (list.empty())
      return;
    llvm::DenseSet<SignalId> selfDrivenSignals;
    if (auto processOp = state.getProcessOp()) {
      processOp.walk([&](llhd::DriveOp driveOp) {
        SignalId drivenId = getSignalId(driveOp.getSignal());
        if (drivenId != 0)
          selfDrivenSignals.insert(drivenId);
      });
    }
    // Include module-level drives fed by this process's results.
    // Also include transitive dependencies: signals that the drive value
    // depends on. If a process drives signal Z via a module-level drive that
    // reads signal X, then both Z and X are "self-driven" and should be
    // filtered from the sensitivity list to prevent zero-delta loops.
    for (auto &entry : moduleDrives) {
      if (entry.procId != procId)
        continue;
      ScopedInstanceContext instScope(*this, entry.instanceId);
      ScopedInputValueMap inputScope(*this, entry.inputMap);
      SignalId drivenId = getSignalId(entry.driveOp.getSignal());
      if (drivenId != 0)
        selfDrivenSignals.insert(drivenId);
      // Collect transitive dependencies from the drive value expression.
      llvm::SmallVector<SignalId, 4> transitiveSignals;
      collectSignalIds(entry.driveOp.getValue(), transitiveSignals);
      for (SignalId sigId : transitiveSignals)
        selfDrivenSignals.insert(sigId);
    }

    if (selfDrivenSignals.empty())
      return;
    bool hasNonSelf = false;
    for (const auto &entry : list.getEntries()) {
      if (!selfDrivenSignals.count(entry.signalId)) {
        hasNonSelf = true;
        break;
      }
    }

    if (hasNonSelf) {
      SensitivityList filtered;
      for (const auto &entry : list.getEntries()) {
        if (selfDrivenSignals.count(entry.signalId))
          continue;
        filtered.addEdge(entry.signalId, entry.edge);
      }
      list = std::move(filtered);
    }
  };

  if (!waitOp.getObserved().empty()) {
    SensitivityList waitList;
    auto cacheIt = state.waitSensitivityCache.find(waitOp.getOperation());
    if (cacheIt != state.waitSensitivityCache.end()) {
      for (const auto &entry : cacheIt->second)
        waitList.addEdge(entry.signalId, entry.edge);
      ++state.waitSensitivityCacheHits;
    }

    if (waitList.empty()) {
      // Helper function to recursively trace back through operations to find
      // the signal being observed. This handles chains like:
      //   %clk_bool = comb.and %value, %not_unknown
      //   %value = hw.struct_extract %probed["value"]
      //   %probed = llhd.prb %signal
      // We need to trace back through this chain to find %signal.
      std::function<SignalId(Value, int)> traceToSignal;
      traceToSignal = [&](Value value, int depth) -> SignalId {
        // Limit recursion depth to prevent infinite loops
        if (depth > 32)
          return 0;

        // Check if it's a signal reference directly
        SignalId sigId = getSignalId(value);
        if (sigId != 0)
          return sigId;

        // Trace through instance results to child outputs.
        auto instMapIt = instanceOutputMap.find(activeInstanceId);
        if (instMapIt != instanceOutputMap.end()) {
          auto instIt = instMapIt->second.find(value);
          if (instIt != instMapIt->second.end()) {
            const auto &info = instIt->second;
            ScopedInstanceContext instScope(
                *const_cast<LLHDProcessInterpreter *>(this), info.instanceId);
            if (info.inputMap.empty())
              return traceToSignal(info.outputValue, depth + 1);
            ScopedInputValueMap scope(
                *const_cast<LLHDProcessInterpreter *>(this), info.inputMap);
            return traceToSignal(info.outputValue, depth + 1);
          }
        }

        // Direct probe case
        if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
          SignalId sigId = resolveSignalId(probeOp.getSignal());
          if (sigId != 0) {
            LLVM_DEBUG(llvm::dbgs() << "  Found signal " << sigId
                                    << " from probe at depth " << depth << "\n");
            return sigId;
          }
        }

        // Block argument - trace through predecessors
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          Value mappedValue;
          InstanceId mappedInstance = activeInstanceId;
          if (lookupInputMapping(blockArg, mappedValue, mappedInstance) &&
              mappedValue != value) {
            ScopedInstanceContext scope(
                *const_cast<LLHDProcessInterpreter *>(this), mappedInstance);
            return traceToSignal(mappedValue, depth + 1);
          }
          Block *block = blockArg.getOwner();
          unsigned argIdx = blockArg.getArgNumber();

          // Look at all predecessors
          for (Block *pred : block->getPredecessors()) {
            Operation *terminator = pred->getTerminator();
            if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
              if (branchOp.getDest() == block &&
                  argIdx < branchOp.getNumOperands()) {
                Value incoming = branchOp.getDestOperands()[argIdx];
                sigId = traceToSignal(incoming, depth + 1);
                if (sigId != 0)
                  return sigId;
              }
            } else if (auto condBrOp =
                           dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
              if (condBrOp.getTrueDest() == block &&
                  argIdx < condBrOp.getNumTrueOperands()) {
                Value incoming = condBrOp.getTrueDestOperands()[argIdx];
                sigId = traceToSignal(incoming, depth + 1);
                if (sigId != 0)
                  return sigId;
              }
              if (condBrOp.getFalseDest() == block &&
                  argIdx < condBrOp.getNumFalseOperands()) {
                Value incoming = condBrOp.getFalseDestOperands()[argIdx];
                sigId = traceToSignal(incoming, depth + 1);
                if (sigId != 0)
                  return sigId;
              }
            }
          }
          return 0;
        }

        // Trace through defining operation's operands
        if (Operation *defOp = value.getDefiningOp()) {
          for (Value operand : defOp->getOperands()) {
            sigId = traceToSignal(operand, depth + 1);
            if (sigId != 0)
              return sigId;
          }
        }

        return 0;
      };

      for (Value observed : waitOp.getObserved()) {
        SignalId sigId = traceToSignal(observed, 0);

        if (sigId != 0) {
          waitList.addLevel(sigId);
          LLVM_DEBUG(llvm::dbgs() << "  Waiting on signal " << sigId << "\n");
        } else {
          llvm::SmallVector<SignalId, 4> fallbackSignals;
          collectSignalIds(observed, fallbackSignals);
          if (!fallbackSignals.empty()) {
            for (SignalId fallbackId : fallbackSignals)
              waitList.addLevel(fallbackId);
            LLVM_DEBUG(llvm::dbgs()
                       << "  Waiting on "
                       << fallbackSignals.size()
                       << " fallback signal(s) for observed value\n");
          } else {
            LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not find signal for "
                                        "observed value (type: "
                                    << observed.getType() << ")\n");
          }
        }
      }
    }

    // Avoid self-triggering on signals driven by this process when other
    // sensitivities exist (prevents zero-delta feedback loops).
    applySelfDrivenFilter(waitList);

    // Register the wait sensitivity with the scheduler
    if (waitList.empty()) {
      if (auto processOp = state.getProcessOp()) {
        processOp.walk([&](llhd::ProbeOp probeOp) {
          SignalId sigId = getSignalId(probeOp.getSignal());
          if (sigId != 0)
            waitList.addLevel(sigId);
        });
      } else if (auto initialOp = state.getInitialOp()) {
        initialOp.walk([&](llhd::ProbeOp probeOp) {
          SignalId sigId = getSignalId(probeOp.getSignal());
          if (sigId != 0)
            waitList.addLevel(sigId);
        });
      }
    }

    applySelfDrivenFilter(waitList);

    if (!waitList.empty()) {
      if (cacheIt == state.waitSensitivityCache.end())
        state.waitSensitivityCache.try_emplace(waitOp.getOperation(),
                                               waitList.getEntries());
      scheduler.suspendProcessForEvents(procId, waitList);
      LLVM_DEBUG(llvm::dbgs() << "  Registered for " << waitList.size()
                              << " signal events\n");
      cacheWaitState(state, scheduler, &waitList, hadDelay);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: No signals found in observed list, process "
                    "will not be triggered by events!\n");
      cacheWaitState(state, scheduler, nullptr, hadDelay);
    }
  }

  // Handle case 3: No delay AND no observed signals (always @(*) semantics)
  // Derive sensitivity from signals that influence outputs or yields, falling
  // back to probes and then a delta resume when no signals can be found.
  if (!waitOp.getDelay() && waitOp.getObserved().empty()) {
    SensitivityList waitList;
    auto cacheIt = state.waitSensitivityCache.find(waitOp.getOperation());
    if (cacheIt != state.waitSensitivityCache.end()) {
      for (const auto &entry : cacheIt->second)
        waitList.addEdge(entry.signalId, entry.edge);
      ++state.waitSensitivityCacheHits;
    } else {
      llvm::SmallVector<SignalId, 8> derivedSignals;
      llvm::DenseSet<SignalId> derivedSignalSet;
      auto appendSignals = [&](Value value) {
        llvm::SmallVector<SignalId, 4> signals;
        collectSignalIds(value, signals);
        for (SignalId sigId : signals) {
          if (sigId != 0 && derivedSignalSet.insert(sigId).second)
            derivedSignals.push_back(sigId);
        }
      };

      for (Value operand : waitOp.getYieldOperands())
        appendSignals(operand);

      if (auto processOp = state.getProcessOp()) {
        processOp.walk([&](llhd::DriveOp driveOp) {
          appendSignals(driveOp.getValue());
          if (driveOp.getEnable())
            appendSignals(driveOp.getEnable());
        });
      } else if (auto initialOp = state.getInitialOp()) {
        initialOp.walk([&](llhd::DriveOp driveOp) {
          appendSignals(driveOp.getValue());
          if (driveOp.getEnable())
            appendSignals(driveOp.getEnable());
        });
      }

      if (!derivedSignals.empty()) {
        for (SignalId sigId : derivedSignals)
          waitList.addLevel(sigId);
      } else if (auto processOp = state.getProcessOp()) {
        processOp.walk([&](llhd::ProbeOp probeOp) {
          SignalId sigId = getSignalId(probeOp.getSignal());
          if (sigId != 0)
            waitList.addLevel(sigId);
        });
      } else if (auto initialOp = state.getInitialOp()) {
        initialOp.walk([&](llhd::ProbeOp probeOp) {
          SignalId sigId = getSignalId(probeOp.getSignal());
          if (sigId != 0)
            waitList.addLevel(sigId);
        });
      }

      applySelfDrivenFilter(waitList);
      if (!waitList.empty())
        state.waitSensitivityCache.try_emplace(waitOp.getOperation(),
                                               waitList.getEntries());
    }

    if (!waitList.empty()) {
      scheduler.suspendProcessForEvents(procId, waitList);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Wait with no delay/no signals - derived "
                 << waitList.size() << " probe signal(s)\n");
      cacheWaitState(state, scheduler, &waitList, hadDelay);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Wait with no delay and no signals - scheduling "
                    "immediate delta-step resumption (always @(*) fallback)\n");

      // Schedule the process to resume on the next delta cycle.
      // This ensures the process doesn't hang when no signals are detected.
      scheduler.getEventScheduler().scheduleNextDelta(
          SchedulingRegion::Active,
          Event([this, procId]() { resumeProcess(procId); }));
      cacheWaitState(state, scheduler, nullptr, hadDelay);
    }
  }

  if (waitOp.getDelay() && waitOp.getObserved().empty())
    cacheWaitState(state, scheduler, nullptr, hadDelay);

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretHalt(ProcessId procId,
                                                     llhd::HaltOp haltOp) {
  auto &state = processStates[procId];

  // Check if this process has active forked children that haven't completed.
  // If so, we cannot halt yet - we must wait for all children to complete.
  // This is critical for UVM phase termination where run_test() spawns UVM
  // phases via fork-join, and the parent process must not halt until all
  // forked children complete.
  if (forkJoinManager.hasActiveChildren(procId)) {
    LLVM_DEBUG(llvm::dbgs() << "  Halt deferred - process has active forked children\n");

    // Suspend the process instead of halting - it will be resumed when
    // all children complete (via the fork/join completion mechanism)
    state.waiting = true;

    // Store the halt operation so we can re-execute it when children complete
    state.destBlock = haltOp->getBlock();
    state.currentOp = mlir::Block::iterator(haltOp);
    state.resumeAtCurrentOp = true; // Resume at halt op, not block beginning

    Process *proc = scheduler.getProcess(procId);
    if (proc)
      proc->setState(ProcessState::Waiting);

    return success();
  }

  // Handle yield operands - these become the process result values
  // The yield values are immediately available to the llhd.drv operations
  // that reference the process results.
  auto yieldOperands = haltOp.getYieldOperands();
  if (!yieldOperands.empty()) {
    // Get the parent process operation
    if (auto processOp = state.getProcessOp()) {
      auto results = processOp.getResults();
      for (auto [result, yieldOp] : llvm::zip(results, yieldOperands)) {
        InterpretedValue yieldVal = getValue(procId, yieldOp);
        // Store the yielded value so it can be accessed when evaluating
        // the process results outside the process
        state.valueMap[result] = yieldVal;
        LLVM_DEBUG(llvm::dbgs() << "  Halt yield value for process result: "
                                << (yieldVal.isX() ? "X"
                                                    : std::to_string(yieldVal.getUInt64()))
                                << "\n");
      }
    }

    // Execute module-level drives that depend on this process's results
    executeModuleDrives(procId);
    executeInstanceOutputUpdates(procId);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Process halted\n");

  finalizeProcess(procId, /*killed=*/false);

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

    if (failed(interpretOperation(procId, &op))) {
      llvm::errs() << "circt-sim: Failed in while condition for process "
                   << procId << "\n";
      llvm::errs() << "  Operation: ";
      op.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
      llvm::errs() << "\n";
      llvm::errs() << "  Location: " << op.getLoc() << "\n";
      return failure();
    }
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

    if (failed(interpretOperation(procId, &op))) {
      llvm::errs() << "circt-sim: Failed in region for process " << procId
                   << "\n";
      llvm::errs() << "  Operation: ";
      op.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
      llvm::errs() << "\n";
      llvm::errs() << "  Location: " << op.getLoc() << "\n";
      return failure();
    }
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

  // Track UVM root construction for re-entrancy handling
  // When m_uvm_get_root is called, we need to mark root construction as started
  // so that re-entrant calls (via uvm_component::new -> get_root) can skip
  // the m_inst != uvm_top comparison that fails during construction.
  StringRef calleeName = callOp.getCallee();

  // Handle process::self() - both the old stub (@self) and the runtime function.
  // Old compilations of circt-verilog generated a stub @self() that returns null.
  // Intercept it here to return the actual process handle, fixing UVM's
  // "run_test() invoked from a non process context" error.
  if (calleeName == "self" && callOp.getNumOperands() == 0 &&
      callOp.getNumResults() == 1 &&
      isa<LLVM::LLVMPointerType>(callOp.getResult(0).getType())) {
    auto &state = processStates[procId];
    void *processHandle = &state;
    uint64_t handleVal = reinterpret_cast<uint64_t>(processHandle);
    LLVM_DEBUG(llvm::dbgs() << "  func.call @self(): returning process handle 0x"
                            << llvm::format_hex(handleVal, 16) << "\n");
    setValue(procId, callOp.getResult(0), InterpretedValue(APInt(64, handleVal)));
    return success();
  }

  bool isGetRoot = calleeName == "m_uvm_get_root";
  if (isGetRoot) {
    ++uvmGetRootDepth;
    if (uvmGetRootDepth == 1) {
      // First call - mark root construction as starting
      __moore_uvm_root_constructing_start();
      LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root entry (depth=1), "
                              << "marking root construction started\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root re-entry (depth="
                              << uvmGetRootDepth << ")\n");
    }
  }

  // Use RAII to ensure depth is decremented even on early returns
  auto decrementDepthOnExit = llvm::make_scope_exit([&]() {
    if (isGetRoot) {
      --uvmGetRootDepth;
      if (uvmGetRootDepth == 0) {
        // Last call completed - mark root construction as finished
        __moore_uvm_root_constructing_end();
        LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root exit (depth=0), "
                                << "marking root construction ended\n");
      }
    }
  });

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

  // Function result caching for hot UVM phase traversal functions.
  // These functions are pure (no side effects on global state) and called
  // thousands of times with the same args during phase graph construction.
  // Caching their results avoids re-executing expensive DFS traversals.
  bool isCacheableFunc = false;
  uint64_t cacheArgHash = 0;
  {
    llvm::StringRef calleeName = callOp.getCallee();
    // Cache functions involved in UVM phase graph traversal.
    // These are called repeatedly with the same args during get_common_domain()
    // and uvm_component::new, and their results only change when add() modifies
    // the graph. Invalidation happens in the add() handler below.
    if (calleeName.contains("uvm_phase::")) {
      if (calleeName.contains("get_schedule") ||
          calleeName.contains("get_domain") ||
          calleeName.contains("get_phase_type") ||
          calleeName.contains("::find") ||
          calleeName.contains("m_find_successor") ||
          calleeName.contains("m_find_predecessor") ||
          calleeName.contains("m_find_successor_by_name") ||
          calleeName.contains("m_find_predecessor_by_name")) {
        isCacheableFunc = true;
      }
    }
    if (isCacheableFunc) {
      // Compute hash of all argument values
      cacheArgHash = 0x517cc1b727220a95ULL; // seed
      for (const auto &arg : args) {
        uint64_t v = arg.isX() ? 0xDEADBEEFULL : arg.getUInt64();
        cacheArgHash = cacheArgHash * 0x9e3779b97f4a7c15ULL + v;
      }
      // Check cache
      auto &cacheState = processStates[procId];
      auto funcIt = cacheState.funcResultCache.find(funcOp.getOperation());
      if (funcIt != cacheState.funcResultCache.end()) {
        auto argIt = funcIt->second.find(cacheArgHash);
        if (argIt != funcIt->second.end()) {
          // Cache hit - return cached results
          const auto &cachedResults = argIt->second;
          auto results = callOp.getResults();
          for (auto [result, cached] : llvm::zip(results, cachedResults)) {
            setValue(procId, result, cached);
          }
          ++cacheState.funcCacheHits;
          LLVM_DEBUG(llvm::dbgs()
                     << "  func.call: cache hit for '" << calleeName
                     << "' (hits=" << cacheState.funcCacheHits << ")\n");
          return success();
        }
      }
    }
  }

  // Invalidate function result cache when uvm_phase::add is called,
  // since it modifies the phase graph (successor/predecessor relationships).
  if (callOp.getCallee().contains("uvm_phase::add")) {
    auto &cacheState = processStates[procId];
    if (!cacheState.funcResultCache.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  func.call: invalidating func result cache ("
                 << cacheState.funcResultCache.size()
                 << " functions cached) due to phase::add\n");
      cacheState.funcResultCache.clear();
    }
  }

  // Check call depth to prevent stack overflow from deep recursion (UVM patterns)
  auto &state = processStates[procId];
  constexpr size_t maxCallDepth = 200;
  if (state.callDepth >= maxCallDepth) {
    LLVM_DEBUG(llvm::dbgs() << "  func.call: max call depth (" << maxCallDepth
                            << ") exceeded, returning zero\n");
    for (Value result : callOp.getResults()) {
      unsigned width = getTypeWidth(result.getType());
      setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
    }
    return success();
  }

  // Recursive DFS cycle detection: when a function calls itself (directly or
  // via mutual recursion), track the arg0 (`this` pointer) to prevent
  // exponential blowup from DFS traversal over graph diamonds. UVM's
  // m_find_successor iterates successors and recurses without a visited set,
  // causing O(2^N) revisits on diamond patterns in the phase DAG.
  Operation *funcKey = funcOp.getOperation();
  uint64_t arg0Val = 0;
  bool hasArg0 = !args.empty() && !args[0].isX();
  if (hasArg0)
    arg0Val = args[0].getUInt64();

  // Check if recursion depth exceeded for this (func, arg0) pair
  constexpr unsigned maxRecursionDepth = 20;
  auto &depthMap = state.recursionVisited[funcKey];
  if (hasArg0 && state.callDepth > 0) {
    unsigned &depth = depthMap[arg0Val];
    if (depth >= maxRecursionDepth) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  func.call: recursion depth " << depth
                 << " exceeded for '" << funcOp.getName() << "' with arg0=0x"
                 << llvm::format_hex(arg0Val, 16)
                 << " at callDepth " << state.callDepth << ", returning zero\n");
      for (Value result : callOp.getResults()) {
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
      }
      return success();
    }
  }

  // Increment depth counter before recursing
  bool addedToVisited = hasArg0;
  if (hasArg0) {
    ++depthMap[arg0Val];
  }

  // Execute the function body with depth tracking
  ++state.callDepth;
  llvm::SmallVector<InterpretedValue, 4> returnValues;
  // Pass the call operation so it can be saved in call stack frames
  LogicalResult funcResult =
      interpretFuncBody(procId, funcOp, args, returnValues, callOp);
  --state.callDepth;

  // Decrement depth counter after returning
  if (addedToVisited) {
    auto &depthRef = processStates[procId].recursionVisited[funcKey][arg0Val];
    if (depthRef > 0)
      --depthRef;
  }

  if (failed(funcResult))
    return failure();

  // Check if process suspended during function execution (e.g., due to wait)
  // If so, return early without setting results - the function didn't complete
  auto &postCallState = processStates[procId];
  if (postCallState.waiting) {
    LLVM_DEBUG(llvm::dbgs() << "  func.call: process suspended during call to '"
                            << callOp.getCallee() << "'\n");
    return success();
  }

  // Map return values to call results
  for (auto [result, retVal] : llvm::zip(callOp.getResults(), returnValues)) {
    setValue(procId, result, retVal);
  }

  // Store result in function cache for cacheable functions
  if (isCacheableFunc && !returnValues.empty()) {
    auto &cacheStore = processStates[procId];
    cacheStore.funcResultCache[funcOp.getOperation()][cacheArgHash] =
        llvm::SmallVector<InterpretedValue, 2>(returnValues.begin(),
                                                returnValues.end());
    LLVM_DEBUG(llvm::dbgs()
               << "  func.call: cached result for '" << callOp.getCallee()
               << "' (argHash=0x" << llvm::format_hex(cacheArgHash, 16)
               << ")\n");
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretFuncBody(
    ProcessId procId, mlir::func::FuncOp funcOp,
    llvm::ArrayRef<InterpretedValue> args,
    llvm::SmallVectorImpl<InterpretedValue> &results,
    mlir::Operation *callOp, mlir::Block *resumeBlock,
    mlir::Block::iterator resumeOp) {
  if (funcOp.getBody().empty())
    return failure();

  Block &entryBlock = funcOp.getBody().front();

  // Track temporary signal mappings created for !llhd.ref function arguments.
  // When a function receives an !llhd.ref argument (e.g., from uvm_config_db),
  // we need to propagate the signal mapping from the caller so that llhd.drv
  // and llhd.prb inside the function can resolve the signal ID.
  llvm::SmallVector<Value, 4> tempSignalMappings;

  // Track recursion depth. When a function calls itself (directly or
  // indirectly), the inner call's SSA values overwrite the outer call's in the
  // shared valueMap/memoryBlocks (since same mlir::Value objects are reused).
  // We only need to save/restore when depth > 1 (i.e., this is a recursive call).
  // Use a local key to avoid dangling reference - DenseMap may rehash when
  // a different function inserts into funcCallDepth during recursive calls.
  Operation *funcKey = funcOp.getOperation();
  unsigned currentDepth = ++funcCallDepth[funcKey];
  bool isRecursive = (currentDepth > 1);

  llvm::DenseMap<Value, InterpretedValue> savedFuncValues;
  llvm::DenseMap<Value, MemoryBlock> savedFuncMemBlocks;
  if (isRecursive) {
    auto &state = processStates[procId];
    for (Block &block : funcOp.getBody()) {
      for (auto arg : block.getArguments()) {
        auto it = state.valueMap.find(arg);
        if (it != state.valueMap.end())
          savedFuncValues[arg] = it->second;
        auto mIt = state.memoryBlocks.find(arg);
        if (mIt != state.memoryBlocks.end())
          savedFuncMemBlocks[arg] = mIt->second;
      }
      for (Operation &op : block) {
        for (auto result : op.getResults()) {
          auto it = state.valueMap.find(result);
          if (it != state.valueMap.end())
            savedFuncValues[result] = it->second;
          auto mIt = state.memoryBlocks.find(result);
          if (mIt != state.memoryBlocks.end())
            savedFuncMemBlocks[result] = mIt->second;
        }
      }
    }
  }

  // Set function arguments (only if not resuming from a saved position)
  if (!resumeBlock) {
    // Get the call operands so we can trace signal refs through function args.
    // For func.call, getOperands() returns just the arguments.
    // For func.call_indirect, the first operand is the callee; use
    // getArgOperands() to skip it and get only the function arguments.
    llvm::SmallVector<Value, 8> callOperands;
    if (callOp) {
      if (auto callIndirectOp = dyn_cast<mlir::func::CallIndirectOp>(callOp)) {
        for (Value operand : callIndirectOp.getArgOperands())
          callOperands.push_back(operand);
      } else {
        for (Value operand : callOp->getOperands())
          callOperands.push_back(operand);
      }
    }

    unsigned idx = 0;
    for (auto [arg, val] : llvm::zip(entryBlock.getArguments(), args)) {
      setValue(procId, arg, val);

      // If the argument type is !llhd.ref<...>, try to propagate signal mapping
      // from the caller. This enables llhd.drv/llhd.prb on ref arguments
      // passed through function calls (e.g., uvm_config_db::set stores values
      // into associative arrays via ref arguments).
      if (isa<llhd::RefType>(arg.getType()) && idx < callOperands.size()) {
        if (SignalId sigId = resolveSignalId(callOperands[idx])) {
          valueToSignal[arg] = sigId;
          tempSignalMappings.push_back(arg);
          LLVM_DEBUG(llvm::dbgs()
                     << "  Created temp signal mapping for func arg " << idx
                     << " -> signal " << sigId << "\n");
        }
      }
      ++idx;
    }
  }

  // Helper to restore saved function values and decrement recursion depth.
  auto restoreSavedFuncValues = [&]() {
    --funcCallDepth[funcKey];
    if (isRecursive) {
      auto &state = processStates[procId];
      for (const auto &[val, saved] : savedFuncValues)
        state.valueMap[val] = saved;
      for (auto &[val, saved] : savedFuncMemBlocks)
        state.memoryBlocks[val] = saved;
    }
  };

  // Set the current function name for progress reporting
  auto &funcState = processStates[procId];
  std::string prevFuncName = funcState.currentFuncName;
  funcState.currentFuncName = funcOp.getName().str();

  // Helper to clean up temporary signal mappings and restore values before
  // returning. Restoring values is critical for recursive calls to the same
  // function - without it, the inner call's values corrupt the outer call's.
  auto cleanupTempMappings = [&]() {
    for (Value v : tempSignalMappings)
      valueToSignal.erase(v);
    restoreSavedFuncValues();
    // Restore previous function name for progress reporting
    auto it = processStates.find(procId);
    if (it != processStates.end())
      it->second.currentFuncName = prevFuncName;
  };

  // Execute operations until we hit a return
  Block *currentBlock = resumeBlock ? resumeBlock : &entryBlock;
  size_t maxOps = 1000000; // Prevent infinite loops (totalSteps is the real limit)
  size_t opCount = 0;

  // Track if we're starting from a resume point
  bool skipToResumeOp = (resumeBlock != nullptr);

  while (currentBlock && opCount < maxOps) {
    bool tookBranch = false;  // Track if we branched to another block
    for (auto opIt = currentBlock->begin(); opIt != currentBlock->end();
         ++opIt) {
      Operation &op = *opIt;

      // If resuming, skip operations until we reach the resume point
      if (skipToResumeOp) {
        if (&op == &*resumeOp) {
          skipToResumeOp = false;
          LLVM_DEBUG(llvm::dbgs()
                     << "  Resuming function " << funcOp.getName()
                     << " from saved position\n");
        } else {
          continue;
        }
      }

      ++opCount;
      // Track func body steps in process state for global step limiting
      {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          ++stIt->second.totalSteps;
          ++stIt->second.funcBodySteps;
          // Progress report every 10M func body steps
          if (stIt->second.funcBodySteps % 10000000 == 0) {
            llvm::errs() << "[circt-sim] func progress: process " << procId
                         << " funcBodySteps=" << stIt->second.funcBodySteps
                         << " totalSteps=" << stIt->second.totalSteps
                         << " in '" << funcOp.getName() << "'"
                         << " (callDepth=" << stIt->second.callDepth << ")"
                         << " op=" << op.getName().getStringRef()
                         << "\n";
          }
          // Enforce global process step limit inside function bodies
          if (maxProcessSteps > 0 &&
              stIt->second.totalSteps > (size_t)maxProcessSteps) {
            llvm::errs()
                << "[circt-sim] ERROR(PROCESS_STEP_OVERFLOW in func): process "
                << procId << " exceeded " << maxProcessSteps
                << " total steps in function '" << funcOp.getName() << "'"
                << " (totalSteps=" << stIt->second.totalSteps << ")\n";
            stIt->second.halted = true;
            cleanupTempMappings();
            return failure();
          }
          // Periodically check for abort (timeout watchdog)
          if (stIt->second.funcBodySteps % 10000 == 0 && isAbortRequested()) {
            stIt->second.halted = true;
            cleanupTempMappings();
            if (abortCallback)
              abortCallback();
            return failure();
          }
        }
      }
      if (opCount >= maxOps) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Function reached max operations\n");
        cleanupTempMappings();
        return failure();
      }

      if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(&op)) {
        // Gather return values
        for (Value operand : returnOp.getOperands()) {
          results.push_back(getValue(procId, operand));
        }
        cleanupTempMappings();
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
        tookBranch = true;
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
        tookBranch = true;
        break;
      }
      if (failed(interpretOperation(procId, &op))) {
        llvm::errs() << "circt-sim: Failed in func body for process " << procId
                     << "\n";
        llvm::errs() << "  Function: " << funcOp.getName() << "\n";
        llvm::errs() << "  Operation: ";
        op.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
        llvm::errs() << "\n";
        llvm::errs() << "  Location: " << op.getLoc() << "\n";
        cleanupTempMappings();
        return failure();
      }

      // Check if process was halted or is waiting (e.g., by sim.terminate,
      // llvm.unreachable, moore.wait_event, or sim.fork with blocking join).
      // This is critical for UVM where wait_for_objection() contains event
      // waits that must suspend execution, and run_test() forks phase
      // execution.
      auto it = processStates.find(procId);
      if (it != processStates.end() && (it->second.halted || it->second.waiting)) {
        LLVM_DEBUG(llvm::dbgs() << "  Process halted/waiting during function body '"
                                << funcOp.getName() << "' - saving call stack frame\n");

        // If waiting (not halted), save the call stack frame so we can resume
        // from the NEXT operation after the one that caused the wait.
        if (it->second.waiting && callOp) {
          // Compute the next operation iterator
          auto nextOpIt = opIt;
          ++nextOpIt;

          // Only save if there are more operations to execute in this function
          if (nextOpIt != currentBlock->end() ||
              currentBlock != &entryBlock) {
            CallStackFrame frame(funcOp, currentBlock, nextOpIt, callOp);
            frame.args.assign(args.begin(), args.end());
            it->second.callStack.push_back(std::move(frame));
            LLVM_DEBUG(llvm::dbgs()
                       << "    Saved call frame for function '"
                       << funcOp.getName() << "' with " << args.size()
                       << " args, will resume after current op\n");
          }
        }

        cleanupTempMappings();
        return success();
      }
    }
    // If we finished the block without a branch, we're done with this block
    // Move to the next block or exit
    if (!tookBranch)
      currentBlock = nullptr;
  }

  cleanupTempMappings();
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

  // Check the cache FIRST. If a value has been explicitly set (e.g., by
  // interpretProbe when the probe operation was executed), use that cached
  // value. This is critical for patterns like posedge detection where we
  // need to compare old vs new signal values:
  //   %old = llhd.prb %sig   // executed before wait, cached
  //   llhd.wait ...
  //   %new = llhd.prb %sig   // executed after wait, gets fresh value
  //   %edge = comb.and %new, (not %old)  // needs OLD cached value for %old
  // Without this check, signal values would be re-read every time, causing
  // %old to return the current (new) value instead of the cached old value.
  auto &valueMap = it->second.valueMap;
  auto valIt = valueMap.find(value);
  if (valIt != valueMap.end())
    return valIt->second;

  // Check module-level init values (for values computed during module init)
  auto moduleIt = moduleInitValueMap.find(value);
  if (moduleIt != moduleInitValueMap.end())
    return moduleIt->second;

  // For direct signal references not in cache, read the current value.
  if (SignalId sigId = getSignalId(value)) {
    const SignalValue &sv = scheduler.getSignalValue(sigId);
    InterpretedValue iv;
    if (sv.isUnknown()) {
      if (auto encoded = getEncodedUnknownForType(value.getType()))
        iv = InterpretedValue(*encoded);
      else
        iv = InterpretedValue::makeX(getTypeWidth(value.getType()));
    } else {
      iv = InterpretedValue::fromSignalValue(sv);
    }
    valueMap[value] = iv;
    return iv;
  }

  auto instMapIt = instanceOutputMap.find(activeInstanceId);
  if (instMapIt != instanceOutputMap.end()) {
    auto instIt = instMapIt->second.find(value);
    if (instIt != instMapIt->second.end()) {
      const auto &info = instIt->second;
      ScopedInstanceContext instScope(*this, info.instanceId);
      if (info.inputMap.empty())
        return getValue(procId, info.outputValue);
      ScopedInputValueMap scope(*this, info.inputMap);
      return getValue(procId, info.outputValue);
    }
  }

  // Handle block arguments that are mapped via inputValueMap (child module
  // inputs mapped to parent values). This is needed when a child module's input
  // is used in a drive and the parent passes a process result as the input.
  if (auto arg = dyn_cast<mlir::BlockArgument>(value)) {
    Value mappedValue;
    InstanceId mappedInstance = activeInstanceId;
    if (lookupInputMapping(arg, mappedValue, mappedInstance) &&
        mappedValue != value) {
      ScopedInstanceContext scope(*this, mappedInstance);
      return getValue(procId, mappedValue);
    }
  }

  // Handle process results. When a value is the result of an llhd::ProcessOp,
  // look up the yielded value from that process's valueMap.
  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto processOp = dyn_cast<llhd::ProcessOp>(result.getOwner())) {
      ProcessId procId = InvalidProcessId;
      if (activeInstanceId != 0) {
        auto ctxIt = instanceOpToProcessId.find(activeInstanceId);
        if (ctxIt != instanceOpToProcessId.end()) {
          auto procIt = ctxIt->second.find(processOp.getOperation());
          if (procIt != ctxIt->second.end())
            procId = procIt->second;
        }
      }
      if (procId == InvalidProcessId) {
        auto procIt = opToProcessId.find(processOp.getOperation());
        if (procIt != opToProcessId.end())
          procId = procIt->second;
      }
      if (procId != InvalidProcessId) {
        auto stateIt = processStates.find(procId);
        if (stateIt != processStates.end()) {
          auto valIt = stateIt->second.valueMap.find(value);
          if (valIt != stateIt->second.valueMap.end())
            return valIt->second;
        }
      }
      // Process result not yet computed - return X
      return InterpretedValue::makeX(getTypeWidth(value.getType()));
    }
  }

  if (auto combOp = value.getDefiningOp<llhd::CombinationalOp>()) {
    llvm::SmallVector<InterpretedValue, 4> results;
    (void)evaluateCombinationalOp(combOp, results);
    auto result = dyn_cast<OpResult>(value);
    if (result && result.getResultNumber() < results.size())
      return results[result.getResultNumber()];
    return InterpretedValue::makeX(getTypeWidth(value.getType()));
  }

  // For probe operations that are NOT in the cache, do a live re-read.
  // This handles the case where a probe result is used but the probe
  // operation itself was defined outside the process (e.g., at module level).
  if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
    SignalId sigId = resolveSignalId(probeOp.getSignal());
    if (sigId != 0) {
      const SignalValue &sv = scheduler.getSignalValue(sigId);
      InterpretedValue iv = InterpretedValue::fromSignalValue(sv);
      LLVM_DEBUG(llvm::dbgs() << "  Live probe of signal " << sigId << " = "
                              << (sv.isUnknown() ? "X"
                                                  : std::to_string(sv.getValue()))
                              << "\n");
      // Cache the value for consistency within this execution
      valueMap[value] = iv;
      return iv;
    }
  }

  // Check if this is a constant defined outside the process
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    APInt constVal = constOp.getValue();
    InterpretedValue iv(constVal);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is an arith.constant defined outside the process
  if (auto arithConstOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConstOp.getValue())) {
      InterpretedValue iv(intAttr.getValue());
      valueMap[value] = iv;
      return iv;
    }
  }

  // Check if this is an llvm.mlir.constant defined outside the process
  if (auto llvmConstOp = value.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(llvmConstOp.getValue())) {
      InterpretedValue iv(intAttr.getValue());
      valueMap[value] = iv;
      return iv;
    }
  }

  // Check if this is an aggregate constant (struct/array)
  if (auto aggConstOp = value.getDefiningOp<hw::AggregateConstantOp>()) {
    APInt constVal = flattenAggregateConstant(aggConstOp);
    InterpretedValue iv(constVal);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is a bitcast operation
  if (auto bitcastOp = value.getDefiningOp<hw::BitcastOp>()) {
    InterpretedValue inputVal = getValue(procId, bitcastOp.getInput());
    unsigned outputWidth = getTypeWidth(bitcastOp.getType());
    InterpretedValue iv;
    if (inputVal.isX()) {
      iv = InterpretedValue::makeX(outputWidth);
    } else {
      APInt result = inputVal.getAPInt();
      if (result.getBitWidth() < outputWidth)
        result = result.zext(outputWidth);
      else if (result.getBitWidth() > outputWidth)
        result = result.trunc(outputWidth);
      iv = InterpretedValue(result);
    }
    valueMap[value] = iv;
    return iv;
  }

  if (auto *defOp = value.getDefiningOp()) {
    if (isa<hw::StructExtractOp, hw::StructCreateOp, hw::StructInjectOp,
            comb::XorOp, comb::AndOp, comb::OrOp, comb::ICmpOp, comb::MuxOp,
            comb::ConcatOp, comb::ExtractOp, comb::AddOp, comb::SubOp>(defOp)) {
      InterpretedValue iv = evaluateContinuousValue(value);
      valueMap[value] = iv;
      return iv;
    }
    if (defOp->getName().getStringRef() == "hw.struct_inject") {
      InterpretedValue iv = evaluateContinuousValue(value);
      valueMap[value] = iv;
      return iv;
    }
  }

  // NOTE: Probe operations are handled at the top of this function
  // to ensure they always re-read the current signal value.

  // Check if this is a constant_time operation
  if (auto constTimeOp = value.getDefiningOp<llhd::ConstantTimeOp>()) {
    // Return a placeholder - actual time conversion happens in convertTimeValue
    InterpretedValue iv(0, 64);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is an llvm.mlir.addressof operation (for vtable support)
  if (auto addrOfOp = value.getDefiningOp<LLVM::AddressOfOp>()) {
    StringRef globalName = addrOfOp.getGlobalName();
    auto addrIt = globalAddresses.find(globalName);
    if (addrIt != globalAddresses.end()) {
      uint64_t addr = addrIt->second;
      InterpretedValue iv(addr, 64);
      valueMap[value] = iv;
      return iv;
    }
    // Global not found - return X
    return InterpretedValue::makeX(64);
  }

  // Check if this is an llvm.mlir.zero (null pointer constant)
  if (auto zeroOp = value.getDefiningOp<LLVM::ZeroOp>()) {
    InterpretedValue iv(0, 64);
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is an llvm.mlir.undef (undefined value)
  if (auto undefOp = value.getDefiningOp<LLVM::UndefOp>()) {
    unsigned width = getTypeWidth(undefOp.getType());
    // Initialize undef to zero (safe default for building structs)
    InterpretedValue iv(APInt::getZero(width));
    valueMap[value] = iv;
    return iv;
  }

  // Check if this is an UnrealizedConversionCastOp - propagate value through
  if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!castOp.getInputs().empty()) {
      InterpretedValue inputVal = getValue(procId, castOp.getInputs()[0]);
      // Adjust width if needed
      unsigned outputWidth = getTypeWidth(value.getType());
      InterpretedValue result;
      if (inputVal.isX()) {
        result = InterpretedValue::makeX(outputWidth);
      } else if (inputVal.getWidth() == outputWidth) {
        result = inputVal;
      } else {
        APInt apVal = inputVal.getAPInt();
        if (outputWidth < apVal.getBitWidth()) {
          result = InterpretedValue(apVal.trunc(outputWidth));
        } else {
          result = InterpretedValue(apVal.zext(outputWidth));
        }
      }
      valueMap[value] = result;
      return result;
    }
  }

  // Check if this is a signal reference
  auto sigIt = valueToSignal.find(value);
  if (sigIt != valueToSignal.end()) {
    // This is a signal reference, return the signal value
    const SignalValue &sv = scheduler.getSignalValue(sigIt->second);
    return InterpretedValue::fromSignalValue(sv);
  }

  // Check if this is an llvm.getelementptr operation that needs on-demand
  // evaluation. This is needed for memory event tracing in moore.wait_event
  // where GEP operations in the wait body haven't been executed yet.
  if (auto gepOp = value.getDefiningOp<LLVM::GEPOp>()) {
    // Get the base pointer value (may recursively evaluate)
    InterpretedValue baseVal = getValue(procId, gepOp.getBase());
    if (baseVal.isX()) {
      InterpretedValue result = InterpretedValue::makeX(64);
      valueMap[value] = result;
      return result;
    }

    uint64_t baseAddr = baseVal.getUInt64();
    uint64_t offset = 0;

    // Get the element type
    Type elemType = gepOp.getElemType();

    // Process indices using the GEPIndicesAdaptor
    auto indices = gepOp.getIndices();
    Type currentType = elemType;

    size_t idx = 0;
    bool hasUnknownIndex = false;
    for (auto indexValue : indices) {
      int64_t indexVal = 0;

      // Check if this is a constant index (IntegerAttr) or dynamic (Value)
      if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(indexValue)) {
        indexVal = intAttr.getInt();
      } else if (auto dynamicIdx = llvm::dyn_cast_if_present<Value>(indexValue)) {
        InterpretedValue dynVal = getValue(procId, dynamicIdx);
        if (dynVal.isX()) {
          hasUnknownIndex = true;
          break;
        }
        indexVal = static_cast<int64_t>(dynVal.getUInt64());
      }

      if (idx == 0) {
        // First index: scales by the size of the pointed-to type
        offset += indexVal * getLLVMTypeSize(elemType);
      } else if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        // Struct indexing: accumulate offsets of previous fields
        auto body = structType.getBody();
        for (int64_t i = 0; i < indexVal && static_cast<size_t>(i) < body.size(); ++i) {
          offset += getLLVMTypeSize(body[i]);
        }
        if (static_cast<size_t>(indexVal) < body.size()) {
          currentType = body[indexVal];
        }
      } else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
        // Array indexing: multiply by element size
        offset += indexVal * getLLVMTypeSize(arrayType.getElementType());
        currentType = arrayType.getElementType();
      } else {
        // For other types, treat as array of the current type
        offset += indexVal * getLLVMTypeSize(currentType);
      }
      ++idx;
    }

    if (hasUnknownIndex) {
      InterpretedValue result = InterpretedValue::makeX(64);
      valueMap[value] = result;
      return result;
    }

    uint64_t resultAddr = baseAddr + offset;
    InterpretedValue result(resultAddr, 64);
    valueMap[value] = result;

    LLVM_DEBUG(llvm::dbgs() << "  getValue GEP on-demand: base=0x"
                            << llvm::format_hex(baseAddr, 16) << " offset="
                            << offset << " result=0x"
                            << llvm::format_hex(resultAddr, 16) << "\n");

    return result;
  }

  // Check if this is an llvm.load operation that needs on-demand evaluation.
  // This is needed for memory event tracing in moore.wait_event where the
  // base pointer might come from a class instance loaded from memory.
  if (auto loadOp = value.getDefiningOp<LLVM::LoadOp>()) {
    // Get the pointer value (may recursively evaluate GEPs)
    InterpretedValue ptrVal = getValue(procId, loadOp.getAddr());
    if (ptrVal.isX()) {
      unsigned bitWidth = getTypeWidth(loadOp.getType());
      InterpretedValue result = InterpretedValue::makeX(bitWidth);
      valueMap[value] = result;
      return result;
    }

    uint64_t addr = ptrVal.getUInt64();
    Type resultType = loadOp.getType();
    unsigned bitWidth = getTypeWidth(resultType);
    unsigned loadSize = getLLVMTypeSize(resultType);

    // Find the memory block containing this address
    MemoryBlock *block = nullptr;
    uint64_t offset = 0;

    // Check global memory blocks
    for (auto &entry : globalAddresses) {
      StringRef globalName = entry.first();
      uint64_t globalBaseAddr = entry.second;
      auto blockIt = globalMemoryBlocks.find(globalName);
      if (blockIt != globalMemoryBlocks.end()) {
        uint64_t globalSize = blockIt->second.size;
        if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
          block = &blockIt->second;
          offset = addr - globalBaseAddr;
          break;
        }
      }
    }

    // Check malloc blocks
    if (!block) {
      for (auto &entry : mallocBlocks) {
        uint64_t mallocBaseAddr = entry.first;
        uint64_t mallocSize = entry.second.size;
        if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
          block = &entry.second;
          offset = addr - mallocBaseAddr;
          break;
        }
      }
    }

    // Check module-level allocas
    if (!block) {
      for (auto &[val, memBlock] : moduleLevelAllocas) {
        auto addrIt = moduleInitValueMap.find(val);
        if (addrIt != moduleInitValueMap.end()) {
          uint64_t blockAddr = addrIt->second.getUInt64();
          if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
            block = &memBlock;
            offset = addr - blockAddr;
            break;
          }
        }
      }
    }

    // Check process-local memory blocks
    if (!block) {
      auto &procState = processStates[procId];
      for (auto &[val, memBlock] : procState.memoryBlocks) {
        auto addrIt = procState.valueMap.find(val);
        if (addrIt != procState.valueMap.end()) {
          uint64_t blockAddr = addrIt->second.getUInt64();
          if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
            block = &memBlock;
            offset = addr - blockAddr;
            break;
          }
        }
      }
    }

    if (block && block->initialized && offset + loadSize <= block->size) {
      // Read bytes from memory
      APInt loadedValue(bitWidth, 0);
      for (unsigned i = 0; i < loadSize && i < (bitWidth + 7) / 8; ++i) {
        uint64_t byte = block->data[offset + i];
        loadedValue |= APInt(bitWidth, byte) << (i * 8);
      }
      InterpretedValue result(loadedValue);
      valueMap[value] = result;

      LLVM_DEBUG(llvm::dbgs() << "  getValue load on-demand: addr=0x"
                              << llvm::format_hex(addr, 16) << " value="
                              << loadedValue.getZExtValue() << "\n");

      return result;
    }

    // Memory not found or not initialized - return X
    InterpretedValue result = InterpretedValue::makeX(bitWidth);
    valueMap[value] = result;
    return result;
  }

  // Check if this is an LLVM call at module level (e.g., string conversion)
  // Execute the call on demand when its result is needed.
  if (auto callOp = value.getDefiningOp<LLVM::CallOp>()) {
    // Check call depth to prevent stack overflow from deep recursion (UVM patterns)
    // This path can recurse: getValue -> interpretLLVMCall -> interpretLLVMFuncBody
    // -> interpretOperation -> getValue
    auto &state = processStates[procId];
    constexpr size_t maxCallDepth = 200;
    if (state.callDepth >= maxCallDepth) {
      LLVM_DEBUG(llvm::dbgs() << "  getValue: max call depth (" << maxCallDepth
                              << ") exceeded for LLVM call, returning zero\n");
      return InterpretedValue(
          llvm::APInt(getTypeWidth(value.getType()), 0));
    }

    // Interpret the call to compute and cache its result with depth tracking
    ++state.callDepth;
    LogicalResult callResult = interpretLLVMCall(procId, callOp);
    --state.callDepth;

    if (succeeded(callResult)) {
      // Now try to get the cached result
      auto it = valueMap.find(value);
      if (it != valueMap.end())
        return it->second;
    }
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

  // Handle LLVM pointer types (64 bits)
  if (isa<LLVM::LLVMPointerType>(type))
    return 64;

  // Handle LLVM struct types
  if (auto llvmStructType = dyn_cast<LLVM::LLVMStructType>(type)) {
    unsigned totalWidth = 0;
    for (Type elemType : llvmStructType.getBody())
      totalWidth += getTypeWidth(elemType);
    return totalWidth;
  }

  // Handle LLVM array types
  if (auto llvmArrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return getTypeWidth(llvmArrayType.getElementType()) *
           llvmArrayType.getNumElements();

  // Handle function types (for function pointers/indirect calls)
  if (isa<FunctionType>(type))
    return 64;

  // Default to 1 bit for unknown types
  return 1;
}

SignalEncoding LLHDProcessInterpreter::getSignalEncoding(Type type) {
  if (auto refType = dyn_cast<llhd::RefType>(type))
    return getSignalEncoding(refType.getNestedType());
  if (isa<seq::ClockType>(type))
    return SignalEncoding::TwoState;
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    auto elements = structType.getElements();
    if (elements.size() == 2 &&
        elements[0].name.getValue() == "value" &&
        elements[1].name.getValue() == "unknown") {
      unsigned valueWidth = getTypeWidth(elements[0].type);
      unsigned unknownWidth = getTypeWidth(elements[1].type);
      if (valueWidth == unknownWidth)
        return SignalEncoding::FourStateStruct;
    }
  }
  return SignalEncoding::TwoState;
}

//===----------------------------------------------------------------------===//
// Sim Dialect Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult
LLHDProcessInterpreter::interpretProcPrint(ProcessId procId,
                                            sim::PrintFormattedProcOp printOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.proc.print\n");

  // Evaluate the format string and print it
  std::string output = evaluateFormatString(procId, printOp.getInput());

  // Print to stdout
  llvm::outs() << output;
  llvm::outs().flush();

  return success();
}

std::string LLHDProcessInterpreter::evaluateFormatString(ProcessId procId,
                                                          Value fmtValue) {
  Operation *defOp = fmtValue.getDefiningOp();
  if (!defOp)
    return "<unknown>";

  // Handle sim.fmt.literal - literal string
  if (auto litOp = dyn_cast<sim::FormatLiteralOp>(defOp)) {
    return litOp.getLiteral().str();
  }

  // Handle sim.fmt.concat - concatenation of format strings
  if (auto concatOp = dyn_cast<sim::FormatStringConcatOp>(defOp)) {
    std::string result;
    for (Value input : concatOp.getInputs()) {
      result += evaluateFormatString(procId, input);
    }
    return result;
  }

  // Handle sim.fmt.hex - hexadecimal integer format
  if (auto hexOp = dyn_cast<sim::FormatHexOp>(defOp)) {
    InterpretedValue val = getValue(procId, hexOp.getValue());
    if (val.isX())
      return "x";
    llvm::SmallString<32> hexStr;
    val.getAPInt().toStringUnsigned(hexStr, 16);
    if (hexOp.getIsHexUppercase()) {
      std::string upperHex = hexStr.str().upper();
      return upperHex;
    }
    return std::string(hexStr.str());
  }

  // Handle sim.fmt.dec - decimal integer format
  if (auto decOp = dyn_cast<sim::FormatDecOp>(defOp)) {
    InterpretedValue val = getValue(procId, decOp.getValue());
    if (val.isX())
      return "x";
    if (decOp.getIsSigned()) {
      return std::to_string(val.getAPInt().getSExtValue());
    }
    return std::to_string(val.getUInt64());
  }

  // Handle sim.fmt.bin - binary integer format
  if (auto binOp = dyn_cast<sim::FormatBinOp>(defOp)) {
    InterpretedValue val = getValue(procId, binOp.getValue());
    if (val.isX())
      return "x";
    llvm::SmallString<64> binStr;
    val.getAPInt().toStringUnsigned(binStr, 2);
    return std::string(binStr.str());
  }

  // Handle sim.fmt.oct - octal integer format
  if (auto octOp = dyn_cast<sim::FormatOctOp>(defOp)) {
    InterpretedValue val = getValue(procId, octOp.getValue());
    if (val.isX())
      return "x";
    llvm::SmallString<32> octStr;
    val.getAPInt().toStringUnsigned(octStr, 8);
    return std::string(octStr.str());
  }

  // Handle sim.fmt.char - character format
  if (auto charOp = dyn_cast<sim::FormatCharOp>(defOp)) {
    InterpretedValue val = getValue(procId, charOp.getValue());
    if (val.isX())
      return "?";
    char c = static_cast<char>(val.getUInt64() & 0xFF);
    return std::string(1, c);
  }

  // Handle sim.fmt.dyn_string - dynamic string
  if (auto dynStrOp = dyn_cast<sim::FormatDynStringOp>(defOp)) {
    // The dynamic string value is a struct {ptr, len}
    // Get the packed value (128-bit: ptr in lower 64, len in upper 64)
    InterpretedValue structVal = getValue(procId, dynStrOp.getValue());

    // Extract pointer and length from the packed 128-bit value
    APInt packedVal = structVal.getAPInt();
    int64_t ptrVal = 0;
    int64_t lenVal = 0;

    if (packedVal.getBitWidth() >= 128) {
      ptrVal = packedVal.extractBits(64, 0).getSExtValue();
      lenVal = packedVal.extractBits(64, 64).getSExtValue();
    } else if (packedVal.getBitWidth() >= 64) {
      // Might be a simpler representation
      ptrVal = packedVal.getSExtValue();
    }

    // Look up in our dynamic strings registry
    auto it = dynamicStrings.find(ptrVal);
    if (it != dynamicStrings.end()) {
      // Found in registry - return the string (may be empty)
      if (it->second.first && it->second.second > 0)
        return std::string(it->second.first, it->second.second);
      // Empty string
      return "";
    }

    // Try reverse address-to-global lookup for string globals
    auto globalIt = addressToGlobal.find(static_cast<uint64_t>(ptrVal));
    if (globalIt != addressToGlobal.end()) {
      std::string globalName = globalIt->second;
      auto blockIt = globalMemoryBlocks.find(globalName);
      if (blockIt != globalMemoryBlocks.end()) {
        const MemoryBlock &block = blockIt->second;
        // Use the length from the struct, or the block size if length is invalid
        size_t effectiveLen = (lenVal > 0 && static_cast<size_t>(lenVal) <= block.data.size())
                                  ? static_cast<size_t>(lenVal)
                                  : block.data.size();
        // Find null terminator if present
        size_t actualLen = 0;
        for (size_t i = 0; i < effectiveLen; ++i) {
          if (block.data[i] == 0)
            break;
          actualLen++;
        }
        if (actualLen > 0) {
          return std::string(reinterpret_cast<const char *>(block.data.data()),
                             actualLen);
        }
      }
    }

    // Fallback: try to interpret as direct pointer (unsafe, for debugging)
    if (ptrVal != 0 && lenVal > 0 && lenVal < 1024) {
      const char *ptr = reinterpret_cast<const char *>(ptrVal);
      // Safety check - only dereference if it looks valid
      if (ptr) {
        return std::string(ptr, lenVal);
      }
    }

    return "<dynamic string>";
  }

  // Handle arith.select - conditional selection of format strings
  if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
    InterpretedValue condVal = getValue(procId, selectOp.getCondition());
    if (condVal.isX()) {
      // Return some indication for X condition
      return "<X>";
    }
    bool condition = condVal.getAPInt().getBoolValue();
    Value selectedValue =
        condition ? selectOp.getTrueValue() : selectOp.getFalseValue();
    return evaluateFormatString(procId, selectedValue);
  }

  // Unknown format operation
  return "<unsupported format>";
}

LogicalResult LLHDProcessInterpreter::interpretTerminate(
    ProcessId procId, sim::TerminateOp terminateOp) {
  bool success = terminateOp.getSuccess();
  bool verbose = terminateOp.getVerbose();
  auto &state = processStates[procId];

  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.terminate ("
                          << (success ? "success" : "failure") << ", "
                          << (verbose ? "verbose" : "quiet") << ")\n");

  // Check if this process has active forked children that haven't completed.
  // If so, we cannot terminate yet - we must wait for all children to complete.
  // This is important for UVM where run_test() forks phase execution and then
  // calls $finish, but the phases should complete first.
  if (forkJoinManager.hasActiveChildren(procId)) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Terminate deferred - process has active forked children\n");

    // Suspend the process instead of terminating - it will be resumed when
    // all children complete (via the fork/join completion mechanism)
    state.waiting = true;

    // Store the terminate operation so we can re-execute it when children complete
    state.destBlock = terminateOp->getBlock();
    state.currentOp = mlir::Block::iterator(terminateOp);
    state.resumeAtCurrentOp = true; // Resume at terminate op, not block beginning

    Process *proc = scheduler.getProcess(procId);
    if (proc)
      proc->setState(ProcessState::Waiting);

    return mlir::success();
  }

  // Print diagnostic info about the termination source for debugging
  // This helps identify where fatal errors occur (e.g., UVM die() -> $finish)
  if (verbose) {
    llvm::errs() << "[circt-sim] sim.terminate triggered in process ID "
                 << procId << " at ";
    terminateOp.getLoc().print(llvm::errs());
    llvm::errs() << "\n";
  }

  // Mark termination requested
  terminationRequested = true;

  // Call the terminate callback if set
  if (terminateCallback) {
    terminateCallback(success, verbose);
  }

  // During global initialization (e.g., UVM global constructors), do not halt
  // the process. UVM's m_uvm_get_root() can be called re-entrantly during
  // uvm_root::new(), which triggers uvm_fatal -> die() -> sim.terminate.
  // If we halt here, the first call to m_uvm_get_root() never stores to
  // uvm_top, causing the m_inst != uvm_top check to fail permanently.
  // Instead, record termination was requested but let the init code finish.
  if (inGlobalInit) {
    LLVM_DEBUG(llvm::dbgs()
               << "  sim.terminate during global init - not halting process "
               << procId << " (termination deferred to after init)\n");
    return mlir::success();
  }

  finalizeProcess(procId, /*killed=*/false);

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Fork/Join Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretSimFork(ProcessId procId,
                                                        sim::SimForkOp forkOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.fork with "
                          << forkOp.getBranches().size() << " branches, join_type="
                          << forkOp.getJoinType() << "\n");


  // Parse the join type
  ForkJoinType joinType = parseForkJoinType(forkOp.getJoinType());

  // Create a fork group
  ForkId forkId = forkJoinManager.createFork(procId, joinType);

  // Store the fork ID as the handle result (used by join/disable_fork)
  setValue(procId, forkOp.getHandle(), InterpretedValue(forkId, 64));

  // Create a child process for each branch region
  for (auto [idx, branch] : llvm::enumerate(forkOp.getBranches())) {
    if (branch.empty())
      continue;

    // Generate a unique name for the child process
    std::string childName = "fork_" + std::to_string(forkId) + "_branch_" +
                            std::to_string(idx);

    // Register the child process with the scheduler
    ProcessId childId = scheduler.registerProcess(childName, []() {});

    // Create execution state for the child process
    ProcessExecutionState childState;
    childState.currentBlock = &branch.front();
    childState.currentOp = childState.currentBlock->begin();
    childState.isInitialBlock = false; // Fork branches are not initial blocks

    // Copy value mappings from parent to child for values defined outside the fork
    // This allows child processes to access parent's local variables
    auto &parentState = processStates[procId];
    childState.valueMap = parentState.valueMap;

    // Share parent-scope allocas via the parent pointer chain instead of
    // deep-copying.  Only allocas defined WITHIN the fork body region are
    // local to the child (automatic variable capture).  Parent-scope allocas
    // (e.g. loop counters, shared variables) are accessed through the parent
    // chain so that child writes are visible to the parent after join.
    //
    // Collect the set of alloca Values that are defined inside this branch.
    llvm::DenseSet<mlir::Value> forkBodyAllocas;
    branch.walk([&](LLVM::AllocaOp allocaOp) {
      forkBodyAllocas.insert(allocaOp.getResult());
    });

    // Deep-copy only fork-body-local allocas from the parent (they may have
    // been pre-populated during value map copy).  Everything else is shared.
    for (auto &[val, block] : parentState.memoryBlocks) {
      if (forkBodyAllocas.contains(val))
        childState.memoryBlocks[val] = block; // deep copy for fork-local
    }

    // Set parent pointer so that lookups fall through to the parent chain
    // for allocas not found locally.
    childState.parentProcessId = procId;

    // Copy processOrInitialOp from parent so that the child can look up functions
    // in the parent module (needed for virtual method dispatch via call_indirect)
    childState.processOrInitialOp = parentState.processOrInitialOp;

    // Store the child state
    registerProcessState(childId, std::move(childState));

    // Set up the callback to execute the child process
    if (auto *proc = scheduler.getProcess(childId))
      proc->setCallback([this, childId]() { executeProcess(childId); });

    // Add the child to the fork group
    forkJoinManager.addChildToFork(forkId, childId);

    // Schedule the child process for execution
    scheduler.scheduleProcess(childId, SchedulingRegion::Active);

    LLVM_DEBUG(llvm::dbgs() << "    Created child process " << childId
                            << " for branch " << idx << "\n");
  }
  // Handle different join types
  switch (joinType) {
  case ForkJoinType::JoinNone:
    // Parent continues immediately - no waiting
    LLVM_DEBUG(llvm::dbgs() << "    join_none: parent continues immediately\n");
    return success();

  case ForkJoinType::Join:
  case ForkJoinType::JoinAny: {
    // Check if fork is already complete (e.g., no branches)
    if (forkJoinManager.join(forkId)) {
      LLVM_DEBUG(llvm::dbgs() << "    Fork already complete, continuing\n");
      return success();
    }

    // Suspend the parent process until fork completes
    auto &state = processStates[procId];
    state.waiting = true;

    // The ForkJoinManager will resume the parent when appropriate
    // (all children for join, any child for join_any)
    Process *parentProc = scheduler.getProcess(procId);
    if (parentProc)
      parentProc->setState(ProcessState::Waiting);

    LLVM_DEBUG(llvm::dbgs() << "    Parent suspended waiting for fork\n");
    return success();
  }
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSimForkTerminator(
    ProcessId procId, sim::SimForkTerminatorOp termOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.fork.terminator\n");

  finalizeProcess(procId, /*killed=*/false);

  LLVM_DEBUG(llvm::dbgs() << "    Fork branch " << procId << " completed\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSimJoin(ProcessId procId,
                                                        sim::SimJoinOp joinOp) {
  // Get the fork handle
  InterpretedValue handleVal = getValue(procId, joinOp.getHandle());
  ForkId forkId = static_cast<ForkId>(handleVal.getUInt64());

  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.join for fork " << forkId << "\n");

  // Check if the fork is complete
  if (forkJoinManager.join(forkId)) {
    LLVM_DEBUG(llvm::dbgs() << "    Fork already complete\n");
    return success();
  }

  // Suspend the process until fork completes
  auto &state = processStates[procId];
  state.waiting = true;

  Process *proc = scheduler.getProcess(procId);
  if (proc)
    proc->setState(ProcessState::Waiting);

  LLVM_DEBUG(llvm::dbgs() << "    Process suspended waiting for fork to complete\n");
  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSimJoinAny(
    ProcessId procId, sim::SimJoinAnyOp joinAnyOp) {
  // Get the fork handle
  InterpretedValue handleVal = getValue(procId, joinAnyOp.getHandle());
  ForkId forkId = static_cast<ForkId>(handleVal.getUInt64());

  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.join_any for fork " << forkId << "\n");

  // Check if any child has completed
  if (forkJoinManager.joinAny(forkId)) {
    LLVM_DEBUG(llvm::dbgs() << "    At least one child already complete\n");
    return success();
  }

  // Suspend the process until any child completes
  auto &state = processStates[procId];
  state.waiting = true;

  Process *proc = scheduler.getProcess(procId);
  if (proc)
    proc->setState(ProcessState::Waiting);

  LLVM_DEBUG(llvm::dbgs() << "    Process suspended waiting for any child to complete\n");
  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSimWaitFork(
    ProcessId procId, sim::SimWaitForkOp waitForkOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.wait_fork\n");

  // Check if all child processes of this parent are complete
  if (forkJoinManager.waitFork(procId)) {
    LLVM_DEBUG(llvm::dbgs() << "    All child processes already complete\n");
    return success();
  }

  // Suspend the process until all children complete
  auto &state = processStates[procId];
  state.waiting = true;

  Process *proc = scheduler.getProcess(procId);
  if (proc)
    proc->setState(ProcessState::Waiting);

  LLVM_DEBUG(llvm::dbgs() << "    Process suspended waiting for all children\n");
  return success();
}

LogicalResult LLHDProcessInterpreter::interpretSimDisableFork(
    ProcessId procId, sim::SimDisableForkOp disableForkOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.disable_fork\n");

  // Disable all fork groups created by this process
  for (ForkId forkId : forkJoinManager.getForksForParent(procId)) {
    if (auto *group = forkJoinManager.getForkGroup(forkId)) {
      for (ProcessId childId : group->childProcesses)
        finalizeProcess(childId, /*killed=*/true);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "    All child processes disabled\n");
  return success();
}

//===----------------------------------------------------------------------===//
// Seq Dialect Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretSeqYield(ProcessId procId,
                                                         seq::YieldOp yieldOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting seq.yield - terminating initial block\n");

  // seq.yield terminates the initial block
  finalizeProcess(procId, /*killed=*/false);

  return success();
}

//===----------------------------------------------------------------------===//
// Moore Dialect Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretMooreWaitEvent(
    ProcessId procId, moore::WaitEventOp waitEventOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting moore.wait_event\n");

  auto &state = processStates[procId];

  // The moore.wait_event operation contains a body region with moore.detect_event
  // operations that specify which signal edges to detect.
  //
  // To properly implement this, we need to:
  // 1. Extract the signals to observe from detect_event ops in the body
  // 2. Set up edge detection (posedge/negedge/anychange)
  // 3. Suspend the process until one of the events fires
  //
  // For now, we implement a simplified version:
  // - Walk the body to find detect_event ops
  // - Extract the input signals they observe
  // - Wait for any change on those signals

  // Check if the body is empty - if so, just halt (infinite wait)
  if (waitEventOp.getBody().front().empty()) {
    LLVM_DEBUG(llvm::dbgs() << "    Empty wait_event body - halting process\n");
    finalizeProcess(procId, /*killed=*/false);
    return success();
  }

  // Collect signals to observe from detect_event ops
  SensitivityList waitList;

  waitEventOp.getBody().walk([&](moore::DetectEventOp detectOp) {
    Value input = detectOp.getInput();

    // Try to trace the input to a signal
    std::function<SignalId(Value, int)> traceToSignal = [&](Value value,
                                                             int depth) -> SignalId {
      if (depth > 10)
        return 0; // Prevent infinite recursion

      // Check if this value is a signal reference
      SignalId sigId = getSignalId(value);
      if (sigId != 0)
        return sigId;

      // Try to trace through the defining operation
      if (Operation *defOp = value.getDefiningOp()) {
        // For probe operations, get the signal being probed
        if (auto probeOp = dyn_cast<llhd::ProbeOp>(defOp)) {
          return getSignalId(probeOp.getSignal());
        }

        // For struct extract, trace the struct
        if (auto extractOp = dyn_cast<hw::StructExtractOp>(defOp)) {
          return traceToSignal(extractOp.getInput(), depth + 1);
        }

        // For LLVM GEP, trace the base
        if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOp)) {
          return traceToSignal(gepOp.getBase(), depth + 1);
        }

        // For llhd.sig.struct_extract, trace to the signal
        if (auto sigExtractOp = dyn_cast<llhd::SigStructExtractOp>(defOp)) {
          return traceToSignal(sigExtractOp.getInput(), depth + 1);
        }

        // For unrealized_conversion_cast, trace through
        if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
          if (!castOp.getInputs().empty()) {
            return traceToSignal(castOp.getInputs()[0], depth + 1);
          }
        }

        // For other operations, try to trace their operands
        for (Value operand : defOp->getOperands()) {
          sigId = traceToSignal(operand, depth + 1);
          if (sigId != 0)
            return sigId;
        }
      }

      return 0;
    };

    SignalId sigId = traceToSignal(input, 0);
    if (sigId != 0) {
      waitList.addLevel(sigId);
      LLVM_DEBUG(llvm::dbgs() << "    Waiting on signal " << sigId
                              << " for edge type " << (int)detectOp.getEdge()
                              << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: Could not trace detect_event "
                                 "input to signal\n");
    }
  });

  // If we found signals to wait on, suspend the process
  if (!waitList.empty()) {
    state.waiting = true;

    // The continuation point after the wait_event is the next operation
    // in the current block (which was already advanced by the main loop)
    // We don't need to set destBlock since we continue sequentially

    // Register the wait sensitivity with the scheduler
    scheduler.suspendProcessForEvents(procId, waitList);

    LLVM_DEBUG(llvm::dbgs() << "    Process suspended waiting on "
                            << waitList.size() << " signals\n");
  } else {
    // No signals found - try to set up memory-based event polling.
    // This is needed for UVM events stored as boolean fields in class instances.
    //
    // The pattern we're looking for:
    //   %ptr = llvm.getelementptr %uvm_obj[0, N] : (!llvm.ptr) -> !llvm.ptr
    //   %val = llvm.load %ptr : !llvm.ptr -> i1
    //   %evt = builtin.unrealized_conversion_cast %val : i1 to !moore.event
    //   moore.detect_event any %evt : event
    //
    // We trace backwards from detect_event input to find the memory pointer.
    bool foundMemoryEvent = false;

    // Track the detected input type to determine edge behavior
    bool isEventType = false;
    waitEventOp.getBody().walk([&](moore::DetectEventOp detectOp) {
      if (foundMemoryEvent)
        return; // Already found one

      Value input = detectOp.getInput();

      // Check if the input is an event type (!moore.event).
      // For event types, we need rising edge detection (0→1 trigger).
      Type inputType = input.getType();
      if (isa<moore::EventType>(inputType)) {
        isEventType = true;
      }

      // Trace through unrealized_conversion_cast to find llvm.load
      std::function<Value(Value, int)> traceToMemoryPtr =
          [&](Value value, int depth) -> Value {
        if (depth > 10)
          return nullptr;

        if (Operation *defOp = value.getDefiningOp()) {
          // If this is a load, return its address operand
          if (auto loadOp = dyn_cast<LLVM::LoadOp>(defOp)) {
            return loadOp.getAddr();
          }

          // Trace through unrealized_conversion_cast
          if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
            if (!castOp.getInputs().empty()) {
              return traceToMemoryPtr(castOp.getInputs()[0], depth + 1);
            }
          }
        }

        return nullptr;
      };

      Value memPtr = traceToMemoryPtr(input, 0);
      if (!memPtr)
        return;

      // Get the address value for this pointer
      InterpretedValue ptrVal = getValue(procId, memPtr);
      if (ptrVal.isX()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Memory event pointer is X, cannot poll\n");
        return;
      }

      uint64_t addr = ptrVal.getUInt64();
      if (addr == 0) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Memory event pointer is null, cannot poll\n");
        return;
      }

      // Find the memory block and read current value
      MemoryBlock *block = nullptr;
      uint64_t offset = 0;

      // First try module-level allocas (accessible from all processes)
      // Check by value directly first
      auto moduleLevelIt = moduleLevelAllocas.find(memPtr);
      if (moduleLevelIt != moduleLevelAllocas.end()) {
        block = &moduleLevelIt->second;
        offset = 0;
      }

      // If not found by value, check by address
      if (!block) {
        for (auto &[val, memBlock] : moduleLevelAllocas) {
          auto addrIt = moduleInitValueMap.find(val);
          if (addrIt != moduleInitValueMap.end()) {
            uint64_t blockAddr = addrIt->second.getUInt64();
            if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
              block = &memBlock;
              offset = addr - blockAddr;
              break;
            }
          }
        }
      }

      // Check malloc blocks
      if (!block) {
        for (auto &entry : mallocBlocks) {
          uint64_t mallocBaseAddr = entry.first;
          uint64_t mallocSize = entry.second.size;
          if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
            block = &entry.second;
            offset = addr - mallocBaseAddr;
            break;
          }
        }
      }

      // Check global memory blocks
      if (!block) {
        for (auto &entry : globalAddresses) {
          StringRef globalName = entry.first();
          uint64_t globalBaseAddr = entry.second;
          auto blockIt = globalMemoryBlocks.find(globalName);
          if (blockIt != globalMemoryBlocks.end()) {
            uint64_t globalSize = blockIt->second.size;
            if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
              block = &blockIt->second;
              offset = addr - globalBaseAddr;
              break;
            }
          }
        }
      }

      // Also check process-local memory blocks
      if (!block) {
        auto &procState = processStates[procId];
        for (auto &[val, memBlock] : procState.memoryBlocks) {
          auto addrIt = procState.valueMap.find(val);
          if (addrIt != procState.valueMap.end()) {
            uint64_t blockAddr = addrIt->second.getUInt64();
            if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
              block = &memBlock;
              offset = addr - blockAddr;
              break;
            }
          }
        }
      }

      if (!block || !block->initialized) {
        LLVM_DEBUG(llvm::dbgs() << "    Memory block not found or not "
                                   "initialized for address 0x"
                                << llvm::format_hex(addr, 16) << "\n");
        return;
      }

      // Read the current value (assume 1 byte for boolean/event)
      unsigned valueSize = 1;
      if (offset + valueSize > block->size)
        return;

      uint64_t currentValue = 0;
      for (unsigned i = 0; i < valueSize; ++i) {
        currentValue |= static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
      }

      // Set up the memory event waiter
      MemoryEventWaiter waiter;
      waiter.address = addr;
      waiter.lastValue = currentValue;
      waiter.valueSize = valueSize;
      // For event types, use rising edge detection (0→1).
      // This is critical for UVM wait_for_objection: if no objection has been
      // raised yet (value=0), we must wait for the NEXT trigger (0→1),
      // not wake up immediately or on any change.
      waiter.waitForRisingEdge = isEventType;
      memoryEventWaiters[procId] = waiter;

      LLVM_DEBUG(llvm::dbgs() << "    Set up memory event waiter for address 0x"
                              << llvm::format_hex(addr, 16)
                              << " with initial value " << currentValue
                              << (isEventType ? " (rising edge mode for event type)" : "") << "\n");

      foundMemoryEvent = true;
    });

    if (foundMemoryEvent) {
      // Suspend the process - it will be woken when memory is written
      state.waiting = true;
      // Don't schedule - checkMemoryEventWaiters() will be called when
      // llvm.store writes to memory, and will wake this process if needed
    } else {
      // No signals and no memory events found - single delta cycle wait
      LLVM_DEBUG(llvm::dbgs() << "    Warning: No signals or memory events "
                                 "found in wait_event, doing single delta "
                                 "wait\n");
      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM Dialect Operation Handlers
//===----------------------------------------------------------------------===//

unsigned LLHDProcessInterpreter::getLLVMTypeAlignment(Type type) {
  // Note: This function exists for future use. Currently the interpreter
  // uses unaligned struct layout to match MooreToCore's sizeof computation
  // (which sums field sizes without alignment padding).
  if (isa<LLVM::LLVMPointerType>(type))
    return 8;
  if (auto intType = dyn_cast<IntegerType>(type)) {
    unsigned bytes = (intType.getWidth() + 7) / 8;
    if (bytes <= 1) return 1;
    if (bytes <= 2) return 2;
    if (bytes <= 4) return 4;
    return 8;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    unsigned maxAlign = 1;
    for (Type field : structType.getBody())
      maxAlign = std::max(maxAlign, getLLVMTypeAlignment(field));
    return maxAlign;
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type))
    return getLLVMTypeAlignment(arrayType.getElementType());
  return 1;
}

unsigned LLHDProcessInterpreter::getLLVMStructFieldOffset(
    LLVM::LLVMStructType structType, unsigned fieldIndex) {
  // Use unaligned layout to match MooreToCore's sizeof computation.
  // MooreToCore computes struct sizes as the sum of field sizes without
  // alignment padding, so we must use the same layout here.
  auto body = structType.getBody();
  unsigned offset = 0;
  for (unsigned i = 0; i < fieldIndex && i < body.size(); ++i)
    offset += getLLVMTypeSize(body[i]);
  return offset;
}

unsigned LLHDProcessInterpreter::getLLVMTypeSize(Type type) {
  // For LLVM pointer types, use 64 bits (8 bytes)
  if (isa<LLVM::LLVMPointerType>(type))
    return 8;

  // For LLVM struct types, sum the sizes of all elements.
  // NOTE: This uses unaligned layout (no padding between fields) to match
  // MooreToCore's sizeof computation. MooreToCore embeds struct sizes as
  // constants in malloc/alloca calls without alignment padding.
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    unsigned size = 0;
    for (Type elemType : structType.getBody()) {
      size += getLLVMTypeSize(elemType);
    }
    return size;
  }

  // For LLVM array types, multiply element size by count
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    return getLLVMTypeSize(arrayType.getElementType()) *
           arrayType.getNumElements();
  }

  // For integer types, round up to bytes
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;

  // Default: try to use getTypeWidth and convert to bytes
  unsigned bitWidth = getTypeWidth(type);
  return (bitWidth + 7) / 8;
}

MemoryBlock *LLHDProcessInterpreter::findMemoryBlock(ProcessId procId,
                                                      Value ptr) {
  // Unwrap GEP / bitcast first so the base pointer is used for lookup.
  if (auto gepOp = ptr.getDefiningOp<LLVM::GEPOp>())
    return findMemoryBlock(procId, gepOp.getBase());
  if (auto bitcastOp = ptr.getDefiningOp<LLVM::BitcastOp>())
    return findMemoryBlock(procId, bitcastOp.getArg());

  // Walk the process -> parent chain looking for the memory block.
  ProcessId cur = procId;
  while (cur != InvalidProcessId) {
    auto stateIt = processStates.find(cur);
    if (stateIt == processStates.end())
      break;
    auto &st = stateIt->second;
    auto it = st.memoryBlocks.find(ptr);
    if (it != st.memoryBlocks.end())
      return &it->second;
    cur = st.parentProcessId;
  }

  // Check module-level allocas (accessible by all processes)
  auto moduleIt = moduleLevelAllocas.find(ptr);
  if (moduleIt != moduleLevelAllocas.end())
    return &moduleIt->second;

  // If ptr is a function entry block argument (i.e., a pointer passed through
  // a function call), the SSA Value differs from the original alloca result.
  // Fall back to address-based lookup for this specific case.
  if (auto blockArg = dyn_cast<BlockArgument>(ptr)) {
    if (blockArg.getOwner()->isEntryBlock()) {
      auto stateIt = processStates.find(procId);
      if (stateIt != processStates.end()) {
        auto valIt = stateIt->second.valueMap.find(ptr);
        if (valIt != stateIt->second.valueMap.end() &&
            !valIt->second.isX()) {
          uint64_t addr = valIt->second.getUInt64();
          return findMemoryBlockByAddress(addr, procId, nullptr);
        }
      }
    }
  }

  return nullptr;
}

MemoryBlock *LLHDProcessInterpreter::findMemoryBlockByAddress(uint64_t addr,
                                                              ProcessId procId,
                                                              uint64_t *outOffset) {
  // Walk the process -> parent chain looking for the memory block by address.
  if (procId != static_cast<ProcessId>(-1)) {
    ProcessId cur = procId;
    while (cur != InvalidProcessId) {
      auto stateIt = processStates.find(cur);
      if (stateIt == processStates.end())
        break;
      auto &state = stateIt->second;
      for (auto &[val, block] : state.memoryBlocks) {
        // Get the address assigned to this Value.
        // Check the original process's valueMap first (child copies parent
        // valueMap), then fall back to the block-owning process's valueMap.
        // NOTE: We must NOT compare iterators across different DenseMaps
        // (would trigger epoch assertion in debug builds when cur != procId).
        uint64_t blockAddr = 0;
        bool foundAddr = false;
        {
          auto &procVM = processStates[procId].valueMap;
          auto it = procVM.find(val);
          if (it != procVM.end() && !it->second.isX()) {
            blockAddr = it->second.getUInt64();
            foundAddr = true;
          }
        }
        if (!foundAddr) {
          auto it2 = state.valueMap.find(val);
          if (it2 != state.valueMap.end() && !it2->second.isX()) {
            blockAddr = it2->second.getUInt64();
            foundAddr = true;
          }
        }
        if (foundAddr && addr >= blockAddr && addr < blockAddr + block.size) {
          if (outOffset) *outOffset = addr - blockAddr;
          return &block;
        }
      }
      cur = state.parentProcessId;
    }
  }
  // Check module-level allocas
  for (auto &[val, block] : moduleLevelAllocas) {
    // Check in moduleInitValueMap for the address
    auto it = moduleInitValueMap.find(val);
    if (it != moduleInitValueMap.end()) {
      uint64_t blockAddr = it->second.getUInt64();
      if (addr >= blockAddr && addr < blockAddr + block.size) {
        if (outOffset) *outOffset = addr - blockAddr;
        return &block;
      }
    }
  }
  // Check malloc'd blocks
  for (auto &[blockAddr, block] : mallocBlocks) {
    if (addr >= blockAddr && addr < blockAddr + block.size) {
      if (outOffset) *outOffset = addr - blockAddr;
      return &block;
    }
  }
  // Check global memory blocks
  for (auto &[globalName, globalAddr] : globalAddresses) {
    auto blockIt = globalMemoryBlocks.find(globalName);
    if (blockIt != globalMemoryBlocks.end()) {
      MemoryBlock &block = blockIt->second;
      if (addr >= globalAddr && addr < globalAddr + block.size) {
        if (outOffset) *outOffset = addr - globalAddr;
        return &block;
      }
    }
  }
  if (outOffset) *outOffset = 0;
  return nullptr;
}

bool LLHDProcessInterpreter::findNativeMemoryBlockByAddress(
    uint64_t addr, uint64_t *outOffset, size_t *outSize) const {
  for (auto &entry : nativeMemoryBlocks) {
    uint64_t baseAddr = entry.first;
    size_t blockSize = entry.second;
    if (addr >= baseAddr && addr < baseAddr + static_cast<uint64_t>(blockSize)) {
      if (outOffset) *outOffset = addr - baseAddr;
      if (outSize) *outSize = blockSize;
      return true;
    }
  }
  if (outOffset) *outOffset = 0;
  if (outSize) *outSize = 0;
  return false;
}

bool LLHDProcessInterpreter::tryReadStringKey(ProcessId procId,
                                               uint64_t strPtrVal,
                                               int64_t strLen,
                                               std::string &out) {
  out.clear();
  if (strLen < 0)
    return false;
  if (strLen == 0)
    return true;
  constexpr int64_t kMaxStringKeyBytes = 1 << 20;
  if (strLen > kMaxStringKeyBytes)
    return false;

  // Check dynamicStrings registry first (for strings from __moore_packed_string_to_string)
  auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtrVal));
  if (dynIt != dynamicStrings.end() && dynIt->second.first && dynIt->second.second > 0) {
    size_t effectiveLen = std::min(static_cast<size_t>(strLen),
                                   static_cast<size_t>(dynIt->second.second));
    out.assign(dynIt->second.first, effectiveLen);
    return true;
  }

  uint64_t offset = 0;
  if (auto *block = findMemoryBlockByAddress(strPtrVal, procId, &offset)) {
    if (!block->initialized)
      return false;
    if (offset + static_cast<uint64_t>(strLen) > block->data.size())
      return false;
    out.assign(reinterpret_cast<const char *>(block->data.data() + offset),
               static_cast<size_t>(strLen));
    return true;
  }

  uint64_t nativeOffset = 0;
  size_t nativeSize = 0;

  if (findNativeMemoryBlockByAddress(strPtrVal, &nativeOffset, &nativeSize)) {
    if (nativeOffset + static_cast<size_t>(strLen) > nativeSize)
      return false;
    auto *nativePtr = reinterpret_cast<const char *>(strPtrVal);
    out.assign(nativePtr, static_cast<size_t>(strLen));
    return true;
  }

  return false;
}

LogicalResult LLHDProcessInterpreter::interpretLLVMAlloca(
    ProcessId procId, LLVM::AllocaOp allocaOp) {
  auto &state = processStates[procId];

  // Get the element type and array size
  Type elemType = allocaOp.getElemType();
  InterpretedValue arraySizeVal = getValue(procId, allocaOp.getArraySize());

  uint64_t arraySize = 1;
  if (!arraySizeVal.isX())
    arraySize = arraySizeVal.getUInt64();

  // Calculate total size in bytes
  unsigned elemSize = getLLVMTypeSize(elemType);
  size_t totalSize = elemSize * arraySize;

  // Create a memory block
  MemoryBlock block(totalSize, getTypeWidth(elemType));
  block.initialized = true;  // Alloca memory is zero-initialized and readable

  // Check if this alloca is at module level (not inside an llhd.process,
  // func.func, or llvm.func). Allocas inside functions should be process-local
  // even if the function is called from a global constructor, because they
  // need to be found via the process's valueMap when findMemoryBlockByAddress
  // is called.
  bool isModuleLevel = !allocaOp->getParentOfType<llhd::ProcessOp>() &&
                       !allocaOp->getParentOfType<mlir::func::FuncOp>() &&
                       !allocaOp->getParentOfType<LLVM::LLVMFuncOp>();

  if (isModuleLevel) {
    // Store in module-level allocas (accessible by all processes)
    moduleLevelAllocas[allocaOp.getResult()] = std::move(block);
  } else {
    // Store in process-local memory
    state.memoryBlocks[allocaOp.getResult()] = std::move(block);
  }

  // Assign a unique address to this pointer (for tracking purposes)
  // Use globalNextAddress to ensure no overlap between module-level
  // and process-level allocas.
  uint64_t addr = globalNextAddress;
  globalNextAddress += totalSize;

  // Store the pointer value (the address)
  setValue(procId, allocaOp.getResult(), InterpretedValue(addr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.alloca: allocated " << totalSize
                          << " bytes at address 0x" << llvm::format_hex(addr, 16)
                          << (isModuleLevel ? " (module level)" : "") << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMLoad(ProcessId procId,
                                                         LLVM::LoadOp loadOp) {
  // If this is a load from an llhd.ref converted to an LLVM pointer,
  // treat it as a signal probe instead of a memory read.
  if (SignalId sigId = resolveSignalId(loadOp.getAddr())) {
    InterpretedValue signalVal;
    auto pendingIt = pendingEpsilonDrives.find(sigId);
    if (pendingIt != pendingEpsilonDrives.end()) {
      signalVal = pendingIt->second;
    } else {
      const SignalValue &sv = scheduler.getSignalValue(sigId);
      signalVal = InterpretedValue::fromSignalValue(sv);
    }

    Type signalType = getSignalValueType(sigId);
    if (!signalType) {
      if (auto castOp =
              loadOp.getAddr()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          if (auto refType =
                  dyn_cast<llhd::RefType>(castOp.getInputs()[0].getType()))
            signalType = refType.getNestedType();
        }
      }
    }
    Type llvmType = loadOp.getType();
    if (!signalVal.isX() && signalType &&
        (isa<hw::StructType, hw::ArrayType>(signalType)) &&
        (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmType))) {
      APInt converted =
          convertHWToLLVMLayout(signalVal.getAPInt(), signalType, llvmType);
      signalVal = InterpretedValue(converted);
    }

    unsigned targetWidth = getTypeWidth(loadOp.getType());
    if (signalVal.isX()) {
      signalVal = InterpretedValue::makeX(targetWidth);
    } else if (signalVal.getWidth() != targetWidth) {
      APInt apVal = signalVal.getAPInt();
      if (apVal.getBitWidth() < targetWidth)
        apVal = apVal.zext(targetWidth);
      else if (apVal.getBitWidth() > targetWidth)
        apVal = apVal.trunc(targetWidth);
      signalVal = InterpretedValue(apVal);
    }

    setValue(procId, loadOp.getResult(), signalVal);
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: read signal " << sigId
                            << " (width=" << targetWidth << ")\n");
    return success();
  }

  // Get the pointer value
  InterpretedValue ptrVal = getValue(procId, loadOp.getAddr());
  Type resultType = loadOp.getType();
  unsigned bitWidth = getTypeWidth(resultType);
  unsigned loadSize = getLLVMTypeSize(resultType);

  // First try to find a local memory block (from alloca)
  MemoryBlock *block = findMemoryBlock(procId, loadOp.getAddr());
  uint64_t offset = 0;
  bool useNative = false;
  uint64_t nativeOffset = 0;
  size_t nativeSize = 0;
  if (block) {
    // Local alloca memory
    // Calculate the offset within the memory block
    if (auto gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
      InterpretedValue baseVal = getValue(procId, gepOp.getBase());
      if (!baseVal.isX() && !ptrVal.isX()) {
        offset = ptrVal.getUInt64() - baseVal.getUInt64();
      }
    }
  } else if (!ptrVal.isX()) {
    // Check if this is a global memory access
    uint64_t addr = ptrVal.getUInt64();

    // Find which global this address belongs to
    for (auto &entry : globalAddresses) {
      StringRef globalName = entry.first();
      uint64_t globalBaseAddr = entry.second;
      auto blockIt = globalMemoryBlocks.find(globalName);
      if (blockIt != globalMemoryBlocks.end()) {
        uint64_t globalSize = blockIt->second.size;
        if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
          block = &blockIt->second;
          offset = addr - globalBaseAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.load: found global '" << globalName
                                  << "' at offset " << offset << "\n");
          break;
        }
      }
    }

    // Check if this is a malloc'd memory access
    if (!block) {
      for (auto &entry : mallocBlocks) {
        uint64_t mallocBaseAddr = entry.first;
        uint64_t mallocSize = entry.second.size;
        if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
          block = &entry.second;
          offset = addr - mallocBaseAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.load: found malloc block at 0x"
                                  << llvm::format_hex(mallocBaseAddr, 16)
                                  << " offset " << offset << "\n");
          break;
        }
      }
    }
    if (!block) {
      if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize)) {
        useNative = true;
        offset = nativeOffset;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.load: found native block at 0x"
                                << llvm::format_hex(addr - nativeOffset, 16)
                                << " offset " << nativeOffset << "\n");
      }
    }
  }

  // Fallback: use comprehensive address-based search (also checks
  // process-local allocas by address, which findMemoryBlock's SSA
  // tracing may miss when the pointer was loaded from memory).
  // Also try this when useNative is set but the native block is too small
  // for the requested load size (e.g., an 8-byte assoc array slot matched
  // but we need to load a 24-byte struct from a larger malloc'd block).
  if (!block && !ptrVal.isX()) {
    unsigned loadSizeCheck = getLLVMTypeSize(loadOp.getResult().getType());
    if (!useNative || (offset + loadSizeCheck > nativeSize)) {
      uint64_t fbOffset = 0;
      auto *fbBlock = findMemoryBlockByAddress(ptrVal.getUInt64(), procId, &fbOffset);
      if (fbBlock) {
        block = fbBlock;
        offset = fbOffset;
        useNative = false;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.load: findMemoryBlockByAddress found "
                                   "block at offset " << offset << "\n");
      }
    }
  }

  // Native pointer access is only allowed for known blocks registered by the
  // runtime (e.g., associative array element refs). If a pointer is not in any
  // tracked block, return X (unknown value).

  if (!block && !useNative) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    setValue(procId, loadOp.getResult(),
             InterpretedValue::makeX(bitWidth));
    return success();
  }

  if (block) {
    if (offset + loadSize > block->size) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.load: out of bounds access (offset="
                              << offset << " size=" << loadSize
                              << " block_size=" << block->size << ")\n");
      setValue(procId, loadOp.getResult(),
               InterpretedValue::makeX(bitWidth));
      return success();
    }

    // Check if memory has been initialized
    if (!block->initialized) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.load: reading uninitialized memory\n");
      setValue(procId, loadOp.getResult(),
               InterpretedValue::makeX(bitWidth));
      return success();
    }
  } else {
    if (offset + loadSize > nativeSize) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.load: native out of bounds access (offset="
                              << offset << " size=" << loadSize
                              << " block_size=" << nativeSize << ")\n");
      setValue(procId, loadOp.getResult(),
               InterpretedValue::makeX(bitWidth));
      return success();
    }
  }

  auto readByte = [&](unsigned i) -> uint8_t {
    if (block)
      return block->data[offset + i];
    auto *nativePtr = reinterpret_cast<const uint8_t *>(ptrVal.getUInt64());
    return nativePtr[i];
  };

  // Read bytes from memory and construct the value (little-endian).
  // Clamp to the number of bytes needed for the value width to avoid
  // shifting past the APInt width when loadSize includes padding.
  unsigned bytesForValue = std::min(loadSize, (bitWidth + 7) / 8);
  uint64_t value = 0;
  for (unsigned i = 0; i < bytesForValue && i < 8; ++i) {
    value |= static_cast<uint64_t>(readByte(i)) << (i * 8);
  }

  // For values larger than 64 bits, use APInt directly
  if (bitWidth > 64) {
    APInt apValue(bitWidth, 0);
    for (unsigned i = 0; i < bytesForValue; ++i) {
      APInt byteVal(bitWidth, readByte(i));
      apValue |= byteVal.shl(i * 8);
    }
    setValue(procId, loadOp.getResult(), InterpretedValue(apValue));
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: loaded wide value ("
                            << loadSize << " bytes) from offset " << offset << "\n");
  } else {
    setValue(procId, loadOp.getResult(), InterpretedValue(value, bitWidth));
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: loaded 0x"
                            << llvm::format_hex(value, 16) << " ("
                            << loadSize << " bytes) from offset " << offset << "\n");
  }

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMStore(
    ProcessId procId, LLVM::StoreOp storeOp) {
  // If this is a store to an llhd.ref converted to an LLVM pointer,
  // treat it as a signal drive instead of a memory write.
  if (SignalId sigId = resolveSignalId(storeOp.getAddr())) {
    InterpretedValue storeVal = getValue(procId, storeOp.getValue());
    const SignalValue &current = scheduler.getSignalValue(sigId);
    unsigned targetWidth = current.getWidth();

    Type signalType = getSignalValueType(sigId);
    if (!signalType) {
      if (auto castOp =
              storeOp.getAddr()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (castOp.getInputs().size() == 1) {
          if (auto refType =
                  dyn_cast<llhd::RefType>(castOp.getInputs()[0].getType()))
            signalType = refType.getNestedType();
        }
      }
    }
    Type llvmType = storeOp.getValue().getType();
    if (!storeVal.isX() && signalType &&
        (isa<hw::StructType, hw::ArrayType>(signalType)) &&
        (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(llvmType))) {
      APInt converted =
          convertLLVMToHWLayout(storeVal.getAPInt(), llvmType, signalType);
      storeVal = InterpretedValue(converted);
    }

    if (storeVal.isX()) {
      storeVal = InterpretedValue::makeX(targetWidth);
    } else if (storeVal.getWidth() != targetWidth) {
      APInt apVal = storeVal.getAPInt();
      if (apVal.getBitWidth() < targetWidth)
        apVal = apVal.zext(targetWidth);
      else if (apVal.getBitWidth() > targetWidth)
        apVal = apVal.trunc(targetWidth);
      storeVal = InterpretedValue(apVal);
    }

    pendingEpsilonDrives[sigId] = storeVal;
    scheduler.updateSignal(sigId, storeVal.toSignalValue());
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: wrote signal " << sigId
                            << " (width=" << targetWidth << ")\n");
    return success();
  }

  // Signal not resolved - going to memory
  // Get the pointer value first
  InterpretedValue ptrVal = getValue(procId, storeOp.getAddr());

  // Find the memory block for this pointer
  MemoryBlock *block = findMemoryBlock(procId, storeOp.getAddr());
  uint64_t offset = 0;
  bool useNative = false;
  uint64_t nativeOffset = 0;
  size_t nativeSize = 0;

  if (block) {
    // Local alloca memory
    // Calculate the offset within the memory block
    if (auto gepOp = storeOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
      InterpretedValue baseVal = getValue(procId, gepOp.getBase());
      if (!baseVal.isX() && !ptrVal.isX()) {
        offset = ptrVal.getUInt64() - baseVal.getUInt64();
      }
    }
  } else if (!ptrVal.isX()) {
    // Check if this is a global memory access
    uint64_t addr = ptrVal.getUInt64();

    // Find which global this address belongs to
    for (auto &entry : globalAddresses) {
      StringRef globalName = entry.first();
      uint64_t globalBaseAddr = entry.second;
      auto blockIt = globalMemoryBlocks.find(globalName);
      if (blockIt != globalMemoryBlocks.end()) {
        uint64_t globalSize = blockIt->second.size;
        if (addr >= globalBaseAddr && addr < globalBaseAddr + globalSize) {
          block = &blockIt->second;
          offset = addr - globalBaseAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found global '" << globalName
                                  << "' at offset " << offset << "\n");
          break;
        }
      }
    }

    // Check if this is a malloc'd memory access
    if (!block) {
      for (auto &entry : mallocBlocks) {
        uint64_t mallocBaseAddr = entry.first;
        uint64_t mallocSize = entry.second.size;
        if (addr >= mallocBaseAddr && addr < mallocBaseAddr + mallocSize) {
          block = &entry.second;
          offset = addr - mallocBaseAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found malloc block at 0x"
                                  << llvm::format_hex(mallocBaseAddr, 16)
                                  << " offset " << offset << "\n");
          break;
        }
      }
    }

    // Check module-level allocas by address. This is needed when the store
    // address is computed via a GEP on a pointer loaded from memory (e.g.,
    // class member access through a heap-allocated class instance).
    if (!block) {
      for (auto &[val, memBlock] : moduleLevelAllocas) {
        auto addrIt = moduleInitValueMap.find(val);
        if (addrIt != moduleInitValueMap.end()) {
          uint64_t blockAddr = addrIt->second.getUInt64();
          if (addr >= blockAddr && addr < blockAddr + memBlock.size) {
            block = &memBlock;
            offset = addr - blockAddr;
            LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found module-level alloca at 0x"
                                    << llvm::format_hex(blockAddr, 16)
                                    << " offset " << offset << "\n");
            break;
          }
        }
      }
    }

    if (!block) {
      if (findNativeMemoryBlockByAddress(addr, &nativeOffset, &nativeSize)) {
        useNative = true;
        offset = nativeOffset;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.store: found native block at 0x"
                                << llvm::format_hex(addr - nativeOffset, 16)
                                << " offset " << nativeOffset << "\n");
      }
    }
  }

  // Get the value to store
  InterpretedValue storeVal = getValue(procId, storeOp.getValue());
  unsigned storeSize = getLLVMTypeSize(storeOp.getValue().getType());

  // Fallback: use comprehensive address-based search (also checks
  // process-local allocas by address, which findMemoryBlock's SSA
  // tracing may miss when the pointer was loaded from memory).
  // Also try this when useNative is set but the native block is too small
  // for the requested store size (e.g., an 8-byte assoc array slot matched
  // but we need to store a 24-byte struct into a larger malloc'd block).
  if (!block && !ptrVal.isX()) {
    if (!useNative || (offset + storeSize > nativeSize)) {
      uint64_t fbOffset = 0;
      auto *fbBlock = findMemoryBlockByAddress(ptrVal.getUInt64(), procId, &fbOffset);
      if (fbBlock) {
        block = fbBlock;
        offset = fbOffset;
        useNative = false;
        LLVM_DEBUG(llvm::dbgs() << "  llvm.store: findMemoryBlockByAddress found "
                                   "block at offset " << offset << "\n");
      }
    }
  }

  // Native pointer access is only allowed for known blocks registered by the
  // runtime. If pointer is not tracked, the store is silently skipped (stores
  // to X are no-ops anyway).

  if (!block && !useNative) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    return success(); // Don't fail, just skip the store
  }

  if (block) {
    if (offset + storeSize > block->size) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.store: out of bounds access\n");
      return success();
    }
  } else {
    if (offset + storeSize > nativeSize) {
      llvm::errs() << "[circt-sim] native store OOB: "
                   << storeSize << " bytes at offset " << offset
                   << " exceeds block size " << nativeSize
                   << " (addr 0x" << llvm::format_hex(ptrVal.getUInt64(), 16) << ")\n";
      return success();
    }
  }

  // Write bytes to memory (little-endian)
  if (!storeVal.isX()) {
    const APInt &apValue = storeVal.getAPInt();
    auto storeByte = [&](unsigned i, uint8_t value) {
      if (block)
        block->data[offset + i] = value;
      else
        reinterpret_cast<uint8_t *>(ptrVal.getUInt64())[i] = value;
    };
    if (apValue.getBitWidth() > 64) {
      // Handle wide values using APInt operations
      unsigned bitWidth = apValue.getBitWidth();
      for (unsigned i = 0; i < storeSize; ++i) {
        unsigned bitPos = i * 8;
        if (bitPos >= bitWidth) {
          // Beyond the value's bit width - store zero
          storeByte(i, 0);
        } else {
          // Extract available bits (up to 8), remaining bits are zero
          unsigned bitsAvailable = std::min(8u, bitWidth - bitPos);
          uint8_t byteVal = static_cast<uint8_t>(
              apValue.extractBits(bitsAvailable, bitPos).getZExtValue());
          storeByte(i, byteVal);
        }
      }
    } else {
      uint64_t value = storeVal.getUInt64();
      for (unsigned i = 0; i < storeSize && i < 8; ++i) {
        storeByte(i, static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
      }
    }
    if (block)
      block->initialized = true;
  }

  LLVM_DEBUG(llvm::dbgs() << "  llvm.store: stored "
                          << (storeVal.isX() ? "X" : std::to_string(storeVal.getUInt64()))
                          << " (" << storeSize << " bytes) at offset " << offset << "\n");

  // UVM fix: when m_inst is stored during global init, also mirror to uvm_top.
  // This prevents the re-entrant m_uvm_get_root() call from seeing
  // m_inst != uvm_top and triggering an infinite fatal loop.
  if (inGlobalInit && !storeVal.isX() && storeVal.getUInt64() != 0) {
    if (auto addrOfOp = storeOp.getAddr().getDefiningOp<LLVM::AddressOfOp>()) {
      if (addrOfOp.getGlobalName().contains("uvm_root::m_inst")) {
        StringRef uvmTopName = "uvm_pkg::uvm_top";
        auto topBlockIt = globalMemoryBlocks.find(uvmTopName);
        if (topBlockIt != globalMemoryBlocks.end()) {
          auto &topBlock = topBlockIt->second;
          uint64_t value = storeVal.getUInt64();
          for (unsigned i = 0; i < storeSize && i < 8 && i < topBlock.size; ++i)
            topBlock.data[i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
          topBlock.initialized = true;
          LLVM_DEBUG(llvm::dbgs()
                     << "  UVM fix: mirrored m_inst store to uvm_top ("
                     << llvm::format_hex(value, 18) << ")\n");
        }
      }
    }
  }
  // Check if any processes are waiting on memory events at this location.
  // If the stored value changed, wake those processes.
  checkMemoryEventWaiters();

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMGEP(ProcessId procId,
                                                        LLVM::GEPOp gepOp) {
  // Get the base pointer value
  InterpretedValue baseVal = getValue(procId, gepOp.getBase());
  if (baseVal.isX()) {
    setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
    return success();
  }

  uint64_t baseAddr = baseVal.getUInt64();

  // Detect null pointer dereference: GEP on NULL (address 0) or near-null
  // addresses (< 0x1000) indicates a null object handle dereference.
  if (baseAddr < 0x1000 && baseAddr != 0) {
    // Non-zero but very small address - likely result of GEP on null
    LLVM_DEBUG(llvm::dbgs() << "  llvm.getelementptr: near-null base address 0x"
                            << llvm::format_hex(baseAddr, 16) << " -> X\n");
    setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
    return success();
  }
  uint64_t offset = 0;

  // Get the element type
  Type elemType = gepOp.getElemType();

  // Process indices using the GEPIndicesAdaptor
  auto indices = gepOp.getIndices();
  Type currentType = elemType;

  size_t idx = 0;
  for (auto indexValue : indices) {
    int64_t indexVal = 0;

    // Check if this is a constant index (IntegerAttr) or dynamic (Value)
    if (auto intAttr = llvm::dyn_cast_if_present<IntegerAttr>(indexValue)) {
      indexVal = intAttr.getInt();
    } else if (auto dynamicIdx = llvm::dyn_cast_if_present<Value>(indexValue)) {
      InterpretedValue dynVal = getValue(procId, dynamicIdx);
      if (dynVal.isX()) {
        setValue(procId, gepOp.getResult(), InterpretedValue::makeX(64));
        return success();
      }
      indexVal = static_cast<int64_t>(dynVal.getUInt64());
    }

    if (idx == 0) {
      // First index: scales by the size of the pointed-to type
      offset += indexVal * getLLVMTypeSize(elemType);
    } else if (auto structType = dyn_cast<LLVM::LLVMStructType>(currentType)) {
      // Struct indexing: accumulate offsets of previous fields
      auto body = structType.getBody();
      for (int64_t i = 0; i < indexVal && static_cast<size_t>(i) < body.size(); ++i) {
        offset += getLLVMTypeSize(body[i]);
      }
      if (static_cast<size_t>(indexVal) < body.size()) {
        currentType = body[indexVal];
      }
    } else if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
      // Array indexing: multiply by element size
      offset += indexVal * getLLVMTypeSize(arrayType.getElementType());
      currentType = arrayType.getElementType();
    } else {
      // For other types, treat as array of the current type
      offset += indexVal * getLLVMTypeSize(currentType);
    }
    ++idx;
  }

  uint64_t resultAddr = baseAddr + offset;
  setValue(procId, gepOp.getResult(), InterpretedValue(resultAddr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.getelementptr: base=0x"
                          << llvm::format_hex(baseAddr, 16) << " offset="
                          << offset << " result=0x"
                          << llvm::format_hex(resultAddr, 16) << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMCall(ProcessId procId,
                                                         LLVM::CallOp callOp) {
  // Get the callee name
  auto callee = callOp.getCallee();
  std::string resolvedCalleeName;

  if (!callee) {
    // Indirect call - try to resolve through vtable
    // The callee operand should be a function pointer loaded from a vtable
    // For indirect calls, the first callee operand is the function pointer
    auto calleeOperands = callOp.getCalleeOperands();
    if (!calleeOperands.empty()) {
      Value calleeOperand = calleeOperands.front();
      InterpretedValue funcPtrVal = getValue(procId, calleeOperand);
      if (!funcPtrVal.isX()) {
        uint64_t funcAddr = funcPtrVal.getUInt64();
        auto it = addressToFunction.find(funcAddr);
        if (it != addressToFunction.end()) {
          resolvedCalleeName = it->second;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: resolved indirect call 0x"
                                  << llvm::format_hex(funcAddr, 16)
                                  << " -> " << resolvedCalleeName << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: indirect call to 0x"
                                  << llvm::format_hex(funcAddr, 16)
                                  << " not in vtable map\n");
        }
      }
    }

    if (resolvedCalleeName.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: indirect call could not be resolved\n");
      for (Value result : callOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
      return success();
    }
  } else {
    resolvedCalleeName = callee->str();
  }

  StringRef calleeName = resolvedCalleeName;

  // Track UVM root construction for re-entrancy handling
  // When m_uvm_get_root is called, we need to mark root construction as started
  // so that re-entrant calls (via uvm_component::new -> get_root) can skip
  // the m_inst != uvm_top comparison that fails during construction.
  bool isGetRoot = calleeName == "m_uvm_get_root";
  if (isGetRoot) {
    ++uvmGetRootDepth;
    if (uvmGetRootDepth == 1) {
      // First call - mark root construction as starting
      __moore_uvm_root_constructing_start();
      LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root entry (depth=1), "
                              << "marking root construction started\n");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root re-entry (depth="
                              << uvmGetRootDepth << ")\n");
    }
  }

  // Use RAII to ensure depth is decremented even on early returns
  auto decrementDepthOnExit = llvm::make_scope_exit([&]() {
    if (isGetRoot) {
      --uvmGetRootDepth;
      if (uvmGetRootDepth == 0) {
        // Last call completed - mark root construction as finished
        __moore_uvm_root_constructing_end();
        LLVM_DEBUG(llvm::dbgs() << "  UVM: m_uvm_get_root exit (depth=0), "
                                << "marking root construction ended\n");
      }
    }
  });

  // Look up the function in the module
  // Safe access: check if process state exists before accessing
  auto stateIt = processStates.find(procId);
  if (stateIt == processStates.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: process state not found for "
                            << "procId=" << procId << ", using rootModule\n");
    // Create a temporary entry to avoid issues
    processStates[procId] = ProcessExecutionState();
    stateIt = processStates.find(procId);
  }
  auto &state = stateIt->second;
  Operation *parent = state.processOrInitialOp;
  while (parent && !isa<ModuleOp>(parent))
    parent = parent->getParentOp();

  // If parent is null (e.g., during module-level init), use rootModule
  if (!parent && rootModule)
    parent = rootModule.getOperation();

  if (!parent) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: could not find module\n");
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  auto moduleOp = cast<ModuleOp>(parent);
  auto funcOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(calleeName);
  if (!funcOp) {
    // Fallback: try looking up as func::FuncOp (common in UVM where
    // LLVM global constructors call func.func methods)
    auto mlirFuncOp = moduleOp.lookupSymbol<func::FuncOp>(calleeName);
    if (mlirFuncOp) {
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: resolved '" << calleeName
                              << "' as func.func (cross-dialect call)\n");
      // Gather arguments
      SmallVector<InterpretedValue, 4> args;
      SmallVector<Value, 4> callOperands;
      for (Value arg : callOp.getOperands()) {
        args.push_back(getValue(procId, arg));
        callOperands.push_back(arg);
      }

      // Check call depth
      auto &crossState = processStates[procId];
      constexpr size_t maxCallDepth = 200;
      if (crossState.callDepth >= maxCallDepth) {
        for (Value result : callOp.getResults()) {
          setValue(procId, result,
                   InterpretedValue::makeX(getTypeWidth(result.getType())));
        }
        return success();
      }

      ++crossState.callDepth;
      SmallVector<InterpretedValue, 2> results;
      LogicalResult funcResult =
          interpretFuncBody(procId, mlirFuncOp, args, results,
                            callOp.getOperation());
      --processStates[procId].callDepth;

      if (failed(funcResult))
        return failure();

      // Check if process suspended
      if (processStates[procId].waiting)
        return success();

      // Set return values
      for (auto [result, retVal] : llvm::zip(callOp.getResults(), results)) {
        setValue(procId, result, retVal);
      }
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: function '" << calleeName
                            << "' not found as llvm.func or func.func\n");
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Check if function has a body
  if (funcOp.isExternal()) {
    // Handle known runtime library functions
    if (calleeName == "__moore_packed_string_to_string") {
      // Get the integer argument (packed string value)
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue arg = getValue(procId, callOp.getOperand(0));
        int64_t value = static_cast<int64_t>(arg.getUInt64());

        // Call the actual runtime function
        MooreString result = __moore_packed_string_to_string(value);

        // Store the result as a struct {ptr, len}
        // For the interpreter, we need to track this specially
        // The result type is !llvm.struct<(ptr, i64)>
        // We'll store the string content directly and return a special marker
        if (callOp.getNumResults() >= 1) {
          // Create a unique ID for this dynamic string
          // We use the pointer value as the ID
          auto ptrVal = reinterpret_cast<int64_t>(result.data);
          auto lenVal = result.len;

          // Store in the dynamic string registry for later retrieval
          dynamicStrings[ptrVal] = {result.data, result.len};

          // For struct result, we pack ptr (lower 64 bits) and len (upper 64)
          // But since struct interpretation is complex, we use a simpler approach:
          // Store the packed value and handle it specially in FormatDynStringOp
          APInt packedResult(128, 0);
          packedResult.insertBits(APInt(64, ptrVal), 0);
          packedResult.insertBits(APInt(64, lenVal), 64);
          setValue(procId, callOp.getResult(),
                   InterpretedValue(packedResult));

          LLVM_DEBUG(llvm::dbgs()
                     << "  llvm.call: __moore_packed_string_to_string("
                     << value << ") = \"";
                     if (result.data && result.len > 0)
                       llvm::dbgs().write(result.data, result.len);
                     llvm::dbgs() << "\"\n");
        }
      }
      return success();
    }

    // Handle string comparison - crucial for UVM factory registration
    if (calleeName == "__moore_string_cmp") {
      // Signature: __moore_string_cmp(ptr to struct{ptr,i64}, ptr to struct{ptr,i64}) -> i32
      // The arguments are pointers to MooreString structs in memory
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        InterpretedValue lhsPtrVal = getValue(procId, callOp.getOperand(0));
        InterpretedValue rhsPtrVal = getValue(procId, callOp.getOperand(1));

        // Helper to extract string data from a pointer to {ptr, i64} struct
        auto extractString = [&](uint64_t structAddr) -> std::pair<const char*, int64_t> {
          // Load the struct from memory - it's 16 bytes: {ptr (8 bytes), len (8 bytes)}
          uint64_t blockOffset = 0;
          MemoryBlock *block = findMemoryBlockByAddress(structAddr, procId, &blockOffset);

          if (block && block->size >= blockOffset + 16) {
            // Extract ptr (first 8 bytes) and len (next 8 bytes)
            uint64_t dataPtr = 0;
            int64_t len = 0;
            for (int i = 0; i < 8; i++) {
              dataPtr |= (static_cast<uint64_t>(block->data[blockOffset + i]) << (i * 8));
              len |= (static_cast<int64_t>(block->data[blockOffset + 8 + i]) << (i * 8));
            }

            // Empty string case (ptr=0, len=0)
            if (dataPtr == 0 || len <= 0) {
              return {nullptr, 0};
            }

            // Look up the actual string data from dynamicStrings registry
            auto dynIt = dynamicStrings.find(static_cast<int64_t>(dataPtr));
            if (dynIt != dynamicStrings.end()) {
              return {dynIt->second.first, std::min(len, static_cast<int64_t>(dynIt->second.second))};
            }

            // Try to find in global memory (for string literals)
            uint64_t strOffset = 0;
            MemoryBlock *strBlock = findMemoryBlockByAddress(dataPtr, procId, &strOffset);
            if (strBlock && strBlock->size >= strOffset + len) {
              return {reinterpret_cast<const char*>(strBlock->data.data() + strOffset), len};
            }
          }
          return {nullptr, -1};  // Error case
        };

        // Handle X (uninitialized) pointer values - don't crash on garbage addresses
        if (lhsPtrVal.isX() || rhsPtrVal.isX()) {
          // Result is indeterminate (X) if either input is X, but we return 0 for safety
          setValue(procId, callOp.getResult(), InterpretedValue(APInt(32, 0, true)));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_cmp() - X input, returning 0\n");
          return success();
        }

        auto [lhsData, lhsLen] = extractString(lhsPtrVal.getUInt64());
        auto [rhsData, rhsLen] = extractString(rhsPtrVal.getUInt64());

        // Perform comparison
        int32_t result = 0;
        bool lhsEmpty = (lhsData == nullptr || lhsLen <= 0);
        bool rhsEmpty = (rhsData == nullptr || rhsLen <= 0);

        if (lhsEmpty && rhsEmpty) {
          result = 0;  // Both empty, equal
        } else if (lhsEmpty) {
          result = -1;  // LHS empty, RHS not
        } else if (rhsEmpty) {
          result = 1;  // LHS not empty, RHS empty
        } else {
          // Both have data, compare lexicographically
          size_t minLen = std::min(static_cast<size_t>(lhsLen), static_cast<size_t>(rhsLen));
          result = std::memcmp(lhsData, rhsData, minLen);
          if (result == 0 && lhsLen != rhsLen) {
            result = (lhsLen < rhsLen) ? -1 : 1;
          }
        }

        setValue(procId, callOp.getResult(), InterpretedValue(APInt(32, result, true)));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_cmp(\"";
                   if (lhsData && lhsLen > 0) llvm::dbgs().write(lhsData, lhsLen);
                   llvm::dbgs() << "\", \"";
                   if (rhsData && rhsLen > 0) llvm::dbgs().write(rhsData, rhsLen);
                   llvm::dbgs() << "\") = " << result << "\n");
      }
      return success();
    }

    // Handle string length
    if (calleeName == "__moore_string_len") {
      // Signature: __moore_string_len(ptr to struct{ptr,i64}) -> i32
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue ptrVal = getValue(procId, callOp.getOperand(0));

        // Handle X (uninitialized) pointer values
        if (ptrVal.isX()) {
          setValue(procId, callOp.getResult(), InterpretedValue(APInt(32, 0, true)));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_len() - X input, returning 0\n");
          return success();
        }

        uint64_t structAddr = ptrVal.getUInt64();

        // Load the struct from memory
        uint64_t blockOffset = 0;
        MemoryBlock *block = findMemoryBlockByAddress(structAddr, procId, &blockOffset);
        int32_t len = 0;

        if (block && block->size >= blockOffset + 16) {
          // Extract len (bytes 8-15)
          int64_t rawLen = 0;
          for (int i = 0; i < 8; i++) {
            rawLen |= (static_cast<int64_t>(block->data[blockOffset + 8 + i]) << (i * 8));
          }
          len = static_cast<int32_t>(rawLen);
        }

        setValue(procId, callOp.getResult(), InterpretedValue(APInt(32, len, true)));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_len() = " << len << "\n");
      }
      return success();
    }

    // Handle malloc - dynamic memory allocation for class instances
    if (calleeName == "malloc") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue sizeArg = getValue(procId, callOp.getOperand(0));
        uint64_t size = sizeArg.isX() ? 256 : sizeArg.getUInt64();  // Default size if X

        // Use global address counter to avoid overlap between processes.
        // Each process has its own nextMemoryAddress for allocas, but malloc
        // blocks are stored globally in mallocBlocks, so we need a global counter.
        uint64_t addr = globalNextAddress;
        globalNextAddress += size;

        // Create a memory block for this allocation
        MemoryBlock block(size, 64);
        block.initialized = true;  // Mark as initialized with zeros
        std::fill(block.data.begin(), block.data.end(), 0);

        // Store the block - use the address as a key
        // We need to track malloc'd blocks separately so findMemoryBlock can find them
        mallocBlocks[addr] = std::move(block);

        setValue(procId, callOp.getResult(), InterpretedValue(addr, 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: malloc(" << size
                                << ") = 0x" << llvm::format_hex(addr, 16) << "\n");
      }
      return success();
    }

    // Helper lambda to resolve a pointer address to string data from global memory.
    // Returns a StringRef to the string content, or empty StringRef if not found.
    auto resolvePointerToString = [&](uint64_t addr, int64_t len) -> std::string {
      if (addr == 0 || len <= 0)
        return "";

      // First check dynamicStrings registry - these are actual C pointers from runtime
      auto dynIt = dynamicStrings.find(static_cast<int64_t>(addr));
      if (dynIt != dynamicStrings.end() && dynIt->second.first && dynIt->second.second > 0) {
        size_t effectiveLen = std::min(static_cast<size_t>(len), static_cast<size_t>(dynIt->second.second));
        return std::string(dynIt->second.first, effectiveLen);
      }
      // Search through global memory blocks for this address
      for (auto &[globalName, globalAddr] : globalAddresses) {
        auto blockIt = globalMemoryBlocks.find(globalName);
        if (blockIt != globalMemoryBlocks.end()) {
          MemoryBlock &block = blockIt->second;
          uint64_t globalSize = block.size;
          if (addr >= globalAddr && addr < globalAddr + globalSize) {
            uint64_t offset = addr - globalAddr;
            size_t availableLen = std::min(static_cast<size_t>(len),
                                           block.data.size() - static_cast<size_t>(offset));
            if (availableLen > 0 && block.initialized) {
              return std::string(reinterpret_cast<const char*>(block.data.data() + offset),
                                 availableLen);
            }
          }
        }
      }

      // Check malloc'd blocks
      for (auto &[blockAddr, block] : mallocBlocks) {
        if (addr >= blockAddr && addr < blockAddr + block.size) {
          uint64_t offset = addr - blockAddr;
          size_t availableLen = std::min(static_cast<size_t>(len),
                                         block.data.size() - static_cast<size_t>(offset));
          if (availableLen > 0) {
            return std::string(reinterpret_cast<const char*>(block.data.data() + offset),
                               availableLen);
          }
        }
      }

      return "";
    };

    // Handle UVM report functions - intercept both runtime calls (__moore_uvm_report_*)
    // and unconverted UVM package calls (uvm_pkg::uvm_report_*)
    if (calleeName == "__moore_uvm_report_info" ||
        calleeName == "__moore_uvm_report_warning" ||
        calleeName == "__moore_uvm_report_error" ||
        calleeName == "__moore_uvm_report_fatal") {
      // Signature: (id, idLen, message, messageLen, verbosity, filename, filenameLen, line, context, contextLen)
      if (callOp.getNumOperands() >= 10) {
        // Extract arguments
        uint64_t idPtr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t idLen = static_cast<int64_t>(getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t msgPtr = getValue(procId, callOp.getOperand(2)).getUInt64();
        int64_t msgLen = static_cast<int64_t>(getValue(procId, callOp.getOperand(3)).getUInt64());
        int32_t verbosity = static_cast<int32_t>(getValue(procId, callOp.getOperand(4)).getUInt64());
        uint64_t filePtr = getValue(procId, callOp.getOperand(5)).getUInt64();
        int64_t fileLen = static_cast<int64_t>(getValue(procId, callOp.getOperand(6)).getUInt64());
        int32_t line = static_cast<int32_t>(getValue(procId, callOp.getOperand(7)).getUInt64());
        uint64_t ctxPtr = getValue(procId, callOp.getOperand(8)).getUInt64();
        int64_t ctxLen = static_cast<int64_t>(getValue(procId, callOp.getOperand(9)).getUInt64());

        // Resolve strings from memory
        std::string idStr = resolvePointerToString(idPtr, idLen);
        std::string msgStr = resolvePointerToString(msgPtr, msgLen);
        std::string fileStr = resolvePointerToString(filePtr, fileLen);
        std::string ctxStr = resolvePointerToString(ctxPtr, ctxLen);

        // Call the appropriate runtime function
        if (calleeName == "__moore_uvm_report_info") {
          __moore_uvm_report_info(idStr.c_str(), idStr.size(),
                                  msgStr.c_str(), msgStr.size(),
                                  verbosity, fileStr.c_str(), fileStr.size(),
                                  line, ctxStr.c_str(), ctxStr.size());
        } else if (calleeName == "__moore_uvm_report_warning") {
          __moore_uvm_report_warning(idStr.c_str(), idStr.size(),
                                     msgStr.c_str(), msgStr.size(),
                                     verbosity, fileStr.c_str(), fileStr.size(),
                                     line, ctxStr.c_str(), ctxStr.size());
        } else if (calleeName == "__moore_uvm_report_error") {
          __moore_uvm_report_error(idStr.c_str(), idStr.size(),
                                   msgStr.c_str(), msgStr.size(),
                                   verbosity, fileStr.c_str(), fileStr.size(),
                                   line, ctxStr.c_str(), ctxStr.size());
        } else if (calleeName == "__moore_uvm_report_fatal") {
          __moore_uvm_report_fatal(idStr.c_str(), idStr.size(),
                                   msgStr.c_str(), msgStr.size(),
                                   verbosity, fileStr.c_str(), fileStr.size(),
                                   line, ctxStr.c_str(), ctxStr.size());
        }

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: " << calleeName
                                << "(id=\"" << idStr << "\", msg=\"" << msgStr << "\")\n");
      }
      return success();
    }

    // Handle unconverted UVM report calls with LLVM struct string arguments
    // Signature: (id_struct, msg_struct, verbosity, filename_struct, line, context_struct, report_enabled_checked)
    // where *_struct is {ptr, i64}
    // Note: The struct values are tracked internally by the interpreter during llvm.call handling.
    // When we encounter uvm_report calls, we look up the struct values in our value map
    // and extract the ptr/len fields that were stored there.
    if (calleeName == "uvm_pkg::uvm_report_info" ||
        calleeName == "uvm_pkg::uvm_report_warning" ||
        calleeName == "uvm_pkg::uvm_report_error" ||
        calleeName == "uvm_pkg::uvm_report_fatal") {
      if (callOp.getNumOperands() >= 7) {
        // For now, print a placeholder message to indicate we intercepted the call
        // Full struct field extraction requires tracking aggregate values in the interpreter
        llvm::outs() << "UVM_INFO <intercepted> @ "
                     << scheduler.getCurrentTime().realTime << " fs: "
                     << "[" << calleeName << " call intercepted - struct args not yet extracted]\n";

        LLVM_DEBUG(llvm::dbgs() << "  func.call: " << calleeName
                                << " intercepted (7 args, struct extraction pending)\n");
      }
      return success();
    }

    // Handle __moore_uvm_report_enabled
    if (calleeName == "__moore_uvm_report_enabled") {
      // Signature: (verbosity, severity, id, idLen) -> int32_t
      if (callOp.getNumOperands() >= 4) {
        int32_t verbosity = static_cast<int32_t>(getValue(procId, callOp.getOperand(0)).getUInt64());
        int32_t severity = static_cast<int32_t>(getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t idPtr = getValue(procId, callOp.getOperand(2)).getUInt64();
        int64_t idLen = static_cast<int64_t>(getValue(procId, callOp.getOperand(3)).getUInt64());

        std::string idStr = resolvePointerToString(idPtr, idLen);
        int32_t result = __moore_uvm_report_enabled(verbosity, severity,
                                                     idStr.c_str(), idStr.size());

        if (callOp.getNumResults() >= 1) {
          setValue(procId, callOp.getResult(), InterpretedValue(result, 32));
        }

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_report_enabled("
                                << verbosity << ", " << severity << ", \"" << idStr
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // Handle __moore_uvm_report_summarize
    if (calleeName == "__moore_uvm_report_summarize") {
      __moore_uvm_report_summarize();
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_report_summarize()\n");
      return success();
    }

    // Handle UVM root re-entrancy runtime functions
    if (calleeName == "__moore_uvm_root_constructing_start") {
      __moore_uvm_root_constructing_start();
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_root_constructing_start()\n");
      return success();
    }

    if (calleeName == "__moore_uvm_root_constructing_end") {
      __moore_uvm_root_constructing_end();
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_root_constructing_end()\n");
      return success();
    }

    if (calleeName == "__moore_uvm_is_root_constructing") {
      bool result = __moore_uvm_is_root_constructing();
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(result ? 1ULL : 0ULL, 1));
      }
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_is_root_constructing() = "
                              << (result ? "true" : "false") << "\n");
      return success();
    }

    if (calleeName == "__moore_uvm_set_root_inst") {
      if (callOp.getNumOperands() >= 1) {
        uint64_t inst = getValue(procId, callOp.getOperand(0)).getUInt64();
        __moore_uvm_set_root_inst(reinterpret_cast<void *>(inst));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_set_root_inst(0x"
                                << llvm::format_hex(inst, 16) << ")\n");
      }
      return success();
    }

    if (calleeName == "__moore_uvm_get_root_inst") {
      void *inst = __moore_uvm_get_root_inst();
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(),
                 InterpretedValue(reinterpret_cast<uint64_t>(inst), 64));
      }
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_uvm_get_root_inst() = 0x"
                              << llvm::format_hex(reinterpret_cast<uint64_t>(inst), 16) << "\n");
      return success();
    }

    // Coverage function stubs (coverage not supported in interpreter)
    // These are needed to run UVM-based testbenches that use covergroups.
    if (calleeName == "__moore_covergroup_create") {
      // Return a dummy covergroup handle (0)
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
      }
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_covergroup_create() -> 0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_covergroup_get_coverage") {
      // Return 0.0 coverage (represented as 0 in fixed-point or just zero bits)
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
      }
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_covergroup_get_coverage() -> 0.0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_coverpoint_init") {
      // No-op: coverpoint initialization is not supported
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverpoint_init() (stub, no-op)\n");
      return success();
    }

    if (calleeName == "__moore_coverpoint_sample") {
      // No-op: coverpoint sampling is not supported
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverpoint_sample() (stub, no-op)\n");
      return success();
    }

    if (calleeName == "__moore_coverpoint_add_ignore_bin") {
      // No-op: ignore bin configuration is not supported
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverpoint_add_ignore_bin() (stub, no-op)\n");
      return success();
    }

    if (calleeName == "__moore_coverpoint_add_illegal_bin") {
      // No-op: illegal bin configuration is not supported
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverpoint_add_illegal_bin() (stub, no-op)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_set_test_name") {
      // No-op: ignore coverage test name in interpreter.
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_set_test_name() (stub, no-op)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_load_db") {
      // Return null handle.
      if (callOp.getNumResults() >= 1)
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_load_db() -> 0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_merge_db") {
      // Return success (0).
      if (callOp.getNumResults() >= 1)
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_merge_db() -> 0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_save_db") {
      // Return success (0).
      if (callOp.getNumResults() >= 1)
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_save_db() -> 0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_load") {
      // Return null handle.
      if (callOp.getNumResults() >= 1)
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_load() -> 0 (stub)\n");
      return success();
    }

    if (calleeName == "__moore_coverage_merge") {
      // Return success (0).
      if (callOp.getNumResults() >= 1)
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_coverage_merge() -> 0 (stub)\n");
      return success();
    }

    // Handle __moore_delay - delay in class method context or fork branches.
    // This is called when a #delay statement appears in a class task/method
    // or inside a fork branch. We accumulate the delay in pendingDelayFs and
    // mark the process as waiting. Multiple sequential __moore_delay calls
    // within the same function will accumulate their delays before the process
    // actually suspends.
    if (calleeName == "__moore_delay") {
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue delayArg = getValue(procId, callOp.getOperand(0));
        int64_t delayFs = delayArg.isX() ? 0 : static_cast<int64_t>(delayArg.getUInt64());

        if (delayFs > 0) {
          auto &state = processStates[procId];

          // Accumulate the delay. Multiple __moore_delay calls in sequence
          // (e.g., #10; #20; #30;) should result in a total delay of 60.
          state.pendingDelayFs += delayFs;

          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_delay(" << delayFs
                                  << " fs) accumulated, total pending = "
                                  << state.pendingDelayFs << " fs\n");

          // Mark the process as waiting. This will cause the executeProcess
          // loop to exit after this step, yielding control to the scheduler.
          // The actual scheduling of the resumption event happens in
          // executeProcess when it detects state.waiting is true and
          // state.pendingDelayFs > 0.
          state.waiting = true;

        } else {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_delay(0) - no delay\n");
        }
      }
      return success();
    }

    // Handle __moore_process_self - get current process handle.
    // Implements the SystemVerilog `process::self()` static method.
    // IEEE 1800-2017 Section 9.7 "Process control"
    // Returns a non-null handle when called from within a process context
    // (llhd.process, initial block, always block, fork branch), or null
    // when called from outside a process context.
    if (calleeName == "__moore_process_self") {
      // We're being called from within the interpreter's process execution,
      // which means we're definitely inside a process context.
      // Return a non-null pointer - we use the process state address as
      // a unique identifier for the process handle.
      auto &state = processStates[procId];
      void *processHandle = &state;

      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_process_self() = 0x"
                              << llvm::format_hex(reinterpret_cast<uint64_t>(processHandle), 16)
                              << " (inside process context)\n");

      // Set the result to the process handle pointer value
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(64, reinterpret_cast<uint64_t>(processHandle))));
      }
      return success();
    }

    // Handle __moore_process_status - query process state.
    // Implements SystemVerilog process::status().
    if (calleeName == "__moore_process_status") {
      uint32_t status = 0; // FINISHED
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        if (!handleVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId) {
            bool killed = false;
            auto stateIt = processStates.find(targetId);
            if (stateIt != processStates.end())
              killed = stateIt->second.killed;

            if (killed) {
              status = 4; // KILLED
            } else if (auto *proc = scheduler.getProcess(targetId)) {
              switch (proc->getState()) {
              case ProcessState::Waiting:
                status = 2; // WAITING
                break;
              case ProcessState::Suspended:
                status = 3; // SUSPENDED
                break;
              case ProcessState::Terminated:
                status = 0; // FINISHED
                break;
              default:
                status = 1; // RUNNING (Ready/Running/Uninitialized)
                break;
              }
            } else {
              status = 0;
            }
          }
        }
      }

      if (callOp.getNumResults() >= 1) {
        Value result = callOp.getResult();
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(APInt(width, status)));
      }
      return success();
    }

    // Handle __moore_process_get_randstate - serialize process RNG state.
    // Implements SystemVerilog process::get_randstate().
    if (calleeName == "__moore_process_get_randstate") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        std::string stateStr;

        if (!handleVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId) {
            auto it = processStates.find(targetId);
            if (it != processStates.end()) {
              std::ostringstream oss;
              oss << it->second.randomGenerator;
              stateStr = oss.str();
            }
          }
        }

        int64_t ptrVal = 0;
        int64_t lenVal = 0;
        if (!stateStr.empty()) {
          interpreterStrings.push_back(std::move(stateStr));
          const std::string &stored = interpreterStrings.back();
          ptrVal = reinterpret_cast<int64_t>(stored.data());
          lenVal = static_cast<int64_t>(stored.size());
          dynamicStrings[ptrVal] = {stored.data(), lenVal};
        }

        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(ptrVal)), 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(lenVal)), 64);
        setValue(procId, callOp.getResult(), InterpretedValue(packedResult));

        LLVM_DEBUG({
          llvm::dbgs() << "  llvm.call: __moore_process_get_randstate(";
          if (handleVal.isX())
            llvm::dbgs() << "X";
          else
            llvm::dbgs() << llvm::format_hex(handleVal.getUInt64(), 16);
          llvm::dbgs() << ") len=" << lenVal << "\n";
        });
      }
      return success();
    }

    // Handle __moore_process_set_randstate - restore process RNG state.
    // Implements SystemVerilog process::set_randstate().
    if (calleeName == "__moore_process_set_randstate") {
      if (callOp.getNumOperands() >= 2) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        InterpretedValue stateVal = getValue(procId, callOp.getOperand(1));

        if (!handleVal.isX() && !stateVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId) {
            auto it = processStates.find(targetId);
            if (it != processStates.end()) {
              APInt bits = stateVal.getAPInt();
              uint64_t ptrVal =
                  bits.extractBits(64, 0).getZExtValue();
              uint64_t lenVal =
                  bits.extractBits(64, 64).getZExtValue();

              std::string stateStr =
                  resolvePointerToString(ptrVal, static_cast<int64_t>(lenVal));

              if (!stateStr.empty()) {
                std::istringstream iss(stateStr);
                iss >> it->second.randomGenerator;
              }
            }
          }
        }
      }
      return success();
    }

    // Handle __moore_process_srandom - seed the process RNG.
    // Implements SystemVerilog process::srandom().
    if (calleeName == "__moore_process_srandom") {
      if (callOp.getNumOperands() >= 2) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        InterpretedValue seedVal = getValue(procId, callOp.getOperand(1));

        if (!handleVal.isX() && !seedVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId) {
            auto it = processStates.find(targetId);
            if (it != processStates.end()) {
              it->second.randomGenerator.seed(
                  static_cast<uint32_t>(seedVal.getUInt64()));
            }
          }
        }
      }
      return success();
    }

    // Handle __moore_process_kill - terminate a process.
    // Implements SystemVerilog process::kill().
    if (calleeName == "__moore_process_kill") {
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        if (!handleVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId)
            finalizeProcess(targetId, /*killed=*/true);
        }
      }
      return success();
    }

    // Handle __moore_process_await - wait for a process to finish or be killed.
    // Implements SystemVerilog process::await().
    if (calleeName == "__moore_process_await") {
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue handleVal = getValue(procId, callOp.getOperand(0));
        if (!handleVal.isX()) {
          ProcessId targetId = resolveProcessHandle(handleVal.getUInt64());
          if (targetId != InvalidProcessId) {
            bool targetKilled = false;
            auto targetStateIt = processStates.find(targetId);
            if (targetStateIt != processStates.end())
              targetKilled = targetStateIt->second.killed;

            bool targetDone = targetKilled;
            if (!targetDone) {
              if (auto *proc = scheduler.getProcess(targetId))
                targetDone = (proc->getState() == ProcessState::Terminated);
              else
                targetDone = true;
            }

            if (!targetDone) {
              processAwaiters[targetId].push_back(procId);
              auto &state = processStates[procId];
              state.waiting = true;
              if (auto *proc = scheduler.getProcess(procId))
                proc->setState(ProcessState::Waiting);
            }
          }
        }
      }
      return success();
    }

    // Handle __moore_wait_condition - wait until condition becomes true.
    // Implements the SystemVerilog `wait(condition)` statement.
    // Signature: __moore_wait_condition(i32 condition)
    // If condition is true (non-zero), return immediately.
    // If condition is false (zero), suspend the process and wait for signal
    // changes, then re-evaluate by restarting from the beginning of the
    // current block.
    if (calleeName == "__moore_wait_condition") {
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue condArg = getValue(procId, callOp.getOperand(0));
        bool conditionTrue = !condArg.isX() && condArg.getUInt64() != 0;

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_wait_condition("
                                << (condArg.isX() ? "X" : std::to_string(condArg.getUInt64()))
                                << ") -> condition is "
                                << (conditionTrue ? "true" : "false") << "\n");

        if (conditionTrue) {
          // Condition is already true, continue immediately.
          // Clear the restart block since we're done waiting.
          auto &state = processStates[procId];
          state.waitConditionRestartBlock = nullptr;
          return success();
        }

        // Condition is false - suspend the process and set up sensitivity
        // to all probed signals so we wake up when something changes.
        auto &state = processStates[procId];
        state.waiting = true;

        // Find the operations that compute the condition value by walking
        // backwards from the condition argument. We need to invalidate these
        // cached values so they get recomputed when we re-check the condition.
        //
        // CRITICAL: We trace through operations that need re-execution:
        // - llvm.load: reads memory that may have changed
        // - llhd.prb: probes a signal that may have changed
        // - Arithmetic/comparison ops that depend on loads/probes
        //
        // We do NOT trace into:
        // - llvm.getelementptr: pointer arithmetic (doesn't read memory)
        // - Constant operations
        //
        // This avoids re-executing side-effecting operations like sim.fork.
        state.waitConditionValuesToInvalidate.clear();
        state.waitConditionRestartBlock = state.currentBlock;

        // Find the earliest load/probe in the condition computation chain
        Value condValue = callOp.getOperand(0);
        llvm::SmallVector<Value, 16> worklist;
        llvm::SmallPtrSet<Value, 16> visited;
        worklist.push_back(condValue);

        Operation *earliestLoadOp = nullptr;

        while (!worklist.empty()) {
          Value v = worklist.pop_back_val();
          if (!visited.insert(v).second)
            continue;

          // If the value has a defining operation in this block, process it
          if (Operation *defOp = v.getDefiningOp()) {
            if (defOp->getBlock() == state.currentBlock) {
              // Check if this is a load or probe operation - these read values that may change
              if (isa<LLVM::LoadOp, llhd::ProbeOp>(defOp)) {
                state.waitConditionValuesToInvalidate.push_back(v);
                // Track the earliest load/probe operation
                if (!earliestLoadOp || defOp->isBeforeInBlock(earliestLoadOp))
                  earliestLoadOp = defOp;
              }

              // Add operands to worklist - trace through comparison, arithmetic,
              // extraction, and load/probe ops but stop at getelementptr
              // (doesn't read memory)
              bool shouldTrace = isa<comb::ICmpOp, LLVM::ZExtOp, LLVM::SExtOp>(defOp) ||
                                 isa<comb::AddOp, comb::SubOp, comb::AndOp>(defOp) ||
                                 isa<comb::OrOp, comb::XorOp>(defOp) ||
                                 isa<comb::ExtractOp, LLVM::ExtractValueOp>(defOp) ||
                                 isa<LLVM::LoadOp>(defOp) ||
                                 isa<llhd::ProbeOp>(defOp);
              if (shouldTrace) {
                state.waitConditionValuesToInvalidate.push_back(v);
                for (Value operand : defOp->getOperands()) {
                  if (!isa<BlockArgument>(operand) &&
                      operand.getDefiningOp() &&
                      operand.getDefiningOp()->getBlock() == state.currentBlock) {
                    worklist.push_back(operand);
                  }
                }
              }
            }
          }
        }

        // Save the restart point - the earliest load/probe operation (or the call itself)
        if (earliestLoadOp) {
          state.waitConditionRestartOp = mlir::Block::iterator(earliestLoadOp);
        } else {
          // No loads/probes found - restart from the call itself
          state.waitConditionRestartOp = mlir::Block::iterator(&*callOp);
        }

        LLVM_DEBUG(llvm::dbgs() << "    Setting restart point for wait_condition "
                                << "re-evaluation (" << state.waitConditionValuesToInvalidate.size()
                                << " values to invalidate)\n");

        // For wait(condition), always use polling instead of signal-based waiting.
        // The condition may depend on class member variables stored in heap memory
        // that isn't tracked via LLHD signals. Even if we find signals in the
        // process, they may not be the ones that affect the condition.
        //
        // TODO: For better performance, we could track which memory locations
        // the condition depends on and only poll when those change.
        constexpr int64_t kPollDelayFs = 1000000; // 1 ps (1000000 fs) polling interval
        SimTime currentTime = scheduler.getCurrentTime();
        SimTime targetTime = currentTime.advanceTime(kPollDelayFs);

        LLVM_DEBUG(llvm::dbgs() << "    Scheduling wait_condition poll after "
                                << kPollDelayFs << " fs\n");

        // Schedule the process to resume after the delay
        scheduler.getEventScheduler().schedule(
            targetTime, SchedulingRegion::Active,
            Event([this, procId]() {
              // Resume the process - it will restart from the wait_condition
              // restart point and re-evaluate the condition
              auto &st = processStates[procId];
              st.waiting = false;
              scheduler.scheduleProcess(procId, SchedulingRegion::Active);
            }));
      }
      return success();
    }

    // Handle __moore_event_trigger - trigger an event.
    // Implements the SystemVerilog `->event` syntax.
    // Signature: __moore_event_trigger(ptr event)
    // Sets the event flag to true and wakes up processes waiting on it.
    if (calleeName == "__moore_event_trigger") {
      if (callOp.getNumOperands() >= 1) {
        InterpretedValue eventPtrVal = getValue(procId, callOp.getOperand(0));
        if (!eventPtrVal.isX()) {
          uint64_t eventAddr = eventPtrVal.getUInt64();
          uint64_t offset = 0;
          MemoryBlock *block = findMemoryBlockByAddress(eventAddr, procId, &offset);
          if (block && block->size >= offset + 1) {
            // Set the event flag to true (1)
            block->data[offset] = 1;
            block->initialized = true;
            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_event_trigger() - "
                                    << "set event at address 0x"
                                    << llvm::format_hex(eventAddr, 16) << " to true\n");

            // Check if any processes are waiting on this memory location
            checkMemoryEventWaiters();
          } else {
            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_event_trigger() - "
                                    << "could not find memory block for address 0x"
                                    << llvm::format_hex(eventAddr, 16) << "\n");
          }
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_event_trigger() - "
                                  << "event pointer is X\n");
        }
      }
      return success();
    }

    // Handle __moore_event_triggered - check if an event was triggered.
    // Implements the SystemVerilog `.triggered` property on events.
    // Signature: __moore_event_triggered(ptr event) -> i1
    // Returns true if the event flag is set.
    if (calleeName == "__moore_event_triggered") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue eventPtrVal = getValue(procId, callOp.getOperand(0));
        bool triggered = false;

        if (!eventPtrVal.isX()) {
          uint64_t eventAddr = eventPtrVal.getUInt64();
          uint64_t offset = 0;
          MemoryBlock *block = findMemoryBlockByAddress(eventAddr, procId, &offset);
          if (block && block->size >= offset + 1 && block->initialized) {
            triggered = (block->data[offset] != 0);
          }
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_event_triggered() - "
                                  << "event at address 0x"
                                  << llvm::format_hex(eventAddr, 16)
                                  << " is " << (triggered ? "triggered" : "not triggered") << "\n");
        } else {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_event_triggered() - "
                                  << "event pointer is X, returning false\n");
        }

        setValue(procId, callOp.getResult(), InterpretedValue(APInt(1, triggered ? 1 : 0)));
      }
      return success();
    }

    // Handle __moore_wait_event - wait for an event in func.func context.
    // This is used when moore.wait_event appears in forked code or class methods
    // where llhd.wait cannot be used (requires llhd.process parent).
    // Signature: __moore_wait_event(i32 edgeKind, ptr valuePtr)
    // edgeKind: 0=AnyChange, 1=PosEdge, 2=NegEdge, 3=BothEdges
    // valuePtr: pointer to the memory location to watch (or null for any change)
    if (calleeName == "__moore_wait_event") {
      if (callOp.getNumOperands() >= 2) {
        InterpretedValue edgeKindVal = getValue(procId, callOp.getOperand(0));
        InterpretedValue valuePtrVal = getValue(procId, callOp.getOperand(1));

        int32_t edgeKind = edgeKindVal.isX() ? 0 : static_cast<int32_t>(edgeKindVal.getUInt64());
        uint64_t valueAddr = valuePtrVal.isX() ? 0 : valuePtrVal.getUInt64();

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_wait_event(edgeKind="
                                << edgeKind << ", valuePtr=0x"
                                << llvm::format_hex(valueAddr, 16) << ")\n");

        auto &state = processStates[procId];
        state.waiting = true;

        if (valueAddr != 0) {
          // Set up a memory event waiter for the specified address.
          uint64_t offset = 0;
          MemoryBlock *block = findMemoryBlockByAddress(valueAddr, procId, &offset);

          // Read current value for edge detection.
          uint64_t currentValue = 0;
          unsigned valueSize = 1;
          if (block && block->size >= offset + 1 && block->initialized) {
            currentValue = block->data[offset];
            // For larger values, read more bytes (up to 8).
            valueSize = std::min(block->size - offset, static_cast<size_t>(8));
            if (valueSize > 1) {
              currentValue = 0;
              for (unsigned i = 0; i < valueSize; ++i)
                currentValue |= static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
            }
          }

          MemoryEventWaiter waiter;
          waiter.address = valueAddr;
          waiter.lastValue = currentValue;
          waiter.valueSize = valueSize;
          // Use rising edge detection for PosEdge, otherwise any change.
          waiter.waitForRisingEdge = (edgeKind == 1); // PosEdge
          memoryEventWaiters[procId] = waiter;

          LLVM_DEBUG(llvm::dbgs() << "    Set up memory event waiter for address 0x"
                                  << llvm::format_hex(valueAddr, 16)
                                  << " with initial value " << currentValue
                                  << " (edgeKind=" << edgeKind << ")\n");
        } else {
          // No specific address to watch. This happens when:
          // 1. The conversion couldn't trace back to a specific memory location
          // 2. We're waiting on a signal passed as a function argument
          //
          // For proper blocking, we need to wait for ANY signal change.
          // Set up the process to be woken on any signal change by using
          // the signal-based sensitivity mechanism.
          //
          // First, check if this process has any signals in its sensitivity list.
          // If so, wait for any of them to change.
          bool hasSignalSensitivity = !state.lastSensitivityEntries.empty();

          if (hasSignalSensitivity) {
            // The process already has sensitivity entries from previous waits.
            // Keep them and wait for any signal change.
            // The process will be woken by triggerSensitiveProcesses when
            // any of those signals change.
            LLVM_DEBUG(llvm::dbgs() << "    No specific address, waiting on "
                                    << state.lastSensitivityEntries.size()
                                    << " signal sensitivities\n");
            // state.waiting is already true, just don't schedule
          } else {
            // No signal sensitivities found. This can happen if we're in a
            // func.func context called from llhd.process but no signals are
            // being watched yet. Use polling as a fallback.
            constexpr int64_t kPollDelayFs = 1000000; // 1 ps polling interval
            SimTime currentTime = scheduler.getCurrentTime();
            SimTime targetTime = currentTime.advanceTime(kPollDelayFs);

            LLVM_DEBUG(llvm::dbgs() << "    No signals or address to watch, "
                                    << "polling after " << kPollDelayFs << " fs\n");

            scheduler.getEventScheduler().schedule(
                targetTime, SchedulingRegion::Active,
                Event([this, procId]() {
                  auto &st = processStates[procId];
                  st.waiting = false;
                  scheduler.scheduleProcess(procId, SchedulingRegion::Active);
                }));
          }
        }
      }
      return success();
    }

    // Handle __moore_queue_push_back - append element to queue
    // Signature: (queue_ptr, element_ptr, element_size)
    if (calleeName == "__moore_queue_push_back") {
      if (callOp.getNumOperands() >= 3) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t elemAddr = getValue(procId, callOp.getOperand(1)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(getValue(procId, callOp.getOperand(2)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            // Read from the correct offset within the block
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            // Allocate new storage with space for one more element.
            // Use global address counter to avoid overlap with other processes.
            int64_t newLen = queueLen + 1;
            uint64_t newDataAddr = globalNextAddress;
            globalNextAddress += newLen * elemSize;

            MemoryBlock newBlock(newLen * elemSize, 64);
            newBlock.initialized = true;

            // Copy existing elements
            if (dataPtr != 0 && queueLen > 0) {
              auto *oldBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (oldBlock && oldBlock->initialized) {
                size_t copySize = std::min(static_cast<size_t>(queueLen * elemSize),
                                           std::min(oldBlock->data.size(), newBlock.data.size()));
                std::memcpy(newBlock.data.data(), oldBlock->data.data(), copySize);
              }
            }

            // Copy new element to the end
            uint64_t elemOffset = 0;
            auto *elemBlock = findMemoryBlockByAddress(elemAddr, procId, &elemOffset);
            if (elemBlock && elemBlock->initialized) {
              size_t availableBytes = (elemOffset < elemBlock->data.size())
                  ? elemBlock->data.size() - elemOffset : 0;
              size_t copySize = std::min(static_cast<size_t>(elemSize), availableBytes);
              if (copySize > 0)
                std::memcpy(newBlock.data.data() + queueLen * elemSize,
                            elemBlock->data.data() + elemOffset, copySize);
            }

            mallocBlocks[newDataAddr] = std::move(newBlock);

            // Update queue struct with new ptr and len (at the correct offset)
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + i] = static_cast<uint8_t>((newDataAddr >> (i * 8)) & 0xFF);
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + 8 + i] = static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);

            LLVM_DEBUG(llvm::dbgs() << "  __moore_queue_push_back: queueAddr=0x"
                                    << llvm::format_hex(queueAddr, 16)
                                    << " queueOffset=" << queueOffset
                                    << " newDataAddr=0x" << llvm::format_hex(newDataAddr, 16)
                                    << " elemSize=" << elemSize
                                    << " newLen=" << newLen << "\n");

            // Queue content changed - check if any process is waiting on memory events
            checkMemoryEventWaiters();
          }
        }
      }
      return success();
    }

    // Handle __moore_queue_size - return queue length
    // Signature: (queue_ptr) -> i64
    if (calleeName == "__moore_queue_size") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t queueLen = 0;

        if (queueAddr != 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(static_cast<uint64_t>(queueLen), 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_size("
                                << "queue=0x" << llvm::format_hex(queueAddr, 16)
                                << ") = " << queueLen << "\n");
      }
      return success();
    }

    // Handle __moore_queue_clear - clear all elements
    // Signature: (queue_ptr)
    if (calleeName == "__moore_queue_clear") {
      if (callOp.getNumOperands() >= 1) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();

        if (queueAddr != 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            // Set data ptr to 0 and len to 0 (at the correct offset)
            std::memset(queueBlock->data.data() + queueOffset, 0, 16);
            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_clear("
                                    << "queue=0x" << llvm::format_hex(queueAddr, 16)
                                    << ")\n");
          }
        }
      }
      return success();
    }

    // Handle __moore_queue_pop_back - remove and return last element
    // Signature: (queue_ptr, element_size) -> element_value
    if (calleeName == "__moore_queue_pop_back") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t result = 0;

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            if (queueLen > 0 && dataPtr != 0) {
              // Read the last element
              auto *dataBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (dataBlock && dataBlock->initialized) {
                size_t offset = (queueLen - 1) * elemSize;
                for (int64_t i = 0; i < std::min(elemSize, int64_t(8)); ++i)
                  result |= static_cast<uint64_t>(dataBlock->data[offset + i]) << (i * 8);
              }

              // Decrement length
              int64_t newLen = queueLen - 1;
              for (int i = 0; i < 8; ++i)
                queueBlock->data[queueOffset + 8 + i] = static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);
            }
          }
        }

        setValue(procId, callOp.getResult(), InterpretedValue(result, 64));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_pop_back("
                                << "queue=0x" << llvm::format_hex(queueAddr, 16)
                                << ") = " << result << "\n");

        // Queue content changed - check if any process is waiting on memory events
        checkMemoryEventWaiters();
      }
      return success();
    }

    // Handle __moore_queue_pop_front - remove and return first element
    // Signature: (queue_ptr, element_size) -> element_value
    if (calleeName == "__moore_queue_pop_front") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t result = 0;

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            if (queueLen > 0 && dataPtr != 0) {
              auto *dataBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (dataBlock && dataBlock->initialized) {
                // Read the first element
                for (int64_t i = 0; i < std::min(elemSize, int64_t(8)); ++i)
                  result |= static_cast<uint64_t>(dataBlock->data[i]) << (i * 8);

                // Shift remaining elements forward
                if (queueLen > 1) {
                  std::memmove(dataBlock->data.data(),
                               dataBlock->data.data() + elemSize,
                               (queueLen - 1) * elemSize);
                }
              }

              // Decrement length
              int64_t newLen = queueLen - 1;
              for (int i = 0; i < 8; ++i)
                queueBlock->data[queueOffset + 8 + i] = static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);
            }
          }
        }

        setValue(procId, callOp.getResult(), InterpretedValue(result, 64));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_pop_front("
                                << "queue=0x" << llvm::format_hex(queueAddr, 16)
                                << ") = " << result << "\n");

        // Queue content changed - check if any process is waiting on memory events
        checkMemoryEventWaiters();
      }
      return success();
    }

    // Handle __moore_queue_push_front - prepend element to queue
    // Signature: (queue_ptr, element_ptr, element_size)
    if (calleeName == "__moore_queue_push_front") {
      if (callOp.getNumOperands() >= 3) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t elemAddr = getValue(procId, callOp.getOperand(1)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(getValue(procId, callOp.getOperand(2)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            // Use global address counter to avoid overlap with other processes.
            int64_t newLen = queueLen + 1;
            uint64_t newDataAddr = globalNextAddress;
            globalNextAddress += newLen * elemSize;

            MemoryBlock newBlock(newLen * elemSize, 64);
            newBlock.initialized = true;

            // Copy new element to the front
            uint64_t elemOffset = 0;
            auto *elemBlock = findMemoryBlockByAddress(elemAddr, procId, &elemOffset);
            if (elemBlock && elemBlock->initialized) {
              size_t availableBytes = (elemOffset < elemBlock->data.size())
                  ? elemBlock->data.size() - elemOffset : 0;
              size_t copySize = std::min(static_cast<size_t>(elemSize), availableBytes);
              if (copySize > 0)
                std::memcpy(newBlock.data.data(),
                            elemBlock->data.data() + elemOffset, copySize);
            }

            // Copy existing elements after the new one
            if (dataPtr != 0 && queueLen > 0) {
              auto *oldBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (oldBlock && oldBlock->initialized) {
                size_t copySize = std::min(static_cast<size_t>(queueLen * elemSize),
                                           oldBlock->data.size());
                std::memcpy(newBlock.data.data() + elemSize,
                            oldBlock->data.data(), copySize);
              }
            }

            mallocBlocks[newDataAddr] = std::move(newBlock);

            // Update queue struct (at the correct offset)
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + i] = static_cast<uint8_t>((newDataAddr >> (i * 8)) & 0xFF);
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + 8 + i] = static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);

            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_push_front("
                                    << "queue=0x" << llvm::format_hex(queueAddr, 16)
                                    << ") -> len=" << newLen << "\n");
          }
        }
      }
      return success();
    }

    // Handle __moore_assoc_create - create an associative array
    // Signature: (key_size: i32, value_size: i32) -> ptr
    if (calleeName == "__moore_assoc_create") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        int32_t keySize = static_cast<int32_t>(getValue(procId, callOp.getOperand(0)).getUInt64());
        int32_t valueSize = static_cast<int32_t>(getValue(procId, callOp.getOperand(1)).getUInt64());

        void *arrayPtr = __moore_assoc_create(keySize, valueSize);
        uint64_t ptrVal = reinterpret_cast<uint64_t>(arrayPtr);

        // Track this as a valid associative array address
        validAssocArrayAddresses.insert(ptrVal);

        setValue(procId, callOp.getResult(), InterpretedValue(ptrVal, 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_create("
                                << keySize << ", " << valueSize << ") = 0x"
                                << llvm::format_hex(ptrVal, 16) << "\n");
      }
      return success();
    }

    // Handle __moore_assoc_get_ref - get reference to element in associative array
    // Signature: (array: ptr, key: ptr, value_size: i32) -> ptr
    // The runtime determines string vs integer keys from the array header.
    if (calleeName == "__moore_assoc_get_ref") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        InterpretedValue arrayVal = getValue(procId, callOp.getOperand(0));
        InterpretedValue keyVal = getValue(procId, callOp.getOperand(1));
        InterpretedValue valueSizeVal = getValue(procId, callOp.getOperand(2));

        // Handle X (uninitialized) values - return null pointer
        if (arrayVal.isX() || keyVal.isX() || valueSizeVal.isX()) {
          setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref - X input, returning null\n");
          return success();
        }

        uint64_t arrayAddr = arrayVal.getUInt64();
        uint64_t keyAddr = keyVal.getUInt64();
        int32_t valueSize = static_cast<int32_t>(valueSizeVal.getUInt64());

        // Check if arrayAddr is null or not a valid associative array address.
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL; // 1TB
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        bool isInValidSet = validAssocArrayAddresses.contains(arrayAddr);
        if (arrayAddr == 0 || (!isInValidSet && !isValidNativeAddr)) {
          // Auto-create the associative array on first access (SystemVerilog semantics).
          int32_t keySize = 8; // default: 8-byte integer key
          if (auto allocaOp = callOp.getOperand(1).getDefiningOp<LLVM::AllocaOp>()) {
            Type elemType = allocaOp.getElemType();
            if (isa<LLVM::LLVMStructType>(elemType)) {
              keySize = 0; // string-keyed
            } else {
              unsigned bits = getTypeWidth(elemType);
              keySize = std::max(1u, (bits + 7) / 8);
            }
          }
          void *newArray = __moore_assoc_create(keySize, valueSize);
          uint64_t newAddr = reinterpret_cast<uint64_t>(newArray);
          validAssocArrayAddresses.insert(newAddr);
          arrayAddr = newAddr;
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref - auto-created array at 0x"
                                  << llvm::format_hex(newAddr, 16) << "\n");
          // Store the new array pointer back to the source memory location.
          if (auto loadOp = callOp.getOperand(0).getDefiningOp<LLVM::LoadOp>()) {
            InterpretedValue srcAddr = getValue(procId, loadOp.getAddr());
            if (!srcAddr.isX()) {
              uint64_t storeAddr = srcAddr.getUInt64();
              uint64_t blockOffset = 0;
              MemoryBlock *block = findMemoryBlockByAddress(storeAddr, procId, &blockOffset);
              if (block && block->initialized && blockOffset + 8 <= block->data.size()) {
                std::memcpy(block->data.data() + blockOffset, &newAddr, 8);
              } else {
                auto nmIt = nativeMemoryBlocks.find(storeAddr);
                if (nmIt != nativeMemoryBlocks.end())
                  std::memcpy(reinterpret_cast<void *>(storeAddr), &newAddr, 8);
              }
            }
          }
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);

        // Read key from interpreter memory into a local buffer
        uint64_t keyOffset = 0;
        auto *keyBlock = findMemoryBlockByAddress(keyAddr, procId, &keyOffset);

        // We need to read the key data and pass a pointer to it.
        // For string keys: {ptr, i64} (16 bytes)
        // For integer keys: i8/i16/i32/i64 (1-8 bytes)
        // The runtime uses keySize stored in array header to know which.
        uint8_t keyBuffer[16] = {0};
        void *keyPtr = keyBuffer;
        MooreString keyString = {nullptr, 0};
        std::string keyStorage;

        if (keyBlock && keyBlock->initialized) {
          // Copy up to 16 bytes of key data
          // Safety check: make sure we don't underflow
          if (keyOffset > keyBlock->data.size()) {
            setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
            return success();
          }
          size_t maxCopy = std::min<size_t>(16, keyBlock->data.size() - keyOffset);
          std::memcpy(keyBuffer, keyBlock->data.data() + keyOffset, maxCopy);

          // Check if this looks like a string key ({ptr, len} struct)
          // by reading the header and checking keySize
          if (arrayPtr) {
            auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
            if (header->type == AssocArrayType_StringKey) {
              // For string keys, interpret as MooreString
              uint64_t strPtrVal = 0;
              int64_t strLen = 0;
              for (int i = 0; i < 8; ++i) {
                strPtrVal |= static_cast<uint64_t>(keyBuffer[i]) << (i * 8);
                strLen |= static_cast<int64_t>(keyBuffer[8 + i]) << (i * 8);
              }
              if (!tryReadStringKey(procId, strPtrVal, strLen, keyStorage)) {
                LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref "
                                           "string key not readable\n");
                setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
                return success();
              }
              keyString.data = keyStorage.data();
              keyString.len = keyStorage.size();
              keyPtr = &keyString;

              LLVM_DEBUG({
                llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref string key: ptr=0x"
                            << llvm::format_hex(strPtrVal, 16) << " len=" << strLen;
                if (keyString.data && keyString.len > 0) {
                  llvm::dbgs() << " = \"";
                  llvm::dbgs().write(keyString.data, std::min<int64_t>(keyString.len, 100));
                  llvm::dbgs() << "\"";
                }
                llvm::dbgs() << "\n";
              });
            } else {
              // Integer key - keyBuffer already contains the value
              LLVM_DEBUG({
                int64_t intKey = 0;
                std::memcpy(&intKey, keyBuffer, std::min<size_t>(8, maxCopy));
                llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref int key: " << intKey << "\n";
              });
            }
          }
        }

        // MooreToCore sometimes uses valueSize=4 for pointer-valued arrays
        // (e.g., m_domains maps string -> uvm_domain handle). The actual stores
        // are 8 bytes (pointer-sized), so we need at least 8 bytes to avoid
        // the native store handler silently dropping the store due to bounds
        // checking.
        int32_t effectiveSize = std::max(valueSize, static_cast<int32_t>(8));
        LLVM_DEBUG(if (effectiveSize != valueSize) {
          llvm::dbgs() << "  assoc_get_ref: expanded valueSize from "
                       << valueSize << " to " << effectiveSize << " for array 0x"
                       << llvm::format_hex(arrayAddr, 16) << "\n";
        });
        void *resultPtr = __moore_assoc_get_ref(arrayPtr, keyPtr, effectiveSize);
        uint64_t resultVal = reinterpret_cast<uint64_t>(resultPtr);

        if (resultPtr && effectiveSize > 0) {
          size_t size = static_cast<size_t>(effectiveSize);
          auto it = nativeMemoryBlocks.find(resultVal);
          if (it == nativeMemoryBlocks.end() || it->second < size)
            nativeMemoryBlocks[resultVal] = size;
        }

        setValue(procId, callOp.getResult(), InterpretedValue(resultVal, 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_get_ref(0x"
                                << llvm::format_hex(arrayAddr, 16) << ", 0x"
                                << llvm::format_hex(keyAddr, 16) << ", "
                                << valueSize << ") = 0x"
                                << llvm::format_hex(resultVal, 16) << "\n");
      }
      return success();
    }

    // Handle __moore_assoc_exists - check if key exists in associative array
    // Signature: (array: ptr, key: ptr) -> i32
    if (calleeName == "__moore_assoc_exists") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Validate that the array address is a properly initialized associative array.
        // Accept addresses tracked in validAssocArrayAddresses OR native C++ heap addresses.
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL; // 1TB
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 || (!validAssocArrayAddresses.contains(arrayAddr) && !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_exists - uninitialized array at 0x"
                                  << llvm::format_hex(arrayAddr, 16) << ", returning false\n");
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);

        // Read key from interpreter memory
        uint64_t keyOffset = 0;
        auto *keyBlock = findMemoryBlockByAddress(keyAddr, procId, &keyOffset);

        uint8_t keyBuffer[16] = {0};
        void *keyPtr = keyBuffer;
        MooreString keyString = {nullptr, 0};
        std::string keyStorage;

        if (keyBlock && keyBlock->initialized) {
          size_t maxCopy = std::min<size_t>(16, keyBlock->data.size() - keyOffset);
          std::memcpy(keyBuffer, keyBlock->data.data() + keyOffset, maxCopy);

          if (arrayPtr) {
            auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
            if (header->type == AssocArrayType_StringKey) {
              uint64_t strPtrVal = 0;
              int64_t strLen = 0;
              for (int i = 0; i < 8; ++i) {
                strPtrVal |= static_cast<uint64_t>(keyBuffer[i]) << (i * 8);
                strLen |= static_cast<int64_t>(keyBuffer[8 + i]) << (i * 8);
              }
              if (!tryReadStringKey(procId, strPtrVal, strLen, keyStorage)) {
                LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_exists "
                                           "string key not readable\n");
                setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
                return success();
              }
              keyString.data = keyStorage.data();
              keyString.len = keyStorage.size();
              keyPtr = &keyString;
            }
          }
        }

        int32_t result = __moore_assoc_exists(arrayPtr, keyPtr);

        // Debug: trace assoc_exists calls with string keys to diagnose NOCHILD
        setValue(procId, callOp.getResult(), InterpretedValue(static_cast<uint64_t>(result), 32));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_exists(0x"
                                << llvm::format_hex(arrayAddr, 16) << ", key) = "
                                << result << "\n");
      }
      return success();
    }

    // Handle __moore_assoc_first - get first key in associative array
    // Signature: (array: ptr, key_out: ptr) -> i1
    if (calleeName == "__moore_assoc_first") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyOutAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Validate that the array address is a properly initialized associative array.
        // Accept addresses tracked in validAssocArrayAddresses OR native C++ heap addresses.
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL; // 1TB
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 || (!validAssocArrayAddresses.contains(arrayAddr) && !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 1));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_first - uninitialized array at 0x"
                                  << llvm::format_hex(arrayAddr, 16) << ", returning false\n");
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);

        // Find key output memory block
        uint64_t keyOutOffset = 0;
        auto *keyOutBlock = findMemoryBlockByAddress(keyOutAddr, procId, &keyOutOffset);

        // Determine key type from array header
        bool isStringKey = false;
        auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
        isStringKey = (header->type == AssocArrayType_StringKey);

        bool result = false;
        if (isStringKey) {
          MooreString keyOut = {nullptr, 0};
          result = __moore_assoc_first(arrayPtr, &keyOut);

          if (result && keyOutBlock && keyOutOffset + 16 <= keyOutBlock->data.size()) {
            uint64_t ptrVal = reinterpret_cast<uint64_t>(keyOut.data);
            int64_t lenVal = keyOut.len;
            for (int i = 0; i < 8; ++i) {
              keyOutBlock->data[keyOutOffset + i] = static_cast<uint8_t>((ptrVal >> (i * 8)) & 0xFF);
              keyOutBlock->data[keyOutOffset + 8 + i] = static_cast<uint8_t>((lenVal >> (i * 8)) & 0xFF);
            }
            keyOutBlock->initialized = true;
          }
        } else {
          // Integer key - pass pointer to memory block directly
          // The key size can be 1, 2, 4, or 8 bytes depending on the index type.
          uint8_t keyBuffer[8] = {0};
          result = __moore_assoc_first(arrayPtr, keyBuffer);

          if (result && keyOutBlock) {
            size_t availableBytes = keyOutBlock->data.size() - keyOutOffset;
            size_t copySize = std::min<size_t>(8, availableBytes);
            std::memcpy(keyOutBlock->data.data() + keyOutOffset, keyBuffer, copySize);
            keyOutBlock->initialized = true;
          }
        }

        setValue(procId, callOp.getResult(), InterpretedValue(result ? 1ULL : 0ULL, 1));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_first(0x"
                                << llvm::format_hex(arrayAddr, 16) << ") = "
                                << result << "\n");
      }
      return success();
    }

    // Handle __moore_assoc_next - get next key in associative array
    // Signature: (array: ptr, key_ref: ptr) -> i1
    if (calleeName == "__moore_assoc_next") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyRefAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Validate that the array address is a properly initialized associative array.
        // Accept addresses tracked in validAssocArrayAddresses OR native C++ heap addresses.
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL; // 1TB
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 || (!validAssocArrayAddresses.contains(arrayAddr) && !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 1));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_next - uninitialized array at 0x"
                                  << llvm::format_hex(arrayAddr, 16) << ", returning false\n");
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);

        // Find key memory block
        uint64_t keyRefOffset = 0;
        auto *keyRefBlock = findMemoryBlockByAddress(keyRefAddr, procId, &keyRefOffset);

        // Determine key type from array header
        auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
        bool isStringKey = (header->type == AssocArrayType_StringKey);

        bool result = false;
        if (isStringKey) {
          MooreString keyRef = {nullptr, 0};
          if (keyRefBlock && keyRefBlock->initialized && keyRefOffset + 16 <= keyRefBlock->data.size()) {
            uint64_t strPtrVal = 0;
            int64_t strLen = 0;
            for (int i = 0; i < 8; ++i) {
              strPtrVal |= static_cast<uint64_t>(keyRefBlock->data[keyRefOffset + i]) << (i * 8);
              strLen |= static_cast<int64_t>(keyRefBlock->data[keyRefOffset + 8 + i]) << (i * 8);
            }
            keyRef.data = reinterpret_cast<char *>(strPtrVal);
            keyRef.len = strLen;
          }

          result = __moore_assoc_next(arrayPtr, &keyRef);

          if (result && keyRefBlock && keyRefOffset + 16 <= keyRefBlock->data.size()) {
            uint64_t ptrVal = reinterpret_cast<uint64_t>(keyRef.data);
            int64_t lenVal = keyRef.len;
            for (int i = 0; i < 8; ++i) {
              keyRefBlock->data[keyRefOffset + i] = static_cast<uint8_t>((ptrVal >> (i * 8)) & 0xFF);
              keyRefBlock->data[keyRefOffset + 8 + i] = static_cast<uint8_t>((lenVal >> (i * 8)) & 0xFF);
            }
          }
        } else {
          // Integer key - the key size can be 1, 2, 4, or 8 bytes.
          uint8_t keyBuffer[8] = {0};
          if (keyRefBlock && keyRefBlock->initialized) {
            size_t availableBytes = keyRefBlock->data.size() - keyRefOffset;
            size_t readSize = std::min<size_t>(8, availableBytes);
            std::memcpy(keyBuffer, keyRefBlock->data.data() + keyRefOffset, readSize);
          }

          int64_t keyBefore = 0;
          std::memcpy(&keyBefore, keyBuffer, 8);

          result = __moore_assoc_next(arrayPtr, keyBuffer);

          if (result && keyRefBlock) {
            size_t availableBytes = keyRefBlock->data.size() - keyRefOffset;
            size_t copySize = std::min<size_t>(8, availableBytes);
            std::memcpy(keyRefBlock->data.data() + keyRefOffset, keyBuffer, copySize);
          }
        }

        setValue(procId, callOp.getResult(), InterpretedValue(result ? 1ULL : 0ULL, 1));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_next(0x"
                                << llvm::format_hex(arrayAddr, 16) << ") = "
                                << result << "\n");
      }
      return success();
    }

    // Handle __moore_assoc_size - get size of associative array
    // Signature: (array: ptr) -> i64
    if (calleeName == "__moore_assoc_size") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();

        // Validate that the array address is a properly initialized associative array.
        // Accept addresses tracked in validAssocArrayAddresses OR native C++ heap addresses.
        // Native heap addresses (> 0x10000000000) may be created in global constructors
        // before the interpreter starts tracking, so we allow them through.
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL; // 1TB
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 || (!validAssocArrayAddresses.contains(arrayAddr) && !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 64));
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_size - uninitialized array at 0x"
                                  << llvm::format_hex(arrayAddr, 16) << ", returning 0\n");
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);

        int64_t result = __moore_assoc_size(arrayPtr);

        setValue(procId, callOp.getResult(), InterpretedValue(static_cast<uint64_t>(result), 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_size(0x"
                                << llvm::format_hex(arrayAddr, 16) << ") = "
                                << result << "\n");
      }
      return success();
    }

    //===------------------------------------------------------------------===//
    // Mailbox DPI Hooks (Phase 1 - Non-blocking operations)
    //===------------------------------------------------------------------===//

    // Handle __moore_mailbox_create - create a new mailbox
    // Signature: (bound: i32) -> i64
    if (calleeName == "__moore_mailbox_create") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        int32_t bound = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());

        MailboxId mboxId = syncPrimitivesManager.createMailbox(bound);

        setValue(procId, callOp.getResult(),
                 InterpretedValue(static_cast<uint64_t>(mboxId), 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_create("
                                << bound << ") = " << mboxId << "\n");
      }
      return success();
    }

    auto wakePeekWaiters = [&](Mailbox *mbox, MailboxId mboxId) {
      if (!mbox || mbox->isEmpty())
        return;
      llvm::SmallVector<ProcessId, 4> peekWaiters;
      mbox->takePeekWaiters(peekWaiters);
      if (peekWaiters.empty())
        return;

      uint64_t peekMsg = 0;
      if (!mbox->tryPeek(peekMsg))
        return;

      for (ProcessId waiterId : peekWaiters) {
        auto waiterIt = processStates.find(waiterId);
        if (waiterIt == processStates.end())
          continue;
        auto &waiterState = waiterIt->second;
        if (waiterState.pendingMailboxPeekResultAddr != 0) {
          uint64_t outOffset = 0;
          auto *outBlock = findMemoryBlockByAddress(
              waiterState.pendingMailboxPeekResultAddr, waiterId, &outOffset);
          if (outBlock) { outBlock->initialized = true;
            for (int i = 0; i < 8; ++i)
              outBlock->data[outOffset + i] =
                  static_cast<uint8_t>((peekMsg >> (i * 8)) & 0xFF);
          }
        }
        waiterState.pendingMailboxPeekResultAddr = 0;
        waiterState.pendingMailboxPeekId = 0;
        waiterState.waiting = false;
        scheduler.scheduleProcess(waiterId, SchedulingRegion::Active);
        LLVM_DEBUG(llvm::dbgs() << "    Woke peek waiter process "
                                << waiterId << " on mailbox " << mboxId
                                << " with message " << peekMsg << "\n");
      }
    };

    // Handle __moore_mailbox_tryput - non-blocking put
    // Signature: (mbox_id: i64, msg: i64) -> i1
    if (calleeName == "__moore_mailbox_tryput") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msg = getValue(procId, callOp.getOperand(1)).getUInt64();

        bool putSuccess = syncPrimitivesManager.mailboxTryPut(mboxId, msg);

        setValue(procId, callOp.getResult(),
                 InterpretedValue(putSuccess ? 1ULL : 0ULL, 1));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_tryput("
                                << mboxId << ", " << msg << ") = "
                                << (putSuccess ? "true" : "false") << "\n");

        // If put succeeded, try to wake a waiting get process
        if (putSuccess) {
          Mailbox *mbox = syncPrimitivesManager.getMailbox(mboxId);
          if (mbox) {
            uint64_t waiterMsg = 0;
            ProcessId waiterId = mbox->trySatisfyGetWaiter(waiterMsg);
            if (waiterId != InvalidProcessId) {
              // Found a waiter - write the message to their output address
              auto waiterIt = processStates.find(waiterId);
              if (waiterIt != processStates.end()) {
                auto &waiterState = waiterIt->second;
                if (waiterState.pendingMailboxGetResultAddr != 0) {
                  uint64_t outOffset = 0;
                  auto *outBlock = findMemoryBlockByAddress(
                      waiterState.pendingMailboxGetResultAddr, waiterId, &outOffset);
                  if (outBlock) { outBlock->initialized = true;
                    for (int i = 0; i < 8; ++i)
                      outBlock->data[outOffset + i] =
                          static_cast<uint8_t>((waiterMsg >> (i * 8)) & 0xFF);
                  }
                }
                waiterState.pendingMailboxGetResultAddr = 0;
                waiterState.pendingMailboxGetId = 0;
                waiterState.waiting = false;
                scheduler.scheduleProcess(waiterId, SchedulingRegion::Active);
                LLVM_DEBUG(llvm::dbgs() << "    Woke get waiter process "
                                        << waiterId << " with message "
                                        << waiterMsg << "\n");
              }
            }
            wakePeekWaiters(mbox, mboxId);
          }
        }
      }
      return success();
    }

    // Handle __moore_mailbox_tryget - non-blocking get
    // Signature: (mbox_id: i64, msg_out: ptr) -> i1
    if (calleeName == "__moore_mailbox_tryget") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msgOutAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        uint64_t msg = 0;
        bool getSuccess = syncPrimitivesManager.mailboxTryGet(mboxId, msg);

        // Write the message to the output pointer if successful
        if (getSuccess && msgOutAddr != 0) {
          uint64_t outOffset = 0;
          auto *outBlock = findMemoryBlockByAddress(msgOutAddr, procId, &outOffset);
          if (outBlock) { outBlock->initialized = true;
            // Write 64-bit message value
            for (int i = 0; i < 8; ++i)
              outBlock->data[outOffset + i] =
                  static_cast<uint8_t>((msg >> (i * 8)) & 0xFF);
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(getSuccess ? 1ULL : 0ULL, 1));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_tryget("
                                << mboxId << ", 0x"
                                << llvm::format_hex(msgOutAddr, 16) << ") = "
                                << (getSuccess ? "true" : "false");
                   if (getSuccess) llvm::dbgs() << ", msg=" << msg;
                   llvm::dbgs() << "\n");

        // If get succeeded, try to wake a waiting put process (for bounded mailboxes)
        if (getSuccess) {
          Mailbox *mbox = syncPrimitivesManager.getMailbox(mboxId);
          if (mbox) {
            ProcessId waiterId = mbox->trySatisfyPutWaiter();
            if (waiterId != InvalidProcessId) {
              // Found a put waiter - their message was already added by trySatisfyPutWaiter
              auto waiterIt = processStates.find(waiterId);
              if (waiterIt != processStates.end()) {
                waiterIt->second.waiting = false;
                scheduler.scheduleProcess(waiterId, SchedulingRegion::Active);
                LLVM_DEBUG(llvm::dbgs() << "    Woke put waiter process "
                                        << waiterId << "\n");
              }
            }
            wakePeekWaiters(mbox, mboxId);
          }
        }
      }
      return success();
    }

    // Handle __moore_mailbox_trypeek - non-blocking peek (no removal)
    // Signature: (mbox_id: i64, msg_out: ptr) -> i1
    if (calleeName == "__moore_mailbox_trypeek") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msgOutAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        uint64_t msg = 0;
        bool peekSuccess = syncPrimitivesManager.mailboxPeek(mboxId, msg);

        // Write the message to the output pointer if successful
        if (peekSuccess && msgOutAddr != 0) {
          uint64_t outOffset = 0;
          auto *outBlock = findMemoryBlockByAddress(msgOutAddr, procId, &outOffset);
          if (outBlock) { outBlock->initialized = true;
            // Write 64-bit message value
            for (int i = 0; i < 8; ++i)
              outBlock->data[outOffset + i] =
                  static_cast<uint8_t>((msg >> (i * 8)) & 0xFF);
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(peekSuccess ? 1ULL : 0ULL, 1));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_trypeek("
                                << mboxId << ", 0x"
                                << llvm::format_hex(msgOutAddr, 16) << ") = "
                                << (peekSuccess ? "true" : "false");
                   if (peekSuccess) llvm::dbgs() << ", msg=" << msg;
                   llvm::dbgs() << "\n");
      }
      return success();
    }

    // Handle __moore_mailbox_num - get message count
    // Signature: (mbox_id: i64) -> i64
    if (calleeName == "__moore_mailbox_num") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());

        size_t count = syncPrimitivesManager.mailboxNum(mboxId);

        setValue(procId, callOp.getResult(),
                 InterpretedValue(static_cast<uint64_t>(count), 64));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_num("
                                << mboxId << ") = " << count << "\n");
      }
      return success();
    }

    // Handle __moore_mailbox_put - blocking put
    // Signature: (mbox_id: i64, msg: i64) -> void
    // Blocks if the mailbox is full (bounded mailbox)
    if (calleeName == "__moore_mailbox_put") {
      if (callOp.getNumOperands() >= 2) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msg = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Try non-blocking put first
        if (syncPrimitivesManager.mailboxTryPut(mboxId, msg)) {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_put("
                                  << mboxId << ", " << msg
                                  << ") - message sent immediately\n");

          // Try to wake a waiting get process
          Mailbox *mbox = syncPrimitivesManager.getMailbox(mboxId);
          if (mbox) {
            uint64_t waiterMsg = 0;
            ProcessId waiterId = mbox->trySatisfyGetWaiter(waiterMsg);
            if (waiterId != InvalidProcessId) {
              // Found a waiter - write the message to their output address
              auto waiterIt = processStates.find(waiterId);
              if (waiterIt != processStates.end()) {
                auto &waiterState = waiterIt->second;
                if (waiterState.pendingMailboxGetResultAddr != 0) {
                  uint64_t outOffset = 0;
                  auto *outBlock = findMemoryBlockByAddress(
                      waiterState.pendingMailboxGetResultAddr, waiterId, &outOffset);
                  if (outBlock) { outBlock->initialized = true;
                    for (int i = 0; i < 8; ++i)
                      outBlock->data[outOffset + i] =
                          static_cast<uint8_t>((waiterMsg >> (i * 8)) & 0xFF);
                  }
                }
                waiterState.pendingMailboxGetResultAddr = 0;
                waiterState.pendingMailboxGetId = 0;
                waiterState.waiting = false;
                scheduler.scheduleProcess(waiterId, SchedulingRegion::Active);
                LLVM_DEBUG(llvm::dbgs() << "    Woke get waiter process "
                                        << waiterId << " with message "
                                        << waiterMsg << "\n");
              }
            }
            wakePeekWaiters(mbox, mboxId);
          }
        } else {
          // Mailbox is full - block until space is available
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_put("
                                  << mboxId << ", " << msg
                                  << ") - blocking (mailbox full)\n");

          // Add this process to the put wait queue
          syncPrimitivesManager.mailboxPut(mboxId, procId, msg);

          // Suspend the process
          auto &state = processStates[procId];
          state.waiting = true;

          // Set up to resume at the next operation after the call
          state.destBlock = callOp->getBlock();
          state.currentOp = std::next(Block::iterator(callOp));
          state.resumeAtCurrentOp = true;

          Process *proc = scheduler.getProcess(procId);
          if (proc)
            proc->setState(ProcessState::Waiting);
        }
      }
      return success();
    }

    // Handle __moore_mailbox_get - blocking get
    // Signature: (mbox_id: i64, msg_out: ptr) -> void
    // Blocks if the mailbox is empty
    if (calleeName == "__moore_mailbox_get") {
      if (callOp.getNumOperands() >= 2) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msgOutAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Try non-blocking get first
        uint64_t msg = 0;
        if (syncPrimitivesManager.mailboxTryGet(mboxId, msg)) {
          // Got a message - write it to the output pointer
          if (msgOutAddr != 0) {
            uint64_t outOffset = 0;
            auto *outBlock = findMemoryBlockByAddress(msgOutAddr, procId, &outOffset);
            if (outBlock) { outBlock->initialized = true;
              // Write 64-bit message value
              for (int i = 0; i < 8; ++i)
                outBlock->data[outOffset + i] =
                    static_cast<uint8_t>((msg >> (i * 8)) & 0xFF);
            }
          }

          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_get("
                                  << mboxId << ", 0x"
                                  << llvm::format_hex(msgOutAddr, 16)
                                  << ") - got message " << msg << "\n");

          // Try to wake a waiting put process (for bounded mailboxes)
          Mailbox *mbox = syncPrimitivesManager.getMailbox(mboxId);
          if (mbox) {
            ProcessId waiterId = mbox->trySatisfyPutWaiter();
            if (waiterId != InvalidProcessId) {
              // Found a put waiter - their message was already added
              auto waiterIt = processStates.find(waiterId);
              if (waiterIt != processStates.end()) {
                waiterIt->second.waiting = false;
                scheduler.scheduleProcess(waiterId, SchedulingRegion::Active);
                LLVM_DEBUG(llvm::dbgs() << "    Woke put waiter process "
                                        << waiterId << "\n");
              }
            }
            wakePeekWaiters(mbox, mboxId);
          }
        } else {
          // Mailbox is empty - block until a message is available
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_get("
                                  << mboxId << ", 0x"
                                  << llvm::format_hex(msgOutAddr, 16)
                                  << ") - blocking (mailbox empty)\n");

          // Add this process to the get wait queue
          syncPrimitivesManager.mailboxGet(mboxId, procId);

          // Save the output address so we can write the result when resumed
          auto &state = processStates[procId];
          state.pendingMailboxGetResultAddr = msgOutAddr;
          state.pendingMailboxGetId = mboxId;
          state.waiting = true;

          // Set up to resume at the next operation after the call
          state.destBlock = callOp->getBlock();
          state.currentOp = std::next(Block::iterator(callOp));
          state.resumeAtCurrentOp = true;

          Process *proc = scheduler.getProcess(procId);
          if (proc)
            proc->setState(ProcessState::Waiting);
        }
      }
      return success();
    }

    // Handle __moore_mailbox_peek - blocking peek (no removal)
    // Signature: (mbox_id: i64, msg_out: ptr) -> void
    // Blocks if the mailbox is empty
    if (calleeName == "__moore_mailbox_peek") {
      if (callOp.getNumOperands() >= 2) {
        MailboxId mboxId = static_cast<MailboxId>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t msgOutAddr = getValue(procId, callOp.getOperand(1)).getUInt64();

        // Try non-blocking peek first
        uint64_t msg = 0;
        if (syncPrimitivesManager.mailboxPeek(mboxId, msg)) {
          // Got a message - write it to the output pointer
          if (msgOutAddr != 0) {
            uint64_t outOffset = 0;
            auto *outBlock = findMemoryBlockByAddress(msgOutAddr, procId, &outOffset);
            if (outBlock) { outBlock->initialized = true;
              // Write 64-bit message value
              for (int i = 0; i < 8; ++i)
                outBlock->data[outOffset + i] =
                    static_cast<uint8_t>((msg >> (i * 8)) & 0xFF);
            }
          }

          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_peek("
                                  << mboxId << ", 0x"
                                  << llvm::format_hex(msgOutAddr, 16)
                                  << ") - got message " << msg << "\n");
        } else {
          // Mailbox is empty - block until a message is available
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_mailbox_peek("
                                  << mboxId << ", 0x"
                                  << llvm::format_hex(msgOutAddr, 16)
                                  << ") - blocking (mailbox empty)\n");

          Mailbox *mbox = syncPrimitivesManager.getOrCreateMailbox(mboxId);
          if (mbox)
            mbox->addPeekWaiter(procId);

          // Save the output address so we can write the result when resumed
          auto &state = processStates[procId];
          state.pendingMailboxPeekResultAddr = msgOutAddr;
          state.pendingMailboxPeekId = mboxId;
          state.waiting = true;

          // Set up to resume at the next operation after the call
          state.destBlock = callOp->getBlock();
          state.currentOp = std::next(Block::iterator(callOp));
          state.resumeAtCurrentOp = true;

          Process *proc = scheduler.getProcess(procId);
          if (proc)
            proc->setState(ProcessState::Waiting);
        }
      }
      return success();
    }

    // Handle __moore_int_to_string - convert i64 to string struct {ptr, i64}
    if (calleeName == "__moore_int_to_string" ||
        calleeName == "__moore_string_itoa") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue intArg = getValue(procId, callOp.getOperand(0));
        std::string str;
        if (intArg.isX()) {
          str = "x";
        } else {
          str = std::to_string(intArg.getAPInt().getSExtValue());
        }

        // Store the string in persistent storage
        interpreterStrings.push_back(std::move(str));
        const std::string &stored = interpreterStrings.back();
        int64_t ptrVal = reinterpret_cast<int64_t>(stored.data());
        int64_t lenVal = static_cast<int64_t>(stored.size());

        // Register in dynamic strings registry
        dynamicStrings[ptrVal] = {stored.data(), lenVal};

        // Pack into 128-bit struct result {ptr(lower 64), len(upper 64)}
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(ptrVal)), 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(lenVal)), 64);
        setValue(procId, callOp.getResult(), InterpretedValue(packedResult));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_int_to_string("
                                << (intArg.isX() ? "X" : std::to_string(intArg.getAPInt().getSExtValue()))
                                << ") = \"" << stored << "\"\n");
      }
      return success();
    }

    // Handle __moore_readmemh / __moore_readmemb - load memory from file
    // Signature: void(filename_ptr, mem_ptr, elem_width_i32, num_elems_i32)
    // filename_ptr points to a stack-allocated {ptr, i64} string struct
    // mem_ptr is the !llhd.ref (signal) or alloca ptr for the memory array
    // elem_width is the logical bit width of each element (e.g. 8 for logic [7:0])
    // num_elems is the number of array elements
    if (calleeName == "__moore_readmemh" ||
        calleeName == "__moore_readmemb") {
      bool isHex = (calleeName == "__moore_readmemh");
      if (callOp.getNumOperands() >= 4) {
        // Extract filename from the string struct pointer (arg 0)
        InterpretedValue filenamePtrVal = getValue(procId, callOp.getOperand(0));
        std::string filename;
        if (!filenamePtrVal.isX()) {
          uint64_t structAddr = filenamePtrVal.getUInt64();
          uint64_t structOffset = 0;
          auto *block = findMemoryBlockByAddress(structAddr, procId, &structOffset);
          if (block && block->initialized && structOffset + 16 <= block->data.size()) {
            uint64_t strPtr = 0;
            int64_t strLen = 0;
            for (int i = 0; i < 8; ++i) {
              strPtr |= static_cast<uint64_t>(block->data[structOffset + i]) << (i * 8);
              strLen |= static_cast<int64_t>(block->data[structOffset + 8 + i]) << (i * 8);
            }
            if (strPtr != 0 && strLen > 0) {
              // Look up in dynamicStrings first
              auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtr));
              if (dynIt != dynamicStrings.end() && dynIt->second.first) {
                filename = std::string(dynIt->second.first,
                    std::min(static_cast<size_t>(strLen),
                             static_cast<size_t>(dynIt->second.second)));
              } else {
                // Try global memory
                uint64_t strOffset = 0;
                auto *strBlock = findMemoryBlockByAddress(strPtr, procId, &strOffset);
                if (strBlock && strBlock->initialized &&
                    strOffset + strLen <= strBlock->data.size()) {
                  filename = std::string(
                      reinterpret_cast<const char *>(strBlock->data.data() + strOffset),
                      strLen);
                }
              }
            }
          }
        }

        // Get elem_width and num_elems from args 2 and 3
        InterpretedValue elemWidthVal = getValue(procId, callOp.getOperand(2));
        InterpretedValue numElemsVal = getValue(procId, callOp.getOperand(3));
        unsigned elemBitWidth = elemWidthVal.isX() ? 0 : static_cast<unsigned>(elemWidthVal.getUInt64());
        unsigned numElems = numElemsVal.isX() ? 0 : static_cast<unsigned>(numElemsVal.getUInt64());

        if (filename.empty() || elemBitWidth == 0 || numElems == 0) {
          LLVM_DEBUG(llvm::dbgs() << "  llvm.call: " << calleeName
                                  << " - invalid args (filename=\"" << filename
                                  << "\", elemWidth=" << elemBitWidth
                                  << ", numElems=" << numElems << ")\n");
          return success();
        }

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: " << calleeName
                                << "(\"" << filename << "\", elemWidth="
                                << elemBitWidth << ", numElems=" << numElems
                                << ")\n");

        // Parse the file - each line may contain: @addr, value, or // comment
        std::ifstream file(filename);
        if (!file.is_open()) {
          llvm::errs() << "Warning: " << (isHex ? "$readmemh" : "$readmemb")
                       << ": cannot open file \"" << filename << "\"\n";
          return success();
        }

        // Parse values from the file
        std::vector<std::pair<unsigned, APInt>> indexedValues;
        unsigned currentAddr = 0;
        std::string line;
        while (std::getline(file, line)) {
          // Strip comments (// style and /* */ style)
          auto commentPos = line.find("//");
          if (commentPos != std::string::npos)
            line = line.substr(0, commentPos);

          std::istringstream iss(line);
          std::string token;
          while (iss >> token) {
            if (token.empty())
              continue;

            // Address specification: @hex_addr
            if (token[0] == '@') {
              std::string addrStr = token.substr(1);
              currentAddr = std::stoul(addrStr, nullptr, 16);
              continue;
            }

            // Parse value token
            APInt val(elemBitWidth, 0);
            bool valid = true;
            if (isHex) {
              // Hex format - allow x/X/z/Z characters (treat as 0)
              std::string cleanToken;
              for (char c : token) {
                if (c == '_') continue; // Skip underscores
                if (c == 'x' || c == 'X' || c == 'z' || c == 'Z')
                  cleanToken += '0';
                else if (std::isxdigit(c))
                  cleanToken += c;
                else {
                  valid = false;
                  break;
                }
              }
              if (valid && !cleanToken.empty()) {
                // Parse hex value, truncating/extending to elemBitWidth
                unsigned numBits = cleanToken.size() * 4;
                if (numBits > elemBitWidth)
                  numBits = elemBitWidth;
                APInt parsed(std::max(numBits, elemBitWidth), cleanToken, 16);
                val = parsed.trunc(elemBitWidth);
              } else {
                valid = false;
              }
            } else {
              // Binary format - allow x/X/z/Z characters (treat as 0)
              std::string cleanToken;
              for (char c : token) {
                if (c == '_') continue;
                if (c == 'x' || c == 'X' || c == 'z' || c == 'Z')
                  cleanToken += '0';
                else if (c == '0' || c == '1')
                  cleanToken += c;
                else {
                  valid = false;
                  break;
                }
              }
              if (valid && !cleanToken.empty()) {
                unsigned numBits = cleanToken.size();
                if (numBits > elemBitWidth)
                  numBits = elemBitWidth;
                APInt parsed(std::max(numBits, elemBitWidth), cleanToken, 2);
                val = parsed.trunc(elemBitWidth);
              } else {
                valid = false;
              }
            }

            if (valid && currentAddr < numElems) {
              indexedValues.push_back({currentAddr, val});
              LLVM_DEBUG(llvm::dbgs() << "    [" << currentAddr << "] = 0x"
                                      << llvm::format_hex_no_prefix(
                                             val.getZExtValue(), (elemBitWidth + 3) / 4)
                                      << "\n");
              ++currentAddr;
            }
          }
        }
        file.close();

        if (indexedValues.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "    No values parsed from file\n");
          return success();
        }

        // Now write the parsed values into the memory array.
        // The mem_ptr operand (arg 1) traces back to the !llhd.ref signal.
        Value memOperand = callOp.getOperand(1);
        SignalId sigId = resolveSignalId(memOperand);

        if (sigId != 0) {
          // Signal-backed memory: read current value, modify, write back
          const SignalValue &currentSV = scheduler.getSignalValue(sigId);
          unsigned totalWidth = currentSV.getWidth();
          APInt arrayBits = currentSV.isUnknown()
                                ? APInt(totalWidth, 0)
                                : currentSV.getAPInt();

          // Determine the total element width in the signal (including unknown bits)
          // For 4-state types: totalElemWidth = 2 * elemBitWidth
          // (struct<value: iN, unknown: iN>)
          unsigned totalElemWidth = totalWidth / numElems;

          for (auto &[idx, val] : indexedValues) {
            if (idx >= numElems)
              continue;
            // MooreToCore maps SV index i to hw.array index (N-1-i).
            // SV mem[0] is at the MSB end (hw.array index N-1).
            unsigned hwIdx = numElems - 1 - idx;
            unsigned elemBase = hwIdx * totalElemWidth;

            if (totalElemWidth == 2 * elemBitWidth) {
              // 4-state element: struct<value: iN, unknown: iN>
              // HW struct layout: field 0 ("value") at high bits,
              //                   field 1 ("unknown") at low bits
              // Set value bits (upper half of element) to the parsed value
              APInt valBits = val;
              if (valBits.getBitWidth() < elemBitWidth)
                valBits = valBits.zext(elemBitWidth);
              else if (valBits.getBitWidth() > elemBitWidth)
                valBits = valBits.trunc(elemBitWidth);
              arrayBits.insertBits(valBits, elemBase + elemBitWidth);
              // Clear unknown bits (lower half of element)
              APInt zeroBits(elemBitWidth, 0);
              arrayBits.insertBits(zeroBits, elemBase);
            } else {
              // 2-state or other: just insert the value directly
              APInt valBits = val;
              if (valBits.getBitWidth() < totalElemWidth)
                valBits = valBits.zext(totalElemWidth);
              else if (valBits.getBitWidth() > totalElemWidth)
                valBits = valBits.trunc(totalElemWidth);
              arrayBits.insertBits(valBits, elemBase);
            }
          }

          scheduler.updateSignal(sigId, SignalValue(arrayBits));
          LLVM_DEBUG(llvm::dbgs() << "    Updated signal " << sigId
                                  << " with " << indexedValues.size()
                                  << " values\n");
        } else {
          // Alloca-backed memory: write values to the memory block
          InterpretedValue memPtrVal = getValue(procId, memOperand);
          if (!memPtrVal.isX()) {
            uint64_t memAddr = memPtrVal.getUInt64();
            uint64_t memOffset = 0;
            auto *memBlock = findMemoryBlockByAddress(memAddr, procId, &memOffset);
            if (memBlock && memBlock->initialized) {
              // For alloca-backed memory, the layout is LLVM-style:
              // each element is stored as a contiguous block of bytes.
              // For 4-state types in LLVM layout: struct{iN, iN} -> {value, unknown}
              // In LLVM layout, field 0 is at offset 0 (low bytes).
              unsigned totalElemWidth = (memBlock->data.size() - memOffset) * 8 / numElems;
              unsigned elemBytes = totalElemWidth / 8;

              for (auto &[idx, val] : indexedValues) {
                if (idx >= numElems)
                  continue;
                unsigned byteBase = memOffset + idx * elemBytes;
                unsigned valueBytes = (elemBitWidth + 7) / 8;

                // Write value bytes (in LLVM layout, value field is first)
                APInt valBits = val;
                if (valBits.getBitWidth() < elemBitWidth)
                  valBits = valBits.zext(elemBitWidth);
                for (unsigned b = 0; b < valueBytes && byteBase + b < memBlock->data.size(); ++b) {
                  memBlock->data[byteBase + b] =
                      static_cast<uint8_t>(valBits.extractBitsAsZExtValue(8, b * 8));
                }
                // Clear unknown bytes (in LLVM layout, unknown field follows value)
                for (unsigned b = valueBytes; b < elemBytes && byteBase + b < memBlock->data.size(); ++b) {
                  memBlock->data[byteBase + b] = 0;
                }
              }

              LLVM_DEBUG(llvm::dbgs() << "    Updated alloca memory at 0x"
                                      << llvm::format_hex(memAddr, 16)
                                      << " with " << indexedValues.size()
                                      << " values\n");
            }
          }
        }
      }
      return success();
    }

    // Handle __moore_string_concat - concatenate two string structs
    // Signature: (lhs_ptr: ptr, rhs_ptr: ptr) -> struct{ptr, i64}
    // lhs_ptr and rhs_ptr point to stack-allocated {ptr, i64} structs
    if (calleeName == "__moore_string_concat") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        // Helper to read a string struct from a pointer address
        auto readStringFromStructPtr = [&](Value operand) -> std::string {
          InterpretedValue ptrArg = getValue(procId, operand);
          if (ptrArg.isX())
            return "";

          uint64_t structAddr = ptrArg.getUInt64();
          uint64_t structOffset = 0;
          auto *block = findMemoryBlockByAddress(structAddr, procId, &structOffset);
          if (!block || !block->initialized || structOffset + 16 > block->data.size())
            return "";

          // Read ptr (first 8 bytes, little-endian) and len (next 8 bytes)
          uint64_t strPtr = 0;
          int64_t strLen = 0;
          for (int i = 0; i < 8; ++i) {
            strPtr |= static_cast<uint64_t>(block->data[structOffset + i]) << (i * 8);
            strLen |= static_cast<int64_t>(block->data[structOffset + 8 + i]) << (i * 8);
          }

          if (strPtr == 0 || strLen <= 0)
            return "";

          // Look up in dynamicStrings registry first
          auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtr));
          if (dynIt != dynamicStrings.end() && dynIt->second.first &&
              dynIt->second.second > 0) {
            return std::string(dynIt->second.first,
                               std::min(static_cast<size_t>(strLen),
                                        static_cast<size_t>(dynIt->second.second)));
          }

          // Try global memory lookup (for string literals)
          for (auto &[globalName, globalAddr] : globalAddresses) {
            auto blockIt = globalMemoryBlocks.find(globalName);
            if (blockIt != globalMemoryBlocks.end()) {
              MemoryBlock &gBlock = blockIt->second;
              if (strPtr >= globalAddr && strPtr < globalAddr + gBlock.size) {
                uint64_t off = strPtr - globalAddr;
                size_t avail = std::min(static_cast<size_t>(strLen),
                                        gBlock.data.size() - static_cast<size_t>(off));
                if (avail > 0 && gBlock.initialized)
                  return std::string(reinterpret_cast<const char *>(
                                         gBlock.data.data() + off),
                                     avail);
              }
            }
          }

          // Try addressToGlobal reverse lookup
          auto globalIt = addressToGlobal.find(strPtr);
          if (globalIt != addressToGlobal.end()) {
            auto blockIt = globalMemoryBlocks.find(globalIt->second);
            if (blockIt != globalMemoryBlocks.end()) {
              const MemoryBlock &gBlock = blockIt->second;
              size_t avail = std::min(static_cast<size_t>(strLen),
                                      gBlock.data.size());
              if (avail > 0 && gBlock.initialized)
                return std::string(reinterpret_cast<const char *>(
                                       gBlock.data.data()),
                                   avail);
            }
          }

          return "";
        };

        std::string lhs = readStringFromStructPtr(callOp.getOperand(0));
        std::string rhs = readStringFromStructPtr(callOp.getOperand(1));
        std::string result = lhs + rhs;

        // Store the concatenated string in persistent storage
        interpreterStrings.push_back(std::move(result));
        const std::string &stored = interpreterStrings.back();
        int64_t ptrVal = reinterpret_cast<int64_t>(stored.data());
        int64_t lenVal = static_cast<int64_t>(stored.size());

        // Register in dynamic strings registry
        dynamicStrings[ptrVal] = {stored.data(), lenVal};

        // Pack into 128-bit struct result
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(ptrVal)), 0);
        packedResult.insertBits(APInt(64, static_cast<uint64_t>(lenVal)), 64);
        setValue(procId, callOp.getResult(), InterpretedValue(packedResult));

        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_concat(\"" << lhs
                                << "\", \"" << rhs << "\") = \"" << stored
                                << "\"\n");
      }
      return success();
    }

    // ---- __moore_dyn_cast_check ----
    if (calleeName == "__moore_dyn_cast_check") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        int32_t srcId = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        int32_t targetId = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int32_t inheritanceDepth = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());
        // Use RTTI parent table for correct hierarchy checking
        bool result = checkRTTICast(srcId, targetId);
        // Return type is i1 (bool) in the runtime, but MLIR may widen to i32
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, result ? 1 : 0)));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_dyn_cast_check(src=" << srcId
                   << ", target=" << targetId
                   << ", depth=" << inheritanceDepth << ") = " << result
                   << "\n");
      }
      return success();
    }

    // ---- __moore_urandom ----
    if (calleeName == "__moore_urandom") {
      uint32_t result = __moore_urandom();
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, result)));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_urandom() = " << result
                               << "\n");
      return success();
    }

    // ---- __moore_urandom_range ----
    if (calleeName == "__moore_urandom_range") {
      uint32_t maxVal = static_cast<uint32_t>(
          getValue(procId, callOp.getOperand(0)).getUInt64());
      uint32_t minVal = static_cast<uint32_t>(
          getValue(procId, callOp.getOperand(1)).getUInt64());
      uint32_t result = __moore_urandom_range(maxVal, minVal);
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, result)));
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_urandom_range(" << maxVal << ", "
                 << minVal << ") = " << result << "\n");
      return success();
    }

    // ---- __moore_random ----
    if (calleeName == "__moore_random") {
      int32_t result = __moore_random();
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, static_cast<uint32_t>(result))));
      LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_random() = " << result
                               << "\n");
      return success();
    }

    // ---- __moore_urandom_seeded ----
    if (calleeName == "__moore_urandom_seeded") {
      int32_t seed = static_cast<int32_t>(
          getValue(procId, callOp.getOperand(0)).getUInt64());
      uint32_t result = __moore_urandom_seeded(seed);
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, result)));
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_urandom_seeded(" << seed
                 << ") = " << result << "\n");
      return success();
    }

    // ---- __moore_random_seeded ----
    if (calleeName == "__moore_random_seeded") {
      int32_t seed = static_cast<int32_t>(
          getValue(procId, callOp.getOperand(0)).getUInt64());
      int32_t result = __moore_random_seeded(seed);
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, static_cast<uint32_t>(result))));
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_random_seeded(" << seed
                 << ") = " << result << "\n");
      return success();
    }

    // ---- __moore_randomize_basic ----
    if (calleeName == "__moore_randomize_basic") {
      // Stub: return 1 (success). Full constrained randomization would need
      // to write random values to the object's rand fields.
      setValue(procId, callOp.getResult(),
               InterpretedValue(APInt(32, 1)));
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_randomize_basic() = 1 (stub)\n");
      return success();
    }

    // ---- Helper lambda: extract Moore string from struct pointer ----
    // (used by fopen, fwrite, fclose)
    auto extractMooreStringFromPtr = [&](Value operand) -> std::string {
      InterpretedValue ptrArg = getValue(procId, operand);
      if (ptrArg.isX())
        return "";

      uint64_t structAddr = ptrArg.getUInt64();
      uint64_t structOffset = 0;
      auto *block =
          findMemoryBlockByAddress(structAddr, procId, &structOffset);
      if (!block || !block->initialized ||
          structOffset + 16 > block->data.size())
        return "";

      // Read ptr (first 8 bytes, little-endian) and len (next 8 bytes)
      uint64_t strPtr = 0;
      int64_t strLen = 0;
      for (int i = 0; i < 8; ++i) {
        strPtr |=
            static_cast<uint64_t>(block->data[structOffset + i]) << (i * 8);
        strLen |= static_cast<int64_t>(block->data[structOffset + 8 + i])
                  << (i * 8);
      }

      if (strPtr == 0 || strLen <= 0)
        return "";

      // Look up in dynamicStrings registry first
      auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtr));
      if (dynIt != dynamicStrings.end() && dynIt->second.first &&
          dynIt->second.second > 0) {
        return std::string(
            dynIt->second.first,
            std::min(static_cast<size_t>(strLen),
                     static_cast<size_t>(dynIt->second.second)));
      }

      // Try global memory lookup (for string literals)
      for (auto &[globalName, globalAddr] : globalAddresses) {
        auto blockIt = globalMemoryBlocks.find(globalName);
        if (blockIt != globalMemoryBlocks.end()) {
          MemoryBlock &gBlock = blockIt->second;
          if (strPtr >= globalAddr &&
              strPtr < globalAddr + gBlock.size) {
            uint64_t off = strPtr - globalAddr;
            size_t avail = std::min(
                static_cast<size_t>(strLen),
                gBlock.data.size() - static_cast<size_t>(off));
            if (avail > 0 && gBlock.initialized)
              return std::string(
                  reinterpret_cast<const char *>(gBlock.data.data() + off),
                  avail);
          }
        }
      }

      // Try addressToGlobal reverse lookup
      auto globalIt = addressToGlobal.find(strPtr);
      if (globalIt != addressToGlobal.end()) {
        auto blockIt = globalMemoryBlocks.find(globalIt->second);
        if (blockIt != globalMemoryBlocks.end()) {
          const MemoryBlock &gBlock = blockIt->second;
          size_t avail = std::min(static_cast<size_t>(strLen),
                                  gBlock.data.size());
          if (avail > 0 && gBlock.initialized)
            return std::string(
                reinterpret_cast<const char *>(gBlock.data.data()), avail);
        }
      }

      // Try memory block direct lookup for heap-allocated strings
      uint64_t strOffset = 0;
      auto *strBlock =
          findMemoryBlockByAddress(strPtr, procId, &strOffset);
      if (strBlock && strBlock->initialized &&
          strOffset + strLen <= strBlock->data.size()) {
        return std::string(
            reinterpret_cast<const char *>(strBlock->data.data() + strOffset),
            strLen);
      }

      return "";
    };

    // ---- __moore_fopen ----
    if (calleeName == "__moore_fopen") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        std::string filename =
            extractMooreStringFromPtr(callOp.getOperand(0));
        std::string mode =
            extractMooreStringFromPtr(callOp.getOperand(1));
        int32_t fd = 0;
        if (!filename.empty()) {
          if (mode.empty())
            mode = "r";
          // Build MooreString structs for the runtime call
          MooreString fnStr = {const_cast<char *>(filename.c_str()),
                               static_cast<int64_t>(filename.size())};
          MooreString modeStr = {const_cast<char *>(mode.c_str()),
                                 static_cast<int64_t>(mode.size())};
          fd = __moore_fopen(&fnStr, &modeStr);
        }
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(32, static_cast<uint32_t>(fd))));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_fopen(\"" << filename << "\", \""
                   << mode << "\") = " << fd << "\n");
      }
      return success();
    }

    // ---- __moore_fwrite ----
    if (calleeName == "__moore_fwrite") {
      if (callOp.getNumOperands() >= 2) {
        int32_t fd = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        std::string message =
            extractMooreStringFromPtr(callOp.getOperand(1));
        if (fd != 0 && !message.empty()) {
          MooreString msgStr = {const_cast<char *>(message.c_str()),
                                static_cast<int64_t>(message.size())};
          __moore_fwrite(fd, &msgStr);
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_fwrite(fd=" << fd << ", \""
                   << message << "\")\n");
      }
      return success();
    }

    // ---- __moore_fclose ----
    if (calleeName == "__moore_fclose") {
      if (callOp.getNumOperands() >= 1) {
        int32_t fd = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        if (fd != 0)
          __moore_fclose(fd);
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_fclose(fd=" << fd << ")\n");
      }
      return success();
    }

    //===------------------------------------------------------------------===//
    // String method interceptors
    //===------------------------------------------------------------------===//

    // Helper lambda to read a MooreString from a struct pointer in interpreter
    // memory. Returns the string content as a std::string.
    // This is defined once and reused by all string interceptors below.
    auto readStringFromPtr = [&](Value operand) -> std::string {
      InterpretedValue ptrArg = getValue(procId, operand);
      if (ptrArg.isX())
        return "";

      uint64_t structAddr = ptrArg.getUInt64();
      uint64_t structOffset = 0;
      auto *block = findMemoryBlockByAddress(structAddr, procId, &structOffset);
      if (!block || !block->initialized || structOffset + 16 > block->data.size())
        return "";

      // Read ptr (first 8 bytes, little-endian) and len (next 8 bytes)
      uint64_t strPtr = 0;
      int64_t strLen = 0;
      for (int i = 0; i < 8; ++i) {
        strPtr |= static_cast<uint64_t>(block->data[structOffset + i]) << (i * 8);
        strLen |= static_cast<int64_t>(block->data[structOffset + 8 + i]) << (i * 8);
      }

      if (strPtr == 0 || strLen <= 0)
        return "";

      // Look up in dynamicStrings registry first
      auto dynIt = dynamicStrings.find(static_cast<int64_t>(strPtr));
      if (dynIt != dynamicStrings.end() && dynIt->second.first &&
          dynIt->second.second > 0) {
        return std::string(dynIt->second.first,
                           std::min(static_cast<size_t>(strLen),
                                    static_cast<size_t>(dynIt->second.second)));
      }

      // Try global memory lookup (for string literals)
      for (auto &[globalName, globalAddr] : globalAddresses) {
        auto blockIt = globalMemoryBlocks.find(globalName);
        if (blockIt != globalMemoryBlocks.end()) {
          MemoryBlock &gBlock = blockIt->second;
          if (strPtr >= globalAddr && strPtr < globalAddr + gBlock.size) {
            uint64_t off = strPtr - globalAddr;
            size_t avail = std::min(static_cast<size_t>(strLen),
                                    gBlock.data.size() - static_cast<size_t>(off));
            if (avail > 0 && gBlock.initialized)
              return std::string(reinterpret_cast<const char *>(
                                     gBlock.data.data() + off),
                                 avail);
          }
        }
      }

      // Try addressToGlobal reverse lookup
      auto globalIt = addressToGlobal.find(strPtr);
      if (globalIt != addressToGlobal.end()) {
        auto blockIt = globalMemoryBlocks.find(globalIt->second);
        if (blockIt != globalMemoryBlocks.end()) {
          const MemoryBlock &gBlock = blockIt->second;
          size_t avail = std::min(static_cast<size_t>(strLen),
                                  gBlock.data.size());
          if (avail > 0 && gBlock.initialized)
            return std::string(reinterpret_cast<const char *>(
                                   gBlock.data.data()),
                               avail);
        }
      }

      // Try finding the data in alloca/malloc blocks directly
      uint64_t strDataOffset = 0;
      auto *strDataBlock = findMemoryBlockByAddress(strPtr, procId, &strDataOffset);
      if (strDataBlock && strDataBlock->initialized &&
          strDataOffset + strLen <= strDataBlock->data.size()) {
        return std::string(
            reinterpret_cast<const char *>(strDataBlock->data.data() + strDataOffset),
            strLen);
      }

      return "";
    };

    // Helper to store a string result and return as packed 128-bit struct
    auto storeStringResult = [&](const std::string &str) {
      interpreterStrings.push_back(str);
      const std::string &stored = interpreterStrings.back();
      int64_t ptrVal = reinterpret_cast<int64_t>(stored.data());
      int64_t lenVal = static_cast<int64_t>(stored.size());
      dynamicStrings[ptrVal] = {stored.data(), lenVal};
      APInt packedResult(128, 0);
      packedResult.insertBits(APInt(64, static_cast<uint64_t>(ptrVal)), 0);
      packedResult.insertBits(APInt(64, static_cast<uint64_t>(lenVal)), 64);
      return InterpretedValue(packedResult);
    };

    // ---- __moore_string_toupper ----
    // Signature: (str_ptr: ptr) -> struct{ptr, i64}
    if (calleeName == "__moore_string_toupper") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        for (auto &c : str)
          c = std::toupper(static_cast<unsigned char>(c));
        setValue(procId, callOp.getResult(), storeStringResult(str));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_toupper() = \""
                                << str << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_tolower ----
    // Signature: (str_ptr: ptr) -> struct{ptr, i64}
    if (calleeName == "__moore_string_tolower") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        for (auto &c : str)
          c = std::tolower(static_cast<unsigned char>(c));
        setValue(procId, callOp.getResult(), storeStringResult(str));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_tolower() = \""
                                << str << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_getc ----
    // Signature: (str_ptr: ptr, index: i32) -> i8
    if (calleeName == "__moore_string_getc") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t index = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int8_t result = 0;
        if (index >= 0 && index < static_cast<int32_t>(str.size()))
          result = static_cast<int8_t>(str[index]);
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(result))));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_getc(\"" << str
                                << "\", " << index << ") = " << static_cast<int>(result)
                                << "\n");
      }
      return success();
    }

    // ---- __moore_string_putc ----
    // Signature: (str_ptr: ptr, index: i32, ch: i8) -> struct{ptr, i64}
    if (calleeName == "__moore_string_putc") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t index = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int8_t ch = static_cast<int8_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());
        // IEEE 1800-2017: putc is a no-op for out of bounds
        if (index >= 0 && index < static_cast<int32_t>(str.size()))
          str[index] = static_cast<char>(ch);
        setValue(procId, callOp.getResult(), storeStringResult(str));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_putc() = \""
                                << str << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_substr ----
    // Signature: (str_ptr: ptr, start: i32, len: i32) -> struct{ptr, i64}
    if (calleeName == "__moore_string_substr") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t start = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int32_t len = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());
        std::string result;
        if (start >= 0 && start < static_cast<int32_t>(str.size()) && len > 0) {
          size_t actualLen = std::min(static_cast<size_t>(len),
                                      str.size() - static_cast<size_t>(start));
          result = str.substr(start, actualLen);
        }
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_substr(\"" << str
                                << "\", " << start << ", " << len << ") = \""
                                << result << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_compare ----
    // Signature: (lhs_ptr: ptr, rhs_ptr: ptr) -> i32
    if (calleeName == "__moore_string_compare") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        std::string lhs = readStringFromPtr(callOp.getOperand(0));
        std::string rhs = readStringFromPtr(callOp.getOperand(1));
        int32_t result = 0;
        if (lhs < rhs) result = -1;
        else if (lhs > rhs) result = 1;
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_compare(\"" << lhs
                                << "\", \"" << rhs << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_icompare ----
    // Signature: (lhs_ptr: ptr, rhs_ptr: ptr) -> i32
    if (calleeName == "__moore_string_icompare") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        std::string lhs = readStringFromPtr(callOp.getOperand(0));
        std::string rhs = readStringFromPtr(callOp.getOperand(1));
        // Case-insensitive comparison
        std::string lhsLower = lhs, rhsLower = rhs;
        for (auto &c : lhsLower) c = std::tolower(static_cast<unsigned char>(c));
        for (auto &c : rhsLower) c = std::tolower(static_cast<unsigned char>(c));
        int32_t result = 0;
        if (lhsLower < rhsLower) result = -1;
        else if (lhsLower > rhsLower) result = 1;
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_icompare(\"" << lhs
                                << "\", \"" << rhs << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_replicate ----
    // Signature: (str_ptr: ptr, count: i32) -> struct{ptr, i64}
    if (calleeName == "__moore_string_replicate") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t count = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        std::string result;
        if (count > 0 && !str.empty()) {
          result.reserve(str.size() * count);
          for (int32_t i = 0; i < count; ++i)
            result += str;
        }
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_replicate(\"" << str
                                << "\", " << count << ") = \"" << result << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_to_int ----
    // Signature: (str_ptr: ptr) -> i64
    if (calleeName == "__moore_string_to_int") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int64_t result = 0;
        if (!str.empty()) {
          { char *endp = nullptr; result = std::strtoll(str.c_str(), &endp, 10); }
        }
        setValue(procId, callOp.getResult(),
                 InterpretedValue(static_cast<uint64_t>(result), 64));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_to_int(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_atoi ----
    // Signature: (str_ptr: ptr) -> i32
    if (calleeName == "__moore_string_atoi") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t result = 0;
        if (!str.empty()) {
          { char *endp = nullptr; result = static_cast<int32_t>(std::strtol(str.c_str(), &endp, 10)); }
        }
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_atoi(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_atohex ----
    // Signature: (str_ptr: ptr) -> i32
    if (calleeName == "__moore_string_atohex") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t result = 0;
        if (!str.empty()) {
          { char *endp = nullptr; result = static_cast<int32_t>(std::strtoul(str.c_str(), &endp, 16)); }
        }
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_atohex(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_atooct ----
    // Signature: (str_ptr: ptr) -> i32
    if (calleeName == "__moore_string_atooct") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t result = 0;
        if (!str.empty()) {
          { char *endp = nullptr; result = static_cast<int32_t>(std::strtoul(str.c_str(), &endp, 8)); }
        }
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_atooct(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_atobin ----
    // Signature: (str_ptr: ptr) -> i32
    if (calleeName == "__moore_string_atobin") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        int32_t result = 0;
        if (!str.empty()) {
          { char *endp = nullptr; result = static_cast<int32_t>(std::strtoul(str.c_str(), &endp, 2)); }
        }
        unsigned resultWidth = getTypeWidth(callOp.getResult().getType());
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(resultWidth, static_cast<uint64_t>(static_cast<uint32_t>(result)), false)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_atobin(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_atoreal ----
    // Signature: (str_ptr: ptr) -> f64
    if (calleeName == "__moore_string_atoreal") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        std::string str = readStringFromPtr(callOp.getOperand(0));
        double result = 0.0;
        if (!str.empty()) {
          { char *endp = nullptr; result = std::strtod(str.c_str(), &endp); }
        }
        // Store as 64-bit IEEE 754 double
        uint64_t bits;
        std::memcpy(&bits, &result, sizeof(bits));
        setValue(procId, callOp.getResult(),
                 InterpretedValue(APInt(64, bits)));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_atoreal(\"" << str
                                << "\") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_string_hextoa ----
    // Signature: (value: i64) -> struct{ptr, i64}
    if (calleeName == "__moore_string_hextoa") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue arg = getValue(procId, callOp.getOperand(0));
        uint64_t value = arg.isX() ? 0 : arg.getUInt64();
        char buffer[32];
        int len = std::snprintf(buffer, sizeof(buffer), "%lx",
                                static_cast<unsigned long>(value));
        std::string result(buffer, len > 0 ? len : 0);
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_hextoa("
                                << value << ") = \"" << result << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_octtoa ----
    // Signature: (value: i64) -> struct{ptr, i64}
    if (calleeName == "__moore_string_octtoa") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue arg = getValue(procId, callOp.getOperand(0));
        uint64_t value = arg.isX() ? 0 : arg.getUInt64();
        char buffer[32];
        int len = std::snprintf(buffer, sizeof(buffer), "%lo",
                                static_cast<unsigned long>(value));
        std::string result(buffer, len > 0 ? len : 0);
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_octtoa("
                                << value << ") = \"" << result << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_bintoa ----
    // Signature: (value: i64) -> struct{ptr, i64}
    if (calleeName == "__moore_string_bintoa") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue arg = getValue(procId, callOp.getOperand(0));
        uint64_t value = arg.isX() ? 0 : arg.getUInt64();
        std::string result;
        if (value == 0) {
          result = "0";
        } else {
          // Find highest set bit
          int highBit = 63;
          while (highBit > 0 && !((value >> highBit) & 1))
            --highBit;
          for (int i = highBit; i >= 0; --i)
            result += ((value >> i) & 1) ? '1' : '0';
        }
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_bintoa("
                                << value << ") = \"" << result << "\"\n");
      }
      return success();
    }

    // ---- __moore_string_realtoa ----
    // Signature: (value: f64) -> struct{ptr, i64}
    if (calleeName == "__moore_string_realtoa") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue arg = getValue(procId, callOp.getOperand(0));
        double value = 0.0;
        if (!arg.isX()) {
          uint64_t bits = arg.getUInt64();
          std::memcpy(&value, &bits, sizeof(value));
        }
        char buffer[64];
        int len = std::snprintf(buffer, sizeof(buffer), "%g", value);
        std::string result(buffer, len > 0 ? len : 0);
        setValue(procId, callOp.getResult(), storeStringResult(result));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_string_realtoa("
                                << value << ") = \"" << result << "\"\n");
      }
      return success();
    }

    //===------------------------------------------------------------------===//
    // Queue method interceptors
    //===------------------------------------------------------------------===//

    // ---- __moore_queue_delete_index ----
    // Signature: (queue_ptr, index: i32, element_size: i64) -> void
    if (calleeName == "__moore_queue_delete_index") {
      if (callOp.getNumOperands() >= 3) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int32_t index = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized &&
              queueOffset + 16 <= queueBlock->data.size()) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            // Bounds check
            if (index >= 0 && index < queueLen && dataPtr != 0) {
              if (queueLen == 1) {
                // Last element - set data to 0 and len to 0
                for (int i = 0; i < 8; ++i)
                  queueBlock->data[queueOffset + i] = 0;
                for (int i = 0; i < 8; ++i)
                  queueBlock->data[queueOffset + 8 + i] = 0;
              } else {
                // Allocate new storage
                int64_t newLen = queueLen - 1;
                uint64_t newDataAddr = globalNextAddress;
                globalNextAddress += newLen * elemSize;
                MemoryBlock newBlock(newLen * elemSize, 64);
                newBlock.initialized = true;

                auto *oldBlock = findMemoryBlockByAddress(dataPtr, procId);
                if (oldBlock && oldBlock->initialized) {
                  // Copy elements before deleted index
                  if (index > 0)
                    std::memcpy(newBlock.data.data(), oldBlock->data.data(),
                                index * elemSize);
                  // Copy elements after deleted index
                  if (index < queueLen - 1)
                    std::memcpy(newBlock.data.data() + index * elemSize,
                                oldBlock->data.data() + (index + 1) * elemSize,
                                (queueLen - index - 1) * elemSize);
                }

                mallocBlocks[newDataAddr] = std::move(newBlock);

                // Update queue struct
                for (int i = 0; i < 8; ++i)
                  queueBlock->data[queueOffset + i] =
                      static_cast<uint8_t>((newDataAddr >> (i * 8)) & 0xFF);
                for (int i = 0; i < 8; ++i)
                  queueBlock->data[queueOffset + 8 + i] =
                      static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);
              }
              checkMemoryEventWaiters();
            }

            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_delete_index("
                                    << "0x" << llvm::format_hex(queueAddr, 16)
                                    << ", " << index << ", " << elemSize << ")\n");
          }
        }
      }
      return success();
    }

    // ---- __moore_queue_insert ----
    // Signature: (queue_ptr, index: i32, element_ptr, element_size: i64) -> void
    if (calleeName == "__moore_queue_insert") {
      if (callOp.getNumOperands() >= 4) {
        uint64_t queueAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        int32_t index = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t elemAddr = getValue(procId, callOp.getOperand(2)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(3)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock = findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized &&
              queueOffset + 16 <= queueBlock->data.size()) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(queueBlock->data[queueOffset + i]) << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(queueBlock->data[queueOffset + 8 + i]) << (i * 8);

            // Clamp index
            if (index < 0) index = 0;
            if (index > queueLen) index = static_cast<int32_t>(queueLen);

            // Allocate new storage
            int64_t newLen = queueLen + 1;
            uint64_t newDataAddr = globalNextAddress;
            globalNextAddress += newLen * elemSize;
            MemoryBlock newBlock(newLen * elemSize, 64);
            newBlock.initialized = true;

            // Copy elements before insertion point
            if (index > 0 && dataPtr != 0) {
              auto *oldBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (oldBlock && oldBlock->initialized) {
                size_t copySize = std::min(static_cast<size_t>(index * elemSize),
                                           oldBlock->data.size());
                std::memcpy(newBlock.data.data(), oldBlock->data.data(), copySize);
              }
            }

            // Copy new element at insertion index
            uint64_t elemOffset = 0;
            auto *elemBlock = findMemoryBlockByAddress(elemAddr, procId, &elemOffset);
            if (elemBlock && elemBlock->initialized) {
              size_t avail = (elemOffset < elemBlock->data.size())
                  ? elemBlock->data.size() - elemOffset : 0;
              size_t copySize = std::min(static_cast<size_t>(elemSize), avail);
              if (copySize > 0)
                std::memcpy(newBlock.data.data() + index * elemSize,
                            elemBlock->data.data() + elemOffset, copySize);
            }

            // Copy elements after insertion point
            if (index < queueLen && dataPtr != 0) {
              auto *oldBlock = findMemoryBlockByAddress(dataPtr, procId);
              if (oldBlock && oldBlock->initialized) {
                size_t srcOffset = index * elemSize;
                size_t dstOffset = (index + 1) * elemSize;
                size_t copySize = (queueLen - index) * elemSize;
                if (srcOffset + copySize <= oldBlock->data.size() &&
                    dstOffset + copySize <= newBlock.data.size())
                  std::memcpy(newBlock.data.data() + dstOffset,
                              oldBlock->data.data() + srcOffset, copySize);
              }
            }

            mallocBlocks[newDataAddr] = std::move(newBlock);

            // Update queue struct
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + i] =
                  static_cast<uint8_t>((newDataAddr >> (i * 8)) & 0xFF);
            for (int i = 0; i < 8; ++i)
              queueBlock->data[queueOffset + 8 + i] =
                  static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);

            checkMemoryEventWaiters();

            LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_queue_insert("
                                    << "0x" << llvm::format_hex(queueAddr, 16)
                                    << ", " << index << ", elemSize=" << elemSize
                                    << ", newLen=" << newLen << ")\n");
          }
        }
      }
      return success();
    }

    //===------------------------------------------------------------------===//
    // Associative array method interceptors
    //===------------------------------------------------------------------===//

    // ---- __moore_assoc_delete ----
    // Signature: (array: ptr) -> void
    if (calleeName == "__moore_assoc_delete") {
      if (callOp.getNumOperands() >= 1) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL;
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr != 0 &&
            (validAssocArrayAddresses.contains(arrayAddr) || isValidNativeAddr)) {
          void *arrayPtr = reinterpret_cast<void *>(arrayAddr);
          __moore_assoc_delete(arrayPtr);
        }
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_delete(0x"
                                << llvm::format_hex(arrayAddr, 16) << ")\n");
      }
      return success();
    }

    // ---- __moore_assoc_delete_key ----
    // Signature: (array: ptr, key: ptr) -> void
    if (calleeName == "__moore_assoc_delete_key") {
      if (callOp.getNumOperands() >= 2) {
        uint64_t arrayAddr = getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyAddr = getValue(procId, callOp.getOperand(1)).getUInt64();
        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL;
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr != 0 &&
            (validAssocArrayAddresses.contains(arrayAddr) || isValidNativeAddr)) {
          void *arrayPtr = reinterpret_cast<void *>(arrayAddr);
          // Read key from interpreter memory
          uint64_t keyOffset = 0;
          auto *keyBlock = findMemoryBlockByAddress(keyAddr, procId, &keyOffset);
          uint8_t keyBuffer[16] = {0};
          void *keyPtr = keyBuffer;
          MooreString keyString = {nullptr, 0};
          std::string keyStorage;

          if (keyBlock && keyBlock->initialized) {
            size_t maxCopy = std::min<size_t>(16, keyBlock->data.size() - keyOffset);
            std::memcpy(keyBuffer, keyBlock->data.data() + keyOffset, maxCopy);

            auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
            if (header->type == AssocArrayType_StringKey) {
              uint64_t strPtrVal = 0;
              int64_t strLen = 0;
              for (int i = 0; i < 8; ++i) {
                strPtrVal |= static_cast<uint64_t>(keyBuffer[i]) << (i * 8);
                strLen |= static_cast<int64_t>(keyBuffer[8 + i]) << (i * 8);
              }
              if (tryReadStringKey(procId, strPtrVal, strLen, keyStorage)) {
                keyString.data = keyStorage.data();
                keyString.len = keyStorage.size();
                keyPtr = &keyString;
              }
            }
          }

          __moore_assoc_delete_key(arrayPtr, keyPtr);
        }
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_assoc_delete_key(0x"
                                << llvm::format_hex(arrayAddr, 16) << ")\n");
      }
      return success();
    }

    // ---- __moore_dyn_array_new ----
    // Signature: (size: i32) -> struct<(ptr, i64)>
    if (calleeName == "__moore_dyn_array_new") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        int32_t size = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        MooreQueue result = __moore_dyn_array_new(size);
        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_dyn_array_new("
                                << size << ") = {0x"
                                << llvm::format_hex(ptrVal, 16) << ", "
                                << lenVal << "}\n");
      }
      return success();
    }

    // ---- __moore_dyn_array_new_copy ----
    // Signature: (size: i32, init: ptr) -> struct<(ptr, i64)>
    if (calleeName == "__moore_dyn_array_new_copy") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        int32_t size = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(0)).getUInt64());
        uint64_t initAddr =
            getValue(procId, callOp.getOperand(1)).getUInt64();
        MooreQueue result = __moore_dyn_array_new(size);
        if (result.data && initAddr != 0 && size > 0) {
          uint64_t initOffset = 0;
          auto *initBlock =
              findMemoryBlockByAddress(initAddr, procId, &initOffset);
          if (initBlock && initBlock->initialized) {
            size_t avail = initBlock->data.size() - initOffset;
            size_t copySize = std::min<size_t>(size, avail);
            std::memcpy(result.data, initBlock->data.data() + initOffset,
                        copySize);
          } else if (initAddr >= 0x10000000000ULL) {
            std::memcpy(result.data,
                        reinterpret_cast<void *>(initAddr), size);
          }
        }
        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_dyn_array_new_copy(" << size
                   << ", 0x" << llvm::format_hex(initAddr, 16) << ")\n");
      }
      return success();
    }

    // ---- __moore_is_rand_enabled ----
    // Signature: (classPtr: ptr, propertyName: ptr) -> i32
    if (calleeName == "__moore_is_rand_enabled") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        // Default: all random variables are enabled (return 1).
        // The runtime tracks rand_mode state keyed by (classPtr, propertyName).
        // Since we can't easily bridge interpreter memory for the C-string
        // property name, we stub to the default enabled state.
        setValue(procId, callOp.getResult(), InterpretedValue(1ULL, 32));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_is_rand_enabled() = 1\n");
      }
      return success();
    }

    // ---- __moore_rand_mode_get ----
    // Signature: (classPtr: ptr, propertyName: ptr) -> i32
    if (calleeName == "__moore_rand_mode_get") {
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(1ULL, 32));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_rand_mode_get() = 1\n");
      }
      return success();
    }

    // ---- __moore_rand_mode_set ----
    // Signature: (classPtr: ptr, propertyName: ptr, mode: i32) -> i32
    if (calleeName == "__moore_rand_mode_set") {
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(1ULL, 32));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_rand_mode_set() = 1\n");
      }
      return success();
    }

    // ---- __moore_randc_next ----
    // Signature: (fieldPtr: ptr, bitWidth: i64) -> i64
    if (calleeName == "__moore_randc_next") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        int64_t bitWidth = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        uint64_t maxVal = (bitWidth >= 64) ? UINT64_MAX
                                            : ((1ULL << bitWidth) - 1);
        uint64_t result = static_cast<uint64_t>(std::rand()) & maxVal;
        setValue(procId, callOp.getResult(),
                 InterpretedValue(result, 64));
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_randc_next(bw="
                                << bitWidth << ") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_queue_pop_back_ptr ----
    // Signature: (queue_ptr: ptr, result_ptr: ptr, elem_size: i64) -> void
    if (calleeName == "__moore_queue_pop_back_ptr") {
      if (callOp.getNumOperands() >= 3) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t resultAddr =
            getValue(procId, callOp.getOperand(1)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock =
              findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(
                             queueBlock->data[queueOffset + i])
                         << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(
                              queueBlock->data[queueOffset + 8 + i])
                          << (i * 8);

            if (queueLen > 0 && dataPtr != 0) {
              // Look up queue data via interpreter memory model
              uint64_t dataOffset = 0;
              auto *dataBlock =
                  findMemoryBlockByAddress(dataPtr, procId, &dataOffset);
              if (dataBlock && dataBlock->initialized) {
                // Read last element from data array
                size_t lastElemOff =
                    dataOffset + (queueLen - 1) * elemSize;

                // Write to result_ptr in interpreter memory
                uint64_t resultOffset = 0;
                auto *resultBlock = findMemoryBlockByAddress(
                    resultAddr, procId, &resultOffset);
                if (resultBlock &&
                    lastElemOff + elemSize <= dataBlock->data.size()) {
                  size_t avail =
                      resultBlock->data.size() - resultOffset;
                  size_t copySize =
                      std::min<size_t>(elemSize, avail);
                  std::memcpy(
                      resultBlock->data.data() + resultOffset,
                      dataBlock->data.data() + lastElemOff, copySize);
                  resultBlock->initialized = true;
                }
              }

              // Decrement length
              int64_t newLen = queueLen - 1;
              for (int i = 0; i < 8; ++i)
                queueBlock->data[queueOffset + 8 + i] =
                    static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);
            }
          }
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_queue_pop_back_ptr(0x"
                   << llvm::format_hex(queueAddr, 16) << ")\n");
        checkMemoryEventWaiters();
      }
      return success();
    }

    // ---- __moore_queue_pop_front_ptr ----
    // Signature: (queue_ptr: ptr, result_ptr: ptr, elem_size: i64) -> void
    if (calleeName == "__moore_queue_pop_front_ptr") {
      if (callOp.getNumOperands() >= 3) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t resultAddr =
            getValue(procId, callOp.getOperand(1)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());

        if (queueAddr != 0 && elemSize > 0) {
          uint64_t queueOffset = 0;
          auto *queueBlock =
              findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (queueBlock && queueBlock->initialized) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(
                             queueBlock->data[queueOffset + i])
                         << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(
                              queueBlock->data[queueOffset + 8 + i])
                          << (i * 8);

            if (queueLen > 0 && dataPtr != 0) {
              // Look up queue data via interpreter memory model
              uint64_t dataOffset = 0;
              auto *dataBlock =
                  findMemoryBlockByAddress(dataPtr, procId, &dataOffset);
              if (dataBlock && dataBlock->initialized) {
                // Write first element to result_ptr
                uint64_t resultOffset = 0;
                auto *resultBlock = findMemoryBlockByAddress(
                    resultAddr, procId, &resultOffset);
                if (resultBlock &&
                    dataOffset + elemSize <= dataBlock->data.size()) {
                  size_t avail =
                      resultBlock->data.size() - resultOffset;
                  size_t copySize =
                      std::min<size_t>(elemSize, avail);
                  std::memcpy(
                      resultBlock->data.data() + resultOffset,
                      dataBlock->data.data() + dataOffset, copySize);
                  resultBlock->initialized = true;
                }

                // Shift remaining elements within the data block
                if (queueLen > 1) {
                  size_t moveSize = (queueLen - 1) * elemSize;
                  if (dataOffset + elemSize + moveSize <=
                      dataBlock->data.size()) {
                    std::memmove(
                        dataBlock->data.data() + dataOffset,
                        dataBlock->data.data() + dataOffset + elemSize,
                        moveSize);
                  }
                }
              }

              // Decrement length
              int64_t newLen = queueLen - 1;
              for (int i = 0; i < 8; ++i)
                queueBlock->data[queueOffset + 8 + i] =
                    static_cast<uint8_t>((newLen >> (i * 8)) & 0xFF);
            }
          }
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_queue_pop_front_ptr(0x"
                   << llvm::format_hex(queueAddr, 16) << ")\n");
        checkMemoryEventWaiters();
      }
      return success();
    }

    // ---- __moore_queue_concat ----
    // Signature: (queues: ptr, count: i64, elem_size: i64)
    //            -> struct<(ptr, i64)>
    if (calleeName == "__moore_queue_concat") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        uint64_t queuesAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t count = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());

        // Compute total length
        int64_t totalLen = 0;
        std::vector<std::pair<void *, int64_t>> srcs;
        if (queuesAddr != 0 && count > 0 && elemSize > 0) {
          uint64_t qOff = 0;
          auto *qBlock =
              findMemoryBlockByAddress(queuesAddr, procId, &qOff);
          if (qBlock && qBlock->initialized) {
            for (int64_t i = 0; i < count; ++i) {
              size_t base = qOff + i * 16;
              if (base + 16 > qBlock->data.size())
                break;
              uint64_t dp = 0;
              int64_t ln = 0;
              for (int b = 0; b < 8; ++b)
                dp |= static_cast<uint64_t>(qBlock->data[base + b])
                      << (b * 8);
              for (int b = 0; b < 8; ++b)
                ln |= static_cast<int64_t>(
                           qBlock->data[base + 8 + b])
                      << (b * 8);
              // Only use dp as native pointer if it's in native memory range
              if (dp >= 0x10000000000ULL)
                srcs.push_back({reinterpret_cast<void *>(dp), ln});
              else
                srcs.push_back({nullptr, 0});
              totalLen += ln;
            }
          }
        }

        MooreQueue result = __moore_dyn_array_new(
            static_cast<int32_t>(totalLen * elemSize));
        if (result.data) {
          int64_t offset = 0;
          for (auto &[srcPtr, srcLen] : srcs) {
            if (srcPtr && srcLen > 0) {
              std::memcpy(static_cast<char *>(result.data) +
                              offset * elemSize,
                          srcPtr, srcLen * elemSize);
              offset += srcLen;
            }
          }
          result.len = totalLen;
        }

        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_queue_concat(count="
                   << count << ", totalLen=" << totalLen << ")\n");
      }
      return success();
    }

    // ---- __moore_assoc_last ----
    // Signature: (array: ptr, key_out: ptr) -> i1
    if (calleeName == "__moore_assoc_last") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyOutAddr =
            getValue(procId, callOp.getOperand(1)).getUInt64();

        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL;
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 ||
            (!validAssocArrayAddresses.contains(arrayAddr) &&
             !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(),
                   InterpretedValue(0ULL, 1));
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);
        uint64_t keyOutOffset = 0;
        auto *keyOutBlock =
            findMemoryBlockByAddress(keyOutAddr, procId, &keyOutOffset);

        auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
        bool isStringKey = (header->type == AssocArrayType_StringKey);

        bool result = false;
        if (isStringKey) {
          MooreString keyOut = {nullptr, 0};
          result = __moore_assoc_last(arrayPtr, &keyOut);
          if (result && keyOutBlock &&
              keyOutOffset + 16 <= keyOutBlock->data.size()) {
            uint64_t pv = reinterpret_cast<uint64_t>(keyOut.data);
            int64_t lv = keyOut.len;
            for (int i = 0; i < 8; ++i) {
              keyOutBlock->data[keyOutOffset + i] =
                  static_cast<uint8_t>((pv >> (i * 8)) & 0xFF);
              keyOutBlock->data[keyOutOffset + 8 + i] =
                  static_cast<uint8_t>((lv >> (i * 8)) & 0xFF);
            }
            keyOutBlock->initialized = true;
          }
        } else {
          uint8_t keyBuffer[8] = {0};
          result = __moore_assoc_last(arrayPtr, keyBuffer);
          if (result && keyOutBlock) {
            size_t avail =
                keyOutBlock->data.size() - keyOutOffset;
            size_t copySize = std::min<size_t>(8, avail);
            std::memcpy(keyOutBlock->data.data() + keyOutOffset,
                        keyBuffer, copySize);
            keyOutBlock->initialized = true;
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(result ? 1ULL : 0ULL, 1));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_assoc_last(0x"
                   << llvm::format_hex(arrayAddr, 16) << ") = "
                   << result << "\n");
      }
      return success();
    }

    // ---- __moore_assoc_prev ----
    // Signature: (array: ptr, key_ref: ptr) -> i1
    if (calleeName == "__moore_assoc_prev") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        uint64_t keyRefAddr =
            getValue(procId, callOp.getOperand(1)).getUInt64();

        constexpr uint64_t kNativeHeapThreshold = 0x10000000000ULL;
        bool isValidNativeAddr = arrayAddr >= kNativeHeapThreshold;
        if (arrayAddr == 0 ||
            (!validAssocArrayAddresses.contains(arrayAddr) &&
             !isValidNativeAddr)) {
          setValue(procId, callOp.getResult(),
                   InterpretedValue(0ULL, 1));
          return success();
        }

        void *arrayPtr = reinterpret_cast<void *>(arrayAddr);
        uint64_t keyRefOffset = 0;
        auto *keyRefBlock =
            findMemoryBlockByAddress(keyRefAddr, procId, &keyRefOffset);

        auto *header = static_cast<AssocArrayHeader *>(arrayPtr);
        bool isStringKey = (header->type == AssocArrayType_StringKey);

        bool result = false;
        if (isStringKey) {
          MooreString keyRef = {nullptr, 0};
          if (keyRefBlock && keyRefBlock->initialized &&
              keyRefOffset + 16 <= keyRefBlock->data.size()) {
            uint64_t spv = 0;
            int64_t sl = 0;
            for (int i = 0; i < 8; ++i) {
              spv |= static_cast<uint64_t>(
                         keyRefBlock->data[keyRefOffset + i])
                     << (i * 8);
              sl |= static_cast<int64_t>(
                        keyRefBlock->data[keyRefOffset + 8 + i])
                    << (i * 8);
            }
            keyRef.data = reinterpret_cast<char *>(spv);
            keyRef.len = sl;
          }

          result = __moore_assoc_prev(arrayPtr, &keyRef);

          if (result && keyRefBlock &&
              keyRefOffset + 16 <= keyRefBlock->data.size()) {
            uint64_t pv = reinterpret_cast<uint64_t>(keyRef.data);
            int64_t lv = keyRef.len;
            for (int i = 0; i < 8; ++i) {
              keyRefBlock->data[keyRefOffset + i] =
                  static_cast<uint8_t>((pv >> (i * 8)) & 0xFF);
              keyRefBlock->data[keyRefOffset + 8 + i] =
                  static_cast<uint8_t>((lv >> (i * 8)) & 0xFF);
            }
          }
        } else {
          uint8_t keyBuffer[8] = {0};
          if (keyRefBlock && keyRefBlock->initialized) {
            size_t avail =
                keyRefBlock->data.size() - keyRefOffset;
            size_t readSize = std::min<size_t>(8, avail);
            std::memcpy(keyBuffer,
                        keyRefBlock->data.data() + keyRefOffset,
                        readSize);
          }

          result = __moore_assoc_prev(arrayPtr, keyBuffer);

          if (result && keyRefBlock) {
            size_t avail =
                keyRefBlock->data.size() - keyRefOffset;
            size_t copySize = std::min<size_t>(8, avail);
            std::memcpy(keyRefBlock->data.data() + keyRefOffset,
                        keyBuffer, copySize);
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(result ? 1ULL : 0ULL, 1));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_assoc_prev(0x"
                   << llvm::format_hex(arrayAddr, 16) << ") = "
                   << result << "\n");
      }
      return success();
    }

    // ---- __moore_stream_concat_bits ----
    // Signature: (queue: ptr, elementBitWidth: i32, isRightToLeft: i1) -> i64
    if (calleeName == "__moore_stream_concat_bits") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        int32_t elemBitWidth = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        bool isRightToLeft = getValue(procId, callOp.getOperand(2)).getUInt64() != 0;

        int64_t result = 0;
        if (queueAddr != 0) {
          // Try to read MooreQueue struct from interpreter memory
          uint64_t queueOffset = 0;
          auto *qBlock =
              findMemoryBlockByAddress(queueAddr, procId, &queueOffset);
          if (qBlock && qBlock->initialized &&
              queueOffset + 16 <= qBlock->data.size()) {
            uint64_t dataPtr = 0;
            int64_t queueLen = 0;
            for (int i = 0; i < 8; ++i)
              dataPtr |= static_cast<uint64_t>(
                             qBlock->data[queueOffset + i])
                         << (i * 8);
            for (int i = 0; i < 8; ++i)
              queueLen |= static_cast<int64_t>(
                              qBlock->data[queueOffset + 8 + i])
                          << (i * 8);
            if (dataPtr != 0 && dataPtr >= 0x10000000000ULL) {
              MooreQueue q;
              q.data = reinterpret_cast<void *>(dataPtr);
              q.len = queueLen;
              result = __moore_stream_concat_bits(
                  &q, elemBitWidth, isRightToLeft);
            }
          } else if (queueAddr >= 0x10000000000ULL) {
            result = __moore_stream_concat_bits(
                reinterpret_cast<MooreQueue *>(queueAddr),
                elemBitWidth, isRightToLeft);
          }
        }

        setValue(procId, callOp.getResult(),
                 InterpretedValue(static_cast<uint64_t>(result), 64));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_stream_concat_bits(bw="
                   << elemBitWidth << ") = " << result << "\n");
      }
      return success();
    }

    // ---- __moore_stream_concat_strings ----
    // Signature: (queue: ptr, isRightToLeft: i1) -> struct<(ptr, i64)>
    if (calleeName == "__moore_stream_concat_strings") {
      if (callOp.getNumOperands() >= 2 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        bool isRtl = getValue(procId, callOp.getOperand(1)).getUInt64() != 0;

        MooreString result = {nullptr, 0};
        if (queueAddr != 0) {
          uint64_t qOff = 0;
          auto *qBlock =
              findMemoryBlockByAddress(queueAddr, procId, &qOff);
          if (qBlock && qBlock->initialized &&
              qOff + 16 <= qBlock->data.size()) {
            uint64_t dp = 0;
            int64_t ln = 0;
            for (int i = 0; i < 8; ++i)
              dp |= static_cast<uint64_t>(qBlock->data[qOff + i])
                    << (i * 8);
            for (int i = 0; i < 8; ++i)
              ln |= static_cast<int64_t>(
                        qBlock->data[qOff + 8 + i])
                    << (i * 8);
            if (dp != 0) {
              // Only call native function if dp is a real native pointer,
              // not a synthetic interpreter address. Interpreter addresses
              // are assigned sequentially from a low range and cannot be
              // dereferenced natively.
              if (dp >= 0x10000000000ULL) {
                MooreQueue q;
                q.data = reinterpret_cast<void *>(dp);
                q.len = ln;
                result = __moore_stream_concat_strings(&q, isRtl);
              }
            }
          } else if (queueAddr >= 0x10000000000ULL) {
            result = __moore_stream_concat_strings(
                reinterpret_cast<MooreQueue *>(queueAddr), isRtl);
          }
        }

        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        if (result.data)
          dynamicStrings[ptrVal] = {result.data, result.len};
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_stream_concat_strings()\n");
      }
      return success();
    }

    // ---- __moore_cross_create ----
    // Signature: (cg: ptr, name: ptr, cp_indices: ptr, num: i32) -> i32
    if (calleeName == "__moore_cross_create") {
      if (callOp.getNumResults() >= 1) {
        // Stub: return a dummy cross index
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_cross_create() = 0 (stub)\n");
      }
      return success();
    }

    // ---- __moore_cross_sample ----
    // Signature: (cg: ptr, cp_values: ptr, num_values: i32) -> void
    if (calleeName == "__moore_cross_sample") {
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_cross_sample() (stub)\n");
      return success();
    }

    // ---- __moore_cross_add_named_bin ----
    // Signature: (cg: ptr, cross_index: i32, name: ptr, ...) -> i32
    if (calleeName == "__moore_cross_add_named_bin") {
      if (callOp.getNumResults() >= 1) {
        setValue(procId, callOp.getResult(), InterpretedValue(0ULL, 32));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_cross_add_named_bin() = 0\n");
      }
      return success();
    }

    // ---- __moore_queue_slice ----
    // Signature: (queue: ptr, start: i64, end: i64, elem_size: i64)
    //            -> struct<(ptr, i64)>
    if (calleeName == "__moore_queue_slice") {
      if (callOp.getNumOperands() >= 4 && callOp.getNumResults() >= 1) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t start = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int64_t end = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(3)).getUInt64());

        MooreQueue result = {nullptr, 0};
        if (queueAddr != 0 && elemSize > 0 && end >= start) {
          uint64_t qOff = 0;
          auto *qBlock =
              findMemoryBlockByAddress(queueAddr, procId, &qOff);
          if (qBlock && qBlock->initialized &&
              qOff + 16 <= qBlock->data.size()) {
            uint64_t dp = 0;
            int64_t ln = 0;
            for (int i = 0; i < 8; ++i)
              dp |= static_cast<uint64_t>(qBlock->data[qOff + i])
                    << (i * 8);
            for (int i = 0; i < 8; ++i)
              ln |= static_cast<int64_t>(
                        qBlock->data[qOff + 8 + i])
                    << (i * 8);
            int64_t sliceLen = std::min(end, ln) - start;
            if (sliceLen > 0 && dp != 0 && dp >= 0x10000000000ULL) {
              result = __moore_dyn_array_new(
                  static_cast<int32_t>(sliceLen * elemSize));
              if (result.data) {
                std::memcpy(result.data,
                            static_cast<char *>(
                                reinterpret_cast<void *>(dp)) +
                                start * elemSize,
                            sliceLen * elemSize);
                result.len = sliceLen;
              }
            }
          }
        }

        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_queue_slice()\n");
      }
      return success();
    }

    // ---- __moore_queue_sort ----
    // Signature: (queue: ptr) -> void
    if (calleeName == "__moore_queue_sort") {
      // Stub - sorting would require knowing element size and comparator
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: __moore_queue_sort() (stub)\n");
      return success();
    }

    // ---- __moore_queue_unique ----
    // Signature: (queue: ptr) -> struct<(ptr, i64)>
    if (calleeName == "__moore_queue_unique") {
      if (callOp.getNumResults() >= 1) {
        uint64_t queueAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        MooreQueue result = {nullptr, 0};
        if (queueAddr >= 0x10000000000ULL) {
          result = __moore_queue_unique(
              reinterpret_cast<MooreQueue *>(queueAddr));
        }
        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_queue_unique()\n");
      }
      return success();
    }

    // ---- __moore_array_max ----
    // Signature: (array: ptr, elemSize: i64, isSigned: i32)
    //            -> struct<(ptr, i64)>
    if (calleeName == "__moore_array_max") {
      if (callOp.getNumOperands() >= 3 && callOp.getNumResults() >= 1) {
        uint64_t arrayAddr =
            getValue(procId, callOp.getOperand(0)).getUInt64();
        int64_t elemSize = static_cast<int64_t>(
            getValue(procId, callOp.getOperand(1)).getUInt64());
        int32_t isSigned = static_cast<int32_t>(
            getValue(procId, callOp.getOperand(2)).getUInt64());
        MooreQueue result = {nullptr, 0};
        if (arrayAddr >= 0x10000000000ULL) {
          result = __moore_array_max(
              reinterpret_cast<MooreQueue *>(arrayAddr),
              elemSize, isSigned);
        }
        auto ptrVal = reinterpret_cast<uint64_t>(result.data);
        auto lenVal = static_cast<uint64_t>(result.len);
        APInt packedResult(128, 0);
        packedResult.insertBits(APInt(64, ptrVal), 0);
        packedResult.insertBits(APInt(64, lenVal), 64);
        setValue(procId, callOp.getResult(),
                 InterpretedValue(packedResult));
        LLVM_DEBUG(llvm::dbgs()
                   << "  llvm.call: __moore_array_max()\n");
      }
      return success();
    }

    // ---- __moore_display ----
    // Signature: (message_ptr: ptr) -> void
    if (calleeName == "__moore_display") {
      if (callOp.getNumOperands() >= 1) {
        std::string message = readStringFromPtr(callOp.getOperand(0));
        if (!message.empty()) {
          std::fwrite(message.data(), 1, message.size(), stdout);
        }
        std::fputc('\n', stdout);
        std::fflush(stdout);
        LLVM_DEBUG(llvm::dbgs() << "  llvm.call: __moore_display(\"" << message
                                << "\")\n");
      }
      return success();
    }

    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: function '" << calleeName
                            << "' is external (no body)\n");
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Gather argument values and operands (for signal reference tracking)
  SmallVector<InterpretedValue, 4> args;
  SmallVector<Value, 4> callOperands;
  for (Value arg : callOp.getArgOperands()) {
    args.push_back(getValue(procId, arg));
    callOperands.push_back(arg);
  }

  // Check call depth to prevent stack overflow from unbounded recursion.
  // Using 100 as the limit since each recursive call uses significant C++ stack
  // space due to the deep call chain: interpretLLVMFuncBody -> interpretOperation
  // -> interpretLLVMCall -> interpretLLVMFuncBody.
  constexpr size_t maxCallDepth = 200;
  if (state.callDepth >= maxCallDepth) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: max call depth (" << maxCallDepth
                            << ") exceeded for '" << calleeName << "'\n");
    // Return zero values for results instead of X to prevent cascading issues
    for (Value result : callOp.getResults()) {
      unsigned width = getTypeWidth(result.getType());
      setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
    }
    return success();
  }

  // Recursive DFS depth detection (same as func.call handler)
  Operation *llvmFuncKey = funcOp.getOperation();
  uint64_t llvmArg0Val = 0;
  bool llvmHasArg0 = !args.empty() && !args[0].isX();
  if (llvmHasArg0)
    llvmArg0Val = args[0].getUInt64();
  constexpr unsigned maxRecursionDepthLLVM = 20;
  auto &llvmDepthMap = state.recursionVisited[llvmFuncKey];
  if (llvmHasArg0 && state.callDepth > 0) {
    unsigned &depth = llvmDepthMap[llvmArg0Val];
    if (depth >= maxRecursionDepthLLVM) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  llvm.call: recursion depth " << depth
                 << " exceeded for '" << calleeName << "' with arg0=0x"
                 << llvm::format_hex(llvmArg0Val, 16) << "\n");
      for (Value result : callOp.getResults()) {
        unsigned width = getTypeWidth(result.getType());
        setValue(procId, result, InterpretedValue(llvm::APInt(width, 0)));
      }
      return success();
    }
  }
  bool llvmAddedToVisited = llvmHasArg0;
  if (llvmHasArg0)
    ++llvmDepthMap[llvmArg0Val];

  // Increment call depth before entering function
  ++state.callDepth;

  // Interpret the function body, passing call operands for signal mapping
  SmallVector<InterpretedValue, 2> results;
  LogicalResult funcResult =
      interpretLLVMFuncBody(procId, funcOp, args, results, callOperands);

  // Decrement call depth after returning
  --state.callDepth;

  // Decrement depth counter after returning
  if (llvmAddedToVisited) {
    auto &depthRef = processStates[procId].recursionVisited[llvmFuncKey][llvmArg0Val];
    if (depthRef > 0)
      --depthRef;
  }

  if (failed(funcResult))
    return failure();

  // Check if process suspended during function execution (e.g., due to wait)
  // If so, return early without setting results - the function didn't complete
  auto &postCallState = processStates[procId];
  if (postCallState.waiting) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: process suspended during call to '"
                            << calleeName << "'\n");
    return success();
  }

  // Set the return values
  for (auto [result, retVal] : llvm::zip(callOp.getResults(), results)) {
    setValue(procId, result, retVal);
  }

  LLVM_DEBUG(llvm::dbgs() << "  llvm.call: called '" << calleeName << "'\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMFuncBody(
    ProcessId procId, LLVM::LLVMFuncOp funcOp, ArrayRef<InterpretedValue> args,
    SmallVectorImpl<InterpretedValue> &results, ArrayRef<Value> callOperands) {
  // IMPORTANT: Do NOT hold a reference to processStates[procId] across operations
  // that may modify processStates (e.g., interpretOperation can create fork
  // children or runtime signals). DenseMap may reallocate, invalidating refs.

  // Map arguments to block arguments
  Block &entryBlock = funcOp.getBody().front();

  // Track signal mappings created for this call (to clean up later)
  SmallVector<Value, 4> tempSignalMappings;

  // Track recursion depth for this function. Use a local key to avoid
  // dangling reference - DenseMap may rehash during recursive calls.
  Operation *funcKey = funcOp.getOperation();
  unsigned currentDepth = ++funcCallDepth[funcKey];
  bool isRecursive = (currentDepth > 1);

  // Save ALL SSA values defined in this function when recursive.
  llvm::DenseMap<Value, InterpretedValue> savedFuncValues;
  llvm::DenseMap<Value, MemoryBlock> savedFuncMemBlocks;
  if (isRecursive) {
    auto &state = processStates[procId];
    for (Block &block : funcOp.getBody()) {
      for (auto arg : block.getArguments()) {
        auto it = state.valueMap.find(arg);
        if (it != state.valueMap.end())
          savedFuncValues[arg] = it->second;
        auto mIt = state.memoryBlocks.find(arg);
        if (mIt != state.memoryBlocks.end())
          savedFuncMemBlocks[arg] = mIt->second;
      }
      for (Operation &op : block) {
        for (auto result : op.getResults()) {
          auto it = state.valueMap.find(result);
          if (it != state.valueMap.end())
            savedFuncValues[result] = it->second;
          auto mIt = state.memoryBlocks.find(result);
          if (mIt != state.memoryBlocks.end())
            savedFuncMemBlocks[result] = mIt->second;
        }
      }
    }
  }

  for (auto [idx, blockArg] :
       llvm::enumerate(entryBlock.getArguments())) {
    if (idx < args.size())
      processStates[procId].valueMap[blockArg] = args[idx];

    // If call operands are provided, create signal mappings for BlockArguments
    // when the corresponding call operand resolves to a signal ID.
    // This allows llhd.prb to work on signal references passed as function args.
    if (idx < callOperands.size()) {
      if (SignalId sigId = resolveSignalId(callOperands[idx])) {
        valueToSignal[blockArg] = sigId;
        tempSignalMappings.push_back(blockArg);
        LLVM_DEBUG(llvm::dbgs()
                   << "  Created temp signal mapping for BlockArg " << idx
                   << " -> signal " << sigId << "\n");
      }
    }
  }

  // Helper to restore saved function values and decrement recursion depth
  auto restoreSavedFuncValues = [&]() {
    --funcCallDepth[funcKey];
    if (isRecursive) {
      auto &state = processStates[procId];
      for (const auto &[val, saved] : savedFuncValues)
        state.valueMap[val] = saved;
      for (auto &[val, saved] : savedFuncMemBlocks)
        state.memoryBlocks[val] = saved;
    }
  };

  // Set current function name for progress reporting and save previous
  auto &funcState = processStates[procId];
  std::string prevFuncName = funcState.currentFuncName;
  funcState.currentFuncName = funcOp.getName().str();

  // Helper to clean up temporary signal mappings and restore values
  auto cleanupTempMappings = [&]() {
    for (Value v : tempSignalMappings) {
      valueToSignal.erase(v);
    }
    restoreSavedFuncValues();
    // Restore previous function name
    auto it = processStates.find(procId);
    if (it != processStates.end())
      it->second.currentFuncName = prevFuncName;
  };

  // Execute the function body with operation limit to prevent infinite loops
  Block *currentBlock = &entryBlock;
  constexpr size_t maxOps = 1000000;
  size_t opCount = 0;

  while (currentBlock && opCount < maxOps) {
    for (Operation &op : *currentBlock) {
      ++opCount;
      // Track func body steps in process state for global step limiting
      {
        auto stIt = processStates.find(procId);
        if (stIt != processStates.end()) {
          ++stIt->second.totalSteps;
          ++stIt->second.funcBodySteps;
          // Progress report every 10M func body steps
          if (stIt->second.funcBodySteps % 10000000 == 0) {
            llvm::errs() << "[circt-sim] func progress: process " << procId
                         << " funcBodySteps=" << stIt->second.funcBodySteps
                         << " totalSteps=" << stIt->second.totalSteps
                         << " in '" << funcOp.getName() << "'"
                         << " (callDepth=" << stIt->second.callDepth << ")"
                         << " op=" << op.getName().getStringRef()
                         << "\n";
          }
          // Enforce global process step limit inside function bodies
          if (maxProcessSteps > 0 &&
              stIt->second.totalSteps > (size_t)maxProcessSteps) {
            llvm::errs()
                << "[circt-sim] ERROR(PROCESS_STEP_OVERFLOW in func): process "
                << procId << " exceeded " << maxProcessSteps
                << " total steps in LLVM function '" << funcOp.getName() << "'"
                << " (totalSteps=" << stIt->second.totalSteps << ")\n";
            stIt->second.halted = true;
            cleanupTempMappings();
            return failure();
          }
          // Periodically check for abort (timeout watchdog)
          if (stIt->second.funcBodySteps % 10000 == 0 && isAbortRequested()) {
            stIt->second.halted = true;
            cleanupTempMappings();
            if (abortCallback)
              abortCallback();
            return failure();
          }
        }
      }
      if (opCount >= maxOps) {
        LLVM_DEBUG(llvm::dbgs() << "  Warning: LLVM function '"
                                << funcOp.getName()
                                << "' reached max operations (" << maxOps
                                << ")\n");
        cleanupTempMappings();
        return failure();
      }

      // Handle return
      if (auto returnOp = dyn_cast<LLVM::ReturnOp>(&op)) {
        for (Value retVal : returnOp.getOperands()) {
          results.push_back(getValue(procId, retVal));
        }
        cleanupTempMappings();
        return success();
      }

      // Handle branch
      if (auto branchOp = dyn_cast<LLVM::BrOp>(&op)) {
        currentBlock = branchOp.getDest();
        for (auto [blockArg, operand] :
             llvm::zip(currentBlock->getArguments(), branchOp.getDestOperands())) {
          processStates[procId].valueMap[blockArg] = getValue(procId, operand);
        }
        break;
      }

      // Handle conditional branch
      if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(&op)) {
        InterpretedValue cond = getValue(procId, condBrOp.getCondition());
        if (!cond.isX() && cond.getUInt64() != 0) {
          currentBlock = condBrOp.getTrueDest();
          for (auto [blockArg, operand] :
               llvm::zip(currentBlock->getArguments(),
                        condBrOp.getTrueDestOperands())) {
            processStates[procId].valueMap[blockArg] = getValue(procId, operand);
          }
        } else {
          currentBlock = condBrOp.getFalseDest();
          for (auto [blockArg, operand] :
               llvm::zip(currentBlock->getArguments(),
                        condBrOp.getFalseDestOperands())) {
            processStates[procId].valueMap[blockArg] = getValue(procId, operand);
          }
        }
        break;
      }

      // Interpret other operations
      if (failed(interpretOperation(procId, &op))) {
        llvm::errs() << "circt-sim: Failed in LLVM func body for process "
                     << procId << "\n";
        llvm::errs() << "  Function: " << funcOp.getName() << "\n";
        llvm::errs() << "  Operation: ";
        op.print(llvm::errs(), OpPrintingFlags().printGenericOpForm());
        llvm::errs() << "\n";
        llvm::errs() << "  Location: " << op.getLoc() << "\n";
        cleanupTempMappings();
        return failure();
      }

      // Check if process was halted or is waiting (e.g., by sim.terminate,
      // llvm.unreachable, or moore.wait_event). This is critical for UVM where
      // wait_for_objection() contains event waits that must suspend execution.
      auto it = processStates.find(procId);
      if (it != processStates.end() && (it->second.halted || it->second.waiting)) {
        LLVM_DEBUG(llvm::dbgs() << "  Process halted/waiting during LLVM function body - "
                                << "returning early\n");
        cleanupTempMappings();
        return success();
      }
    }

    // If we didn't branch, we're done
    if (currentBlock == &entryBlock ||
        !currentBlock->back().hasTrait<OpTrait::IsTerminator>())
      break;
  }

  // If no return was encountered, return nothing
  cleanupTempMappings();
  return success();
}

//===----------------------------------------------------------------------===//
// RTTI Parent Table (for $cast hierarchy checking)
//===----------------------------------------------------------------------===//

void LLHDProcessInterpreter::loadRTTIParentTable() {
  if (rttiTableLoaded)
    return;
  rttiTableLoaded = true;

  if (!rootModule)
    return;

  // Look for the circt.rtti_parent_table module attribute emitted by
  // MooreToCore. It maps typeId -> parentTypeId (0 = root).
  // The attribute is on the builtin.module (rootModule itself), not its parent.
  auto tableAttr =
      rootModule->getAttrOfType<DenseIntElementsAttr>("circt.rtti_parent_table");
  if (!tableAttr)
    return;

  rttiParentTable.clear();
  for (auto val : tableAttr.getValues<int32_t>())
    rttiParentTable.push_back(val);

  LLVM_DEBUG(llvm::dbgs() << "Loaded RTTI parent table with "
                          << rttiParentTable.size() << " entries\n");
}

bool LLHDProcessInterpreter::checkRTTICast(int32_t srcTypeId,
                                            int32_t targetTypeId) {
  if (srcTypeId == 0 || targetTypeId == 0)
    return false;
  if (srcTypeId == targetTypeId)
    return true;

  // Load the RTTI table on first use
  loadRTTIParentTable();

  // If we have a hierarchy table, walk the parent chain
  if (!rttiParentTable.empty()) {
    int32_t current = srcTypeId;
    // Guard against infinite loops with a max depth
    for (int i = 0; i < 1000 && current != 0; ++i) {
      if (current < 0 || current >= static_cast<int32_t>(rttiParentTable.size()))
        break;
      current = rttiParentTable[current];
      if (current == targetTypeId)
        return true;
    }
    return false;
  }

  // Fallback: use the simple >= heuristic (backward compat for old MLIR files)
  return srcTypeId >= targetTypeId;
}

//===----------------------------------------------------------------------===//
// Global Variable and VTable Support
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::initializeGlobals() {
  if (!rootModule)
    return success();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initializing globals\n");

  // === Use iterative discovery for globals (stack overflow prevention) ===
  DiscoveredGlobalOps globalOps;
  discoverGlobalOpsIteratively(globalOps);

  // Process all pre-discovered LLVM global operations (no walk() needed)
  for (LLVM::GlobalOp globalOp : globalOps.globals) {
    StringRef globalName = globalOp.getSymName();

    LLVM_DEBUG(llvm::dbgs() << "  Found global: " << globalName << "\n");

    // Get the global's type to calculate size
    Type globalType = globalOp.getGlobalType();
    unsigned size = getLLVMTypeSize(globalType);
    if (size == 0)
      size = 8; // Default minimum size

    // Allocate memory for the global
    uint64_t addr = nextGlobalAddress;
    nextGlobalAddress += ((size + 7) / 8) * 8; // Align to 8 bytes

    globalAddresses[globalName] = addr;

    // Also populate the reverse map for address-to-global lookup
    addressToGlobal[addr] = globalName.str();

    // Create memory block
    MemoryBlock block(size, 64);

    // Check the initializer attribute
    // Handle both #llvm.zero and string constant initializers
    if (auto initAttr = globalOp.getValueOrNull()) {
      block.initialized = true;

      // Check if this is a string initializer
      if (auto strAttr = dyn_cast<StringAttr>(initAttr)) {
        StringRef strContent = strAttr.getValue();
        // Copy the string content to the memory block
        size_t copyLen = std::min(strContent.size(), block.data.size());
        std::memcpy(block.data.data(), strContent.data(), copyLen);
        LLVM_DEBUG(llvm::dbgs() << "    Initialized with string: \""
                                << strContent << "\" (" << copyLen << " bytes)\n");
      } else {
        // For #llvm.zero or other initializers, data is already zeroed
        LLVM_DEBUG(llvm::dbgs() << "    Initialized to zero\n");
      }
    }

    // Check if this is a vtable (has circt.vtable_entries attribute)
    if (auto vtableEntriesAttr = globalOp->getAttr("circt.vtable_entries")) {
      LLVM_DEBUG(llvm::dbgs() << "    This is a vtable with entries\n");

      if (auto entriesArray = dyn_cast<ArrayAttr>(vtableEntriesAttr)) {
        // Each entry is [index, funcSymbol]
        for (auto entry : entriesArray) {
          if (auto entryArray = dyn_cast<ArrayAttr>(entry)) {
            if (entryArray.size() >= 2) {
              auto indexAttr = dyn_cast<IntegerAttr>(entryArray[0]);
              auto funcSymbol = dyn_cast<FlatSymbolRefAttr>(entryArray[1]);

              if (indexAttr && funcSymbol) {
                unsigned index = indexAttr.getInt();
                StringRef funcName = funcSymbol.getValue();

                // Create a unique "function address" for this function
                // We use a simple scheme: high bits identify it as a function ptr
                uint64_t funcAddr = 0xF0000000 + addressToFunction.size();
                addressToFunction[funcAddr] = funcName.str();

                // Store the function address in the vtable memory
                // (little-endian)
                for (unsigned i = 0; i < 8 && (index * 8 + i) < block.data.size(); ++i) {
                  block.data[index * 8 + i] = (funcAddr >> (i * 8)) & 0xFF;
                }
                block.initialized = true;

                LLVM_DEBUG(llvm::dbgs() << "      Entry " << index << ": "
                                        << funcName << " -> 0x"
                                        << llvm::format_hex(funcAddr, 16) << "\n");
              }
            }
          }
        }
      }
    }

    globalMemoryBlocks[globalName] = std::move(block);
  }

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initialized "
                          << globalMemoryBlocks.size() << " globals, "
                          << addressToFunction.size() << " vtable entries\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::executeGlobalConstructors() {
  if (!rootModule)
    return success();

  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: Executing global constructors\n");

  // === Use iterative discovery for global constructors (stack overflow prevention) ===
  DiscoveredGlobalOps globalOps;
  discoverGlobalOpsIteratively(globalOps);

  // Collect all constructor entries with their priorities from pre-discovered ops
  SmallVector<std::pair<int32_t, StringRef>, 4> ctorEntries;

  for (LLVM::GlobalCtorsOp ctorsOp : globalOps.ctors) {
    ArrayAttr ctors = ctorsOp.getCtors();
    ArrayAttr priorities = ctorsOp.getPriorities();

    for (auto [ctorAttr, priorityAttr] : llvm::zip(ctors, priorities)) {
      auto ctorRef = cast<FlatSymbolRefAttr>(ctorAttr);
      auto priority = cast<IntegerAttr>(priorityAttr).getInt();
      ctorEntries.emplace_back(priority, ctorRef.getValue());
      LLVM_DEBUG(llvm::dbgs() << "  Found constructor: " << ctorRef.getValue()
                              << " (priority " << priority << ")\n");
    }
  }

  if (ctorEntries.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No global constructors found\n");
    return success();
  }

  // Sort by priority (lower priority values execute first)
  llvm::sort(ctorEntries,
             [](const auto &a, const auto &b) { return a.first < b.first; });

  // Create a temporary process state for executing constructors.
  // Must use a non-zero ID because InvalidProcessId == 0 and
  // findMemoryBlockByAddress's walk loop skips process ID 0.
  ProcessExecutionState tempState;
  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId) || tempProcId == InvalidProcessId)
    tempProcId = nextTempProcId++;
  processStates[tempProcId] = std::move(tempState);

  // Execute each constructor in priority order
  for (auto &[priority, ctorName] : ctorEntries) {
    LLVM_DEBUG(llvm::dbgs() << "  Calling constructor: " << ctorName
                            << " (priority " << priority << ")\n");

    // Reset the temporary process state between constructors.
    // Global constructors are independent; if one sets halted/waiting
    // (e.g., due to an X vtable dispatch that triggers llvm.unreachable),
    // subsequent constructors must not inherit that state.
    {
      auto &ts = processStates[tempProcId];
      ts.halted = false;
      ts.waiting = false;
    }

    // Look up the LLVM function
    auto funcOp = rootModule.lookupSymbol<LLVM::LLVMFuncOp>(ctorName);
    if (!funcOp) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: constructor function '"
                              << ctorName << "' not found\n");
      continue;
    }

    // Call the constructor with no arguments
    SmallVector<InterpretedValue, 2> results;
    if (failed(interpretLLVMFuncBody(tempProcId, funcOp, {}, results))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Warning: failed to execute constructor '" << ctorName
                 << "'\n");
      // Continue with other constructors even if one fails
    }
  }

  // Clean up the temporary process state
  processStates.erase(tempProcId);

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executed "
                          << ctorEntries.size() << " global constructors\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::executeModuleLevelLLVMOps(
    hw::HWModuleOp hwModule) {
  LLVM_DEBUG(llvm::dbgs()
             << "LLHDProcessInterpreter: Executing module-level LLVM ops\n");

  // Create a temporary process state for executing module-level ops.
  // Must use a non-zero ID because InvalidProcessId == 0 and
  // findMemoryBlockByAddress's walk loop skips process ID 0.
  ProcessExecutionState tempState;
  ProcessId tempProcId = nextTempProcId++;
  while (processStates.count(tempProcId) || tempProcId == InvalidProcessId)
    tempProcId = nextTempProcId++;
  processStates[tempProcId] = std::move(tempState);

  unsigned opsExecuted = 0;

  // Walk the module body (but not inside processes) and execute LLVM ops
  // We need to execute them in order, so iterate through the block directly.
  for (Operation &op : hwModule.getBody().front()) {
    // Skip llhd.process and seq.initial - those have their own execution
    if (isa<llhd::ProcessOp, seq::InitialOp, llhd::CombinationalOp>(&op))
      continue;

    // Execute LLVM operations that need initialization
    if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(&op)) {
      (void)interpretLLVMAlloca(tempProcId, allocaOp);
      ++opsExecuted;
    } else if (auto storeOp = dyn_cast<LLVM::StoreOp>(&op)) {
      (void)interpretLLVMStore(tempProcId, storeOp);
      ++opsExecuted;
    } else if (auto callOp = dyn_cast<LLVM::CallOp>(&op)) {
      (void)interpretLLVMCall(tempProcId, callOp);
      ++opsExecuted;
    } else if (auto constOp = dyn_cast<LLVM::ConstantOp>(&op)) {
      // Evaluate LLVM constants so they're in the value map
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        setValue(tempProcId, constOp.getResult(),
                 InterpretedValue(intAttr.getValue()));
        ++opsExecuted;
      }
    } else if (auto hwConstOp = dyn_cast<hw::ConstantOp>(&op)) {
      // Evaluate HW constants (used as arguments to LLVM calls)
      setValue(tempProcId, hwConstOp.getResult(),
               InterpretedValue(hwConstOp.getValue()));
      ++opsExecuted;
    } else if (auto undefOp = dyn_cast<LLVM::UndefOp>(&op)) {
      unsigned width = getTypeWidth(undefOp.getType());
      setValue(tempProcId, undefOp.getResult(),
               InterpretedValue(APInt::getZero(width)));
      ++opsExecuted;
    } else if (auto zeroOp = dyn_cast<LLVM::ZeroOp>(&op)) {
      setValue(tempProcId, zeroOp.getResult(), InterpretedValue(0, 64));
      ++opsExecuted;
    } else if (isa<LLVM::InsertValueOp>(&op)) {
      (void)interpretOperation(tempProcId, &op);
      ++opsExecuted;
    }
  }

  // Copy the module-level value map to a special "module init" state
  // that processes can access for module-level values
  moduleInitValueMap = std::move(processStates[tempProcId].valueMap);

  // Copy module-level memory blocks too
  for (auto &[value, block] : processStates[tempProcId].memoryBlocks) {
    moduleLevelAllocas[value] = std::move(block);
  }

  // Clean up the temporary process state
  processStates.erase(tempProcId);

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Executed " << opsExecuted
                          << " module-level LLVM ops\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMAddressOf(
    ProcessId procId, LLVM::AddressOfOp addrOfOp) {
  StringRef globalName = addrOfOp.getGlobalName();

  // Look up the global's address
  auto it = globalAddresses.find(globalName);
  if (it == globalAddresses.end()) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.addressof: global '" << globalName
                            << "' not found, returning X\n");
    setValue(procId, addrOfOp.getResult(),
             InterpretedValue::makeX(64));
    return success();
  }

  uint64_t addr = it->second;
  setValue(procId, addrOfOp.getResult(), InterpretedValue(addr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.addressof: " << globalName << " = 0x"
                          << llvm::format_hex(addr, 16) << "\n");

  return success();
}
