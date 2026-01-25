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
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

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

  // Store the module name for hierarchical path construction
  moduleName = hwModule.getName().str();

  // Store the root module for symbol lookup
  rootModule = hwModule->getParentOfType<ModuleOp>();

  // Register all signals first
  if (failed(registerSignals(hwModule)))
    return failure();

  // Export signals to MooreRuntime signal registry for DPI/VPI access
  exportSignalsToRegistry();

  // Then register all processes
  if (failed(registerProcesses(hwModule)))
    return failure();

  // Register combinational processes for static module-level drives
  // (continuous assignments like port connections)
  registerContinuousAssignments(hwModule);

  // Recursively process child module instances
  if (failed(initializeChildInstances(hwModule)))
    return failure();

  // Initialize LLVM global variables (especially vtables)
  if (failed(initializeGlobals()))
    return failure();

  // Execute LLVM global constructors (e.g., __moore_global_init_uvm_pkg::uvm_top)
  // This initializes UVM globals like uvm_top before processes start
  if (failed(executeGlobalConstructors()))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Registered "
                          << getNumSignals() << " signals and "
                          << getNumProcesses() << " processes\n");

  return success();
}

LogicalResult
LLHDProcessInterpreter::initializeChildInstances(hw::HWModuleOp hwModule) {
  // Find all hw.instance operations in this module
  hwModule.walk([&](hw::InstanceOp instOp) {
    // Get the referenced module name
    StringRef childModuleName = instOp.getReferencedModuleName();

    LLVM_DEBUG(llvm::dbgs() << "  Found instance '" << instOp.getInstanceName()
                            << "' of module '" << childModuleName << "'\n");

    // Look up the child module in the symbol table
    if (!rootModule) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: No root module for symbol lookup\n");
      return;
    }

    auto childModule =
        rootModule.lookupSymbol<hw::HWModuleOp>(childModuleName);
    if (!childModule) {
      LLVM_DEBUG(llvm::dbgs() << "    Warning: Could not find module '"
                              << childModuleName << "'\n");
      return;
    }

    // Skip if we've already processed this module (to handle multiple instances)
    if (processedModules.contains(childModuleName)) {
      LLVM_DEBUG(llvm::dbgs() << "    Skipping already processed module\n");
      return;
    }
    processedModules.insert(childModuleName);

    // Register signals and processes from the child module
    // Note: We don't recursively call initialize() to avoid re-exporting signals
    // and to maintain the current hierarchical context

    // Register signals from child module
    childModule.walk([&](llhd::SignalOp sigOp) {
      // Use hierarchical name for child signals
      std::string name = sigOp.getName().value_or("").str();
      if (name.empty()) {
        name = "sig_" + std::to_string(valueToSignal.size());
      }
      std::string hierName = instOp.getInstanceName().str() + "." + name;

      Type innerType = sigOp.getInit().getType();
      unsigned width = getTypeWidth(innerType);

      SignalId sigId = scheduler.registerSignal(hierName, width);
      valueToSignal[sigOp.getResult()] = sigId;
      signalIdToName[sigId] = hierName;

      LLVM_DEBUG(llvm::dbgs() << "    Registered child signal '" << hierName
                              << "' with ID " << sigId << "\n");
    });

    // Register processes from child module
    childModule.walk([&](llhd::ProcessOp processOp) {
      std::string procName = instOp.getInstanceName().str() + ".llhd_process_" +
                             std::to_string(processStates.size());

      ProcessExecutionState state(processOp);
      ProcessId procId = scheduler.registerProcess(
          procName, [this, procId = processStates.size() + 1]() {
            executeProcess(procId);
          });

      state.currentBlock = &processOp.getBody().front();
      state.currentOp = state.currentBlock->begin();
      processStates[procId] = std::move(state);
      opToProcessId[processOp.getOperation()] = procId;

      LLVM_DEBUG(llvm::dbgs() << "    Registered child process '" << procName
                              << "' with ID " << procId << "\n");

      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    });

    // Register seq.initial blocks from child module
    childModule.walk([&](seq::InitialOp initialOp) {
      std::string initName = instOp.getInstanceName().str() + ".seq_initial_" +
                             std::to_string(processStates.size());

      ProcessExecutionState state(initialOp);
      ProcessId procId = scheduler.registerProcess(
          initName, [this, procId = processStates.size() + 1]() {
            executeProcess(procId);
          });

      state.currentBlock = initialOp.getBodyBlock();
      state.currentOp = state.currentBlock->begin();
      processStates[procId] = std::move(state);
      opToProcessId[initialOp.getOperation()] = procId;

      LLVM_DEBUG(llvm::dbgs() << "    Registered child initial block '" << initName
                              << "' with ID " << procId << "\n");

      scheduler.scheduleProcess(procId, SchedulingRegion::Active);
    });

    // Recursively process child module's instances
    (void)initializeChildInstances(childModule);
  });

  return success();
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Flatten an aggregate constant (struct or array) into a single APInt.
/// This is used for initializing signals with aggregate types.
/// For 4-state logic, structs typically have the form {value, unknown}.
static APInt flattenAggregateConstant(hw::AggregateConstantOp aggConstOp) {
  ArrayAttr fields = aggConstOp.getFields();
  Type resultType = aggConstOp.getResult().getType();

  // Calculate total width
  unsigned totalWidth = LLHDProcessInterpreter::getTypeWidth(resultType);
  APInt result(totalWidth, 0);

  // For struct types, concatenate field values from high to low
  if (auto structType = dyn_cast<hw::StructType>(resultType)) {
    auto elements = structType.getElements();
    unsigned bitOffset = totalWidth;

    for (size_t i = 0; i < fields.size() && i < elements.size(); ++i) {
      Attribute fieldAttr = fields[i];
      unsigned fieldWidth =
          LLHDProcessInterpreter::getTypeWidth(elements[i].type);
      bitOffset -= fieldWidth;

      if (auto intAttr = dyn_cast<IntegerAttr>(fieldAttr)) {
        APInt fieldValue = intAttr.getValue();
        if (fieldValue.getBitWidth() < fieldWidth)
          fieldValue = fieldValue.zext(fieldWidth);
        else if (fieldValue.getBitWidth() > fieldWidth)
          fieldValue = fieldValue.trunc(fieldWidth);
        result.insertBits(fieldValue, bitOffset);
      }
      // Nested arrays/structs would need recursive handling - for now, they
      // remain zero
    }
  }
  // For array types, concatenate elements
  else if (auto arrayType = dyn_cast<hw::ArrayType>(resultType)) {
    unsigned elementWidth =
        LLHDProcessInterpreter::getTypeWidth(arrayType.getElementType());
    unsigned bitOffset = totalWidth;

    for (Attribute fieldAttr : fields) {
      bitOffset -= elementWidth;
      if (auto intAttr = dyn_cast<IntegerAttr>(fieldAttr)) {
        APInt fieldValue = intAttr.getValue();
        if (fieldValue.getBitWidth() < elementWidth)
          fieldValue = fieldValue.zext(elementWidth);
        else if (fieldValue.getBitWidth() > elementWidth)
          fieldValue = fieldValue.trunc(elementWidth);
        result.insertBits(fieldValue, bitOffset);
      }
    }
  }

  return result;
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
    // SignalValue only supports up to 64 bits, truncate if wider
    uint64_t val64 = initValue.getBitWidth() > 64
                         ? initValue.trunc(64).getZExtValue()
                         : initValue.getZExtValue();
    unsigned svWidth = width > 64 ? 64 : width;
    SignalValue sv(val64, svWidth);
    scheduler.updateSignal(sigId, sv);
    LLVM_DEBUG(llvm::dbgs() << "  Set initial value to " << initValue << "\n");
  } else if (auto aggConstOp =
                 sigOp.getInit().getDefiningOp<hw::AggregateConstantOp>()) {
    // Handle aggregate constant (struct/array) initialization
    // For 4-state logic signals, the struct contains {value, unknown} fields
    // We need to flatten the aggregate into a single APInt value
    APInt initValue = flattenAggregateConstant(aggConstOp);
    // SignalValue only supports up to 64 bits, truncate if wider
    uint64_t val64 = initValue.getBitWidth() > 64
                         ? initValue.trunc(64).getZExtValue()
                         : initValue.getZExtValue();
    unsigned svWidth = width > 64 ? 64 : width;
    SignalValue sv(val64, svWidth);
    scheduler.updateSignal(sigId, sv);
    LLVM_DEBUG(llvm::dbgs() << "  Set aggregate initial value to " << initValue
                            << "\n");
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
LLHDProcessInterpreter::registerProcesses(hw::HWModuleOp hwModule) {
  // Walk the module body to find all llhd.process operations
  // Note: walk() visits operations in nested regions, so we need to ensure
  // we're only registering top-level processes (not processes inside other
  // processes, which shouldn't exist but let's be safe).
  hwModule.walk([&](llhd::ProcessOp processOp) {
    LLVM_DEBUG(llvm::dbgs() << "  Found llhd.process op (numResults="
                            << processOp.getNumResults() << ")\n");
    registerProcess(processOp);
  });

  // Also handle llhd.combinational operations
  hwModule.walk([&](llhd::CombinationalOp combOp) {
    // TODO: Handle combinational processes in Phase 1B
    LLVM_DEBUG(llvm::dbgs() << "  Found combinational process (TODO)\n");
  });

  // Also handle seq.initial operations for $display/$finish support
  hwModule.walk([&](seq::InitialOp initialOp) {
    registerInitialBlock(initialOp);
  });

  // Handle module-level llhd.drv operations
  // These are continuous assignments driven by process results
  hwModule.getBody().walk([&](llhd::DriveOp driveOp) {
    // Only handle drives at module level (not inside processes)
    if (!driveOp->getParentOfType<llhd::ProcessOp>() &&
        !driveOp->getParentOfType<seq::InitialOp>()) {
      registerModuleDrive(driveOp);
    }
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

ProcessId LLHDProcessInterpreter::registerInitialBlock(seq::InitialOp initialOp) {
  // Generate a process name for the initial block
  std::string name = "seq_initial_" + std::to_string(processStates.size());

  // Create the execution state for this initial block
  ProcessExecutionState state(initialOp);

  // Register with the scheduler, providing a callback that executes this block
  ProcessId procId = scheduler.registerProcess(
      name, [this, procId = processStates.size() + 1]() {
        executeProcess(procId);
      });

  // Store the state - initial blocks have a body with a single block
  state.currentBlock = initialOp.getBodyBlock();
  state.currentOp = state.currentBlock->begin();
  processStates[procId] = std::move(state);
  opToProcessId[initialOp.getOperation()] = procId;

  LLVM_DEBUG(llvm::dbgs() << "  Registered initial block '" << name
                          << "' with ID " << procId << "\n");

  // Schedule the initial block to run at time 0 (initialization)
  scheduler.scheduleProcess(procId, SchedulingRegion::Active);

  return procId;
}

void LLHDProcessInterpreter::registerModuleDrive(llhd::DriveOp driveOp) {
  // Module-level drives need special handling:
  // The drive value comes from process results which are populated when
  // the process executes llhd.wait yield or llhd.halt with yield operands.
  //
  // For each module-level drive, we need to:
  // 1. Track the drive operation
  // 2. Identify the source process (if the value comes from a process result)
  // 3. Schedule the drive when the process yields

  Value driveValue = driveOp.getValue();

  // Check if the drive value comes from a process result
  if (auto processOp = driveValue.getDefiningOp<llhd::ProcessOp>()) {
    // Find the process ID for this process
    auto procIt = opToProcessId.find(processOp.getOperation());
    if (procIt != opToProcessId.end()) {
      ProcessId procId = procIt->second;

      // Store this drive for later execution when the process yields
      moduleDrives.push_back({driveOp, procId});

      LLVM_DEBUG(llvm::dbgs() << "  Registered module-level drive connected to "
                              << "process " << procId << "\n");
    }
  } else {
    // For non-process-connected drives, schedule them immediately
    // This handles static/constant drives at module level
    LLVM_DEBUG(llvm::dbgs() << "  Found module-level drive (static)\n");
    // These will be handled during initialization
    staticModuleDrives.push_back(driveOp);
  }
}

void LLHDProcessInterpreter::executeModuleDrives(ProcessId procId) {
  // Execute all module-level drives connected to this process
  for (auto &[driveOp, driveProcId] : moduleDrives) {
    if (driveProcId != procId)
      continue;

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
    scheduler.getEventScheduler().schedule(
        targetTime, SchedulingRegion::NBA,
        Event([this, sigId, newVal]() {
          scheduler.updateSignal(sigId, newVal);
        }));
  }
}

void LLHDProcessInterpreter::registerContinuousAssignments(
    hw::HWModuleOp hwModule) {
  // For each static module-level drive, we need to:
  // 1. Find which signals the drive value depends on (via llhd.prb)
  // 2. Create a combinational process that re-executes when those signals change
  // 3. The process evaluates the drive value and schedules the signal update

  for (llhd::DriveOp driveOp : staticModuleDrives) {
    // Find the signal being driven
    SignalId targetSigId = getSignalId(driveOp.getSignal());
    if (targetSigId == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: Unknown target signal in continuous assignment\n");
      continue;
    }

    // Find all source signals (signals that are probed to compute the value)
    llvm::SmallVector<SignalId, 4> sourceSignals;

    // Walk back from the drive value to find all llhd.prb operations
    llvm::SmallVector<mlir::Value, 8> worklist;
    llvm::DenseSet<mlir::Value> visited;
    worklist.push_back(driveOp.getValue());

    while (!worklist.empty()) {
      mlir::Value val = worklist.pop_back_val();
      if (!visited.insert(val).second)
        continue;

      // If this value comes from a probe, record the signal
      if (auto probeOp = val.getDefiningOp<llhd::ProbeOp>()) {
        SignalId srcSigId = getSignalId(probeOp.getSignal());
        if (srcSigId != 0 && srcSigId != targetSigId) {
          sourceSignals.push_back(srcSigId);
        }
        continue;
      }

      // For other operations, add their operands to the worklist
      if (Operation *defOp = val.getDefiningOp()) {
        for (Value operand : defOp->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }

    if (sourceSignals.empty()) {
      // No source signals - this is a constant drive, execute once at init
      LLVM_DEBUG(llvm::dbgs()
                 << "  Constant continuous assignment to signal " << targetSigId
                 << "\n");
      executeContinuousAssignment(driveOp);
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
        processName, [this, driveOp]() { executeContinuousAssignment(driveOp); });

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
  scheduler.getEventScheduler().schedule(
      targetTime, SchedulingRegion::Active,
      Event([this, targetSigId, newVal]() {
        scheduler.updateSignal(targetSigId, newVal);
      }));
}

/// Evaluate a value for continuous assignments by reading from current signal
/// state.
InterpretedValue LLHDProcessInterpreter::evaluateContinuousValue(
    mlir::Value value) {
  // Check if this is a probe operation - read from signal state
  if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
    SignalId sigId = getSignalId(probeOp.getSignal());
    if (sigId != 0) {
      const SignalValue &sv = scheduler.getSignalValue(sigId);
      return InterpretedValue::fromSignalValue(sv);
    }
    return InterpretedValue::makeX(getTypeWidth(value.getType()));
  }

  // Handle constants
  if (auto constOp = value.getDefiningOp<hw::ConstantOp>()) {
    return InterpretedValue(constOp.getValue());
  }

  // Handle aggregate constants
  if (auto aggConstOp = value.getDefiningOp<hw::AggregateConstantOp>()) {
    llvm::APInt flatValue = flattenAggregateConstant(aggConstOp);
    return InterpretedValue(flatValue);
  }

  // Handle struct extract
  if (auto extractOp = value.getDefiningOp<hw::StructExtractOp>()) {
    InterpretedValue inputVal = evaluateContinuousValue(extractOp.getInput());
    if (inputVal.isX())
      return InterpretedValue::makeX(getTypeWidth(value.getType()));

    auto structType = hw::type_cast<hw::StructType>(extractOp.getInput().getType());
    StringRef fieldName = extractOp.getFieldName();

    // Find the field offset and width
    unsigned bitOffset = 0;
    unsigned fieldWidth = 0;
    auto elements = structType.getElements();
    // Struct fields are packed from high bits to low bits
    unsigned totalWidth = getTypeWidth(structType);
    for (auto &element : elements) {
      unsigned elemWidth = getTypeWidth(element.type);
      if (element.name == fieldName) {
        fieldWidth = elemWidth;
        break;
      }
      bitOffset += elemWidth;
    }

    // Extract the field value
    uint64_t val = inputVal.getUInt64();
    // bitOffset is from the MSB, so we need to adjust
    unsigned shiftAmount = totalWidth - bitOffset - fieldWidth;
    uint64_t fieldVal = (val >> shiftAmount) & ((1ULL << fieldWidth) - 1);
    return InterpretedValue(fieldVal, fieldWidth);
  }

  // Handle struct create
  if (auto createOp = value.getDefiningOp<hw::StructCreateOp>()) {
    auto structType = hw::type_cast<hw::StructType>(createOp.getType());
    unsigned totalWidth = getTypeWidth(structType);
    llvm::APInt result(totalWidth, 0);

    auto elements = structType.getElements();
    unsigned bitOffset = totalWidth;
    for (size_t i = 0; i < createOp.getInput().size(); ++i) {
      InterpretedValue fieldVal =
          evaluateContinuousValue(createOp.getInput()[i]);
      unsigned fieldWidth = getTypeWidth(elements[i].type);
      bitOffset -= fieldWidth;

      if (!fieldVal.isX()) {
        APInt fieldBits(fieldWidth, fieldVal.getUInt64());
        result.insertBits(fieldBits, bitOffset);
      }
    }
    return InterpretedValue(result);
  }

  // Handle comb operations
  if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
    InterpretedValue lhs = evaluateContinuousValue(xorOp.getOperand(0));
    InterpretedValue rhs = evaluateContinuousValue(xorOp.getOperand(1));
    if (lhs.isX() || rhs.isX())
      return InterpretedValue::makeX(getTypeWidth(value.getType()));
    return InterpretedValue(lhs.getUInt64() ^ rhs.getUInt64(),
                            lhs.getWidth());
  }

  if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
    uint64_t result = ~0ULL;
    unsigned width = getTypeWidth(value.getType());
    for (Value operand : andOp.getOperands()) {
      InterpretedValue opVal = evaluateContinuousValue(operand);
      if (opVal.isX())
        return InterpretedValue::makeX(width);
      result &= opVal.getUInt64();
    }
    return InterpretedValue(result, width);
  }

  if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
    uint64_t result = 0;
    unsigned width = getTypeWidth(value.getType());
    for (Value operand : orOp.getOperands()) {
      InterpretedValue opVal = evaluateContinuousValue(operand);
      if (opVal.isX())
        return InterpretedValue::makeX(width);
      result |= opVal.getUInt64();
    }
    return InterpretedValue(result, width);
  }

  if (auto icmpOp = value.getDefiningOp<comb::ICmpOp>()) {
    InterpretedValue lhs = evaluateContinuousValue(icmpOp.getLhs());
    InterpretedValue rhs = evaluateContinuousValue(icmpOp.getRhs());
    if (lhs.isX() || rhs.isX())
      return InterpretedValue::makeX(1);

    bool result = false;
    uint64_t lVal = lhs.getUInt64();
    uint64_t rVal = rhs.getUInt64();

    switch (icmpOp.getPredicate()) {
    case comb::ICmpPredicate::eq:
      result = (lVal == rVal);
      break;
    case comb::ICmpPredicate::ne:
      result = (lVal != rVal);
      break;
    case comb::ICmpPredicate::ult:
      result = (lVal < rVal);
      break;
    case comb::ICmpPredicate::ule:
      result = (lVal <= rVal);
      break;
    case comb::ICmpPredicate::ugt:
      result = (lVal > rVal);
      break;
    case comb::ICmpPredicate::uge:
      result = (lVal >= rVal);
      break;
    default:
      // Signed comparisons would need sign extension handling
      result = false;
      break;
    }
    return InterpretedValue(result ? 1ULL : 0ULL, 1);
  }

  // Default: return unknown
  LLVM_DEBUG(llvm::dbgs() << "  Warning: Cannot evaluate continuous value for "
                          << *value.getDefiningOp() << "\n");
  return InterpretedValue::makeX(getTypeWidth(value.getType()));
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
  } else if (state.waiting) {
    // Handle the case where a process was triggered by an event (via
    // triggerSensitiveProcesses) rather than by the delay callback
    // (resumeProcess). In this case, state.waiting may still be true but
    // destBlock is null because the scheduler directly scheduled the process.
    // This can happen when a process is triggered by a signal change while
    // it was waiting for that signal.
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
    SignalId sigId = scheduler.registerSignal(name, width);

    // Store the mapping from the signal result to the signal ID
    valueToSignal[sigOp.getResult()] = sigId;
    signalIdToName[sigId] = name;

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
      result &= operandVal;
    }
    setValue(procId, andOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto orOp = dyn_cast<comb::OrOp>(op)) {
    unsigned targetWidth = getTypeWidth(orOp.getType());
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
      result |= operandVal;
    }
    setValue(procId, orOp.getResult(), InterpretedValue(result));
    return success();
  }

  if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
    unsigned targetWidth = getTypeWidth(xorOp.getType());
    llvm::APInt result(targetWidth, 0); // Start with all 0s for XOR
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
      result ^= operandVal;
    }
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
    unsigned targetWidth = getTypeWidth(addOp.getType());

    if (lhs.isX() || rhs.isX()) {
      setValue(procId, addOp.getResult(),
               InterpretedValue::makeX(targetWidth));
    } else {
      APInt lhsVal = lhs.getAPInt();
      APInt rhsVal = rhs.getAPInt();
      normalizeWidths(lhsVal, rhsVal, targetWidth);
      APInt result = lhsVal + rhsVal;
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

    if (funcPtrVal.isX()) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: callee is X\n");
      for (Value result : callIndirectOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
      return success();
    }

    // Look up the function name from the vtable
    uint64_t funcAddr = funcPtrVal.getUInt64();
    auto it = addressToFunction.find(funcAddr);
    if (it == addressToFunction.end()) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: address 0x"
                              << llvm::format_hex(funcAddr, 16)
                              << " not in vtable map\n");
      for (Value result : callIndirectOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
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

    if (!parent) {
      LLVM_DEBUG(llvm::dbgs() << "  func.call_indirect: could not find module\n");
      for (Value result : callIndirectOp.getResults()) {
        setValue(procId, result,
                 InterpretedValue::makeX(getTypeWidth(result.getType())));
      }
      return success();
    }

    auto moduleOp = cast<ModuleOp>(parent);
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

    // Call the function
    SmallVector<InterpretedValue, 2> results;
    if (failed(interpretFuncBody(procId, funcOp, args, results)))
      return failure();

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
  // HW Struct Operations
  //===--------------------------------------------------------------------===//

  if (auto structExtractOp = dyn_cast<hw::StructExtractOp>(op)) {
    InterpretedValue structVal = getValue(procId, structExtractOp.getInput());
    if (structVal.isX()) {
      setValue(procId, structExtractOp.getResult(),
               InterpretedValue::makeX(getTypeWidth(structExtractOp.getType())));
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
      if (val.isX()) {
        hasX = true;
        break;
      }
      unsigned fieldWidth = getTypeWidth(elements[i].type);
      bitOffset -= fieldWidth;
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

  // Handle builtin.unrealized_conversion_cast - propagate values through
  if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    // For casts, propagate the input values to the output values
    if (castOp.getNumOperands() == castOp.getNumResults()) {
      // Simple 1:1 mapping
      for (auto [input, output] : llvm::zip(castOp.getInputs(), castOp.getOutputs())) {
        InterpretedValue val = getValue(procId, input);
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
  // Get the signal ID for the probed signal
  SignalId sigId = getSignalId(probeOp.getSignal());
  if (sigId == 0) {
    // Check if this is a global variable access via UnrealizedConversionCastOp
    // This happens when static class properties are accessed - they're stored
    // in LLVM globals, not LLHD signals.
    Value signal = probeOp.getSignal();
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
      }
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

  // Generate a unique driver ID based on the DriveOp
  // Use hash of operation pointer as driver ID
  uint64_t driverId = reinterpret_cast<uint64_t>(driveOp.getOperation());

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

    // Helper function to recursively trace back through operations to find
    // the signal being observed. This handles chains like:
    //   %clk_bool = comb.and %value, %not_unknown
    //   %value = hw.struct_extract %probed["value"]
    //   %probed = llhd.prb %signal
    // We need to trace back through this chain to find %signal.
    std::function<SignalId(Value, int)> traceToSignal;
    traceToSignal = [&](Value value, int depth) -> SignalId {
      // Limit recursion depth to prevent infinite loops
      if (depth > 10)
        return 0;

      // Direct probe case
      if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
        SignalId sigId = getSignalId(probeOp.getSignal());
        if (sigId != 0) {
          LLVM_DEBUG(llvm::dbgs() << "  Found signal " << sigId
                                  << " from probe at depth " << depth << "\n");
          return sigId;
        }
      }

      // Check if it's a signal reference directly
      SignalId sigId = getSignalId(value);
      if (sigId != 0)
        return sigId;

      // Block argument - trace through predecessors
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        Block *block = blockArg.getOwner();
        unsigned argIdx = blockArg.getArgNumber();

        // Look at all predecessors
        for (Block *pred : block->getPredecessors()) {
          Operation *terminator = pred->getTerminator();
          if (auto branchOp = dyn_cast<mlir::cf::BranchOp>(terminator)) {
            if (branchOp.getDest() == block && argIdx < branchOp.getNumOperands()) {
              Value incoming = branchOp.getDestOperands()[argIdx];
              sigId = traceToSignal(incoming, depth + 1);
              if (sigId != 0)
                return sigId;
            }
          } else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
            if (condBrOp.getTrueDest() == block && argIdx < condBrOp.getNumTrueOperands()) {
              Value incoming = condBrOp.getTrueDestOperands()[argIdx];
              sigId = traceToSignal(incoming, depth + 1);
              if (sigId != 0)
                return sigId;
            }
            if (condBrOp.getFalseDest() == block && argIdx < condBrOp.getNumFalseOperands()) {
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
        LLVM_DEBUG(llvm::dbgs() << "  Warning: Could not find signal for "
                                    "observed value (type: "
                                << observed.getType() << ")\n");
      }
    }

    // Register the wait sensitivity with the scheduler
    if (!waitList.empty()) {
      scheduler.suspendProcessForEvents(procId, waitList);
      LLVM_DEBUG(llvm::dbgs() << "  Registered for " << waitList.size()
                              << " signal events\n");
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Warning: No signals found in observed list, process "
                    "will not be triggered by events!\n");
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

  // Check the cache first. If a value has been explicitly set (e.g., by
  // interpretProbe when the probe operation was executed), use that cached
  // value. This is important for patterns like posedge detection where we
  // need to compare old vs new signal values:
  //   %old = llhd.prb %sig   // executed before wait, cached
  //   llhd.wait ...
  //   %new = llhd.prb %sig   // executed after wait, gets fresh value
  //   %edge = comb.and %new, (not %old)  // needs OLD cached value for %old
  auto &valueMap = it->second.valueMap;
  auto valIt = valueMap.find(value);
  if (valIt != valueMap.end())
    return valIt->second;

  // For probe operations that are NOT in the cache, do a live re-read.
  // This handles the case where a probe result is used but the probe
  // operation itself was defined outside the process (e.g., at module level).
  if (auto probeOp = value.getDefiningOp<llhd::ProbeOp>()) {
    SignalId sigId = getSignalId(probeOp.getSignal());
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
    if (it != dynamicStrings.end() && it->second.first && it->second.second > 0) {
      return std::string(it->second.first, it->second.second);
    }

    // Fallback: try to interpret as direct pointer
    if (ptrVal != 0 && lenVal > 0 && lenVal < 1024) {
      const char *ptr = reinterpret_cast<const char *>(ptrVal);
      // Safety check - only dereference if it looks valid
      if (ptr) {
        return std::string(ptr, lenVal);
      }
    }

    return "<dynamic string>";
  }

  // Unknown format operation
  return "<unsupported format>";
}

LogicalResult LLHDProcessInterpreter::interpretTerminate(
    ProcessId procId, sim::TerminateOp terminateOp) {
  bool success = terminateOp.getSuccess();
  bool verbose = terminateOp.getVerbose();

  LLVM_DEBUG(llvm::dbgs() << "  Interpreting sim.terminate ("
                          << (success ? "success" : "failure") << ", "
                          << (verbose ? "verbose" : "quiet") << ")\n");

  // Mark termination requested
  terminationRequested = true;

  // Call the terminate callback if set
  if (terminateCallback) {
    terminateCallback(success, verbose);
  }

  // Mark the process as halted
  auto &state = processStates[procId];
  state.halted = true;

  // Terminate the process in the scheduler
  scheduler.terminateProcess(procId);

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Seq Dialect Operation Handlers
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::interpretSeqYield(ProcessId procId,
                                                         seq::YieldOp yieldOp) {
  LLVM_DEBUG(llvm::dbgs() << "  Interpreting seq.yield - terminating initial block\n");

  auto &state = processStates[procId];

  // seq.yield terminates the initial block - mark it as halted
  state.halted = true;

  // Terminate the process in the scheduler
  scheduler.terminateProcess(procId);

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
    state.halted = true;
    scheduler.terminateProcess(procId);
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
    // No signals found - treat as a single delta cycle wait
    // This prevents infinite loops when wait_event body has no detectable signals
    LLVM_DEBUG(llvm::dbgs() << "    Warning: No signals found in wait_event, "
                               "doing single delta wait\n");

    // Just schedule the process to run in the next active region
    // This will advance at least one delta cycle
    scheduler.scheduleProcess(procId, SchedulingRegion::Active);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LLVM Dialect Operation Handlers
//===----------------------------------------------------------------------===//

unsigned LLHDProcessInterpreter::getLLVMTypeSize(Type type) {
  // For LLVM pointer types, use 64 bits (8 bytes)
  if (isa<LLVM::LLVMPointerType>(type))
    return 8;

  // For LLVM struct types, sum the sizes of all elements
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
  auto &state = processStates[procId];

  // First check if this is a direct alloca result
  auto it = state.memoryBlocks.find(ptr);
  if (it != state.memoryBlocks.end())
    return &it->second;

  // Check if this is a GEP result - trace back to find the base pointer
  if (auto gepOp = ptr.getDefiningOp<LLVM::GEPOp>()) {
    return findMemoryBlock(procId, gepOp.getBase());
  }

  // Check if this is a bitcast - trace through
  if (auto bitcastOp = ptr.getDefiningOp<LLVM::BitcastOp>()) {
    return findMemoryBlock(procId, bitcastOp.getArg());
  }

  return nullptr;
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

  // Store the memory block
  state.memoryBlocks[allocaOp.getResult()] = std::move(block);

  // Assign a unique address to this pointer (for tracking purposes)
  uint64_t addr = state.nextMemoryAddress;
  state.nextMemoryAddress += totalSize;

  // Store the pointer value (the address)
  setValue(procId, allocaOp.getResult(), InterpretedValue(addr, 64));

  LLVM_DEBUG(llvm::dbgs() << "  llvm.alloca: allocated " << totalSize
                          << " bytes at address 0x" << llvm::format_hex(addr, 16)
                          << "\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMLoad(ProcessId procId,
                                                         LLVM::LoadOp loadOp) {
  // Get the pointer value
  InterpretedValue ptrVal = getValue(procId, loadOp.getAddr());
  Type resultType = loadOp.getType();
  unsigned bitWidth = getTypeWidth(resultType);
  unsigned loadSize = getLLVMTypeSize(resultType);

  // First try to find a local memory block (from alloca)
  MemoryBlock *block = findMemoryBlock(procId, loadOp.getAddr());
  uint64_t offset = 0;

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
  }

  if (!block) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.load: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    setValue(procId, loadOp.getResult(),
             InterpretedValue::makeX(bitWidth));
    return success();
  }

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

  // Read bytes from memory and construct the value (little-endian)
  uint64_t value = 0;
  for (unsigned i = 0; i < loadSize && i < 8; ++i) {
    value |= static_cast<uint64_t>(block->data[offset + i]) << (i * 8);
  }

  // For values larger than 64 bits, use APInt directly
  if (bitWidth > 64) {
    APInt apValue(bitWidth, 0);
    for (unsigned i = 0; i < loadSize; ++i) {
      APInt byteVal(bitWidth, block->data[offset + i]);
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
  // Get the pointer value first
  InterpretedValue ptrVal = getValue(procId, storeOp.getAddr());

  // Find the memory block for this pointer
  MemoryBlock *block = findMemoryBlock(procId, storeOp.getAddr());
  uint64_t offset = 0;

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
  }

  if (!block) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: no memory block found for pointer 0x"
                            << llvm::format_hex(ptrVal.isX() ? 0 : ptrVal.getUInt64(), 16) << "\n");
    return success(); // Don't fail, just skip the store
  }

  // Get the value to store
  InterpretedValue storeVal = getValue(procId, storeOp.getValue());
  unsigned storeSize = getLLVMTypeSize(storeOp.getValue().getType());

  if (offset + storeSize > block->size) {
    LLVM_DEBUG(llvm::dbgs() << "  llvm.store: out of bounds access\n");
    return success();
  }

  // Write bytes to memory (little-endian)
  if (!storeVal.isX()) {
    const APInt &apValue = storeVal.getAPInt();
    if (apValue.getBitWidth() > 64) {
      // Handle wide values using APInt operations
      for (unsigned i = 0; i < storeSize; ++i) {
        block->data[offset + i] =
            static_cast<uint8_t>(apValue.extractBits(8, i * 8).getZExtValue());
      }
    } else {
      uint64_t value = storeVal.getUInt64();
      for (unsigned i = 0; i < storeSize && i < 8; ++i) {
        block->data[offset + i] = static_cast<uint8_t>((value >> (i * 8)) & 0xFF);
      }
    }
    block->initialized = true;
  }

  LLVM_DEBUG(llvm::dbgs() << "  llvm.store: stored "
                          << (storeVal.isX() ? "X" : std::to_string(storeVal.getUInt64()))
                          << " (" << storeSize << " bytes) at offset " << offset << "\n");

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

  // Look up the function in the module
  auto &state = processStates[procId];
  Operation *parent = state.processOrInitialOp;
  while (parent && !isa<ModuleOp>(parent))
    parent = parent->getParentOp();

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
    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: function '" << calleeName
                            << "' not found\n");
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

    // Handle malloc - dynamic memory allocation for class instances
    if (calleeName == "malloc") {
      if (callOp.getNumOperands() >= 1 && callOp.getNumResults() >= 1) {
        InterpretedValue sizeArg = getValue(procId, callOp.getOperand(0));
        uint64_t size = sizeArg.isX() ? 256 : sizeArg.getUInt64();  // Default size if X

        // Allocate memory block for this allocation
        auto &procState = processStates[procId];
        uint64_t addr = procState.nextMemoryAddress;
        procState.nextMemoryAddress += size;

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

    LLVM_DEBUG(llvm::dbgs() << "  llvm.call: function '" << calleeName
                            << "' is external (no body)\n");
    for (Value result : callOp.getResults()) {
      setValue(procId, result,
               InterpretedValue::makeX(getTypeWidth(result.getType())));
    }
    return success();
  }

  // Gather argument values
  SmallVector<InterpretedValue, 4> args;
  for (Value arg : callOp.getArgOperands()) {
    args.push_back(getValue(procId, arg));
  }

  // Interpret the function body
  SmallVector<InterpretedValue, 2> results;
  if (failed(interpretLLVMFuncBody(procId, funcOp, args, results)))
    return failure();

  // Set the return values
  for (auto [result, retVal] : llvm::zip(callOp.getResults(), results)) {
    setValue(procId, result, retVal);
  }

  LLVM_DEBUG(llvm::dbgs() << "  llvm.call: called '" << calleeName << "'\n");

  return success();
}

LogicalResult LLHDProcessInterpreter::interpretLLVMFuncBody(
    ProcessId procId, LLVM::LLVMFuncOp funcOp, ArrayRef<InterpretedValue> args,
    SmallVectorImpl<InterpretedValue> &results) {
  auto &state = processStates[procId];

  // Map arguments to block arguments
  Block &entryBlock = funcOp.getBody().front();
  for (auto [blockArg, argVal] : llvm::zip(entryBlock.getArguments(), args)) {
    state.valueMap[blockArg] = argVal;
  }

  // Execute the function body
  Block *currentBlock = &entryBlock;
  while (currentBlock) {
    for (Operation &op : *currentBlock) {
      // Handle return
      if (auto returnOp = dyn_cast<LLVM::ReturnOp>(&op)) {
        for (Value retVal : returnOp.getOperands()) {
          results.push_back(getValue(procId, retVal));
        }
        return success();
      }

      // Handle branch
      if (auto branchOp = dyn_cast<LLVM::BrOp>(&op)) {
        currentBlock = branchOp.getDest();
        for (auto [blockArg, operand] :
             llvm::zip(currentBlock->getArguments(), branchOp.getDestOperands())) {
          state.valueMap[blockArg] = getValue(procId, operand);
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
            state.valueMap[blockArg] = getValue(procId, operand);
          }
        } else {
          currentBlock = condBrOp.getFalseDest();
          for (auto [blockArg, operand] :
               llvm::zip(currentBlock->getArguments(),
                        condBrOp.getFalseDestOperands())) {
            state.valueMap[blockArg] = getValue(procId, operand);
          }
        }
        break;
      }

      // Interpret other operations
      if (failed(interpretOperation(procId, &op)))
        return failure();
    }

    // If we didn't branch, we're done
    if (currentBlock == &entryBlock ||
        !currentBlock->back().hasTrait<OpTrait::IsTerminator>())
      break;
  }

  // If no return was encountered, return nothing
  return success();
}

//===----------------------------------------------------------------------===//
// Global Variable and VTable Support
//===----------------------------------------------------------------------===//

LogicalResult LLHDProcessInterpreter::initializeGlobals() {
  if (!rootModule)
    return success();

  LLVM_DEBUG(llvm::dbgs() << "LLHDProcessInterpreter: Initializing globals\n");

  // Find all LLVM global operations in the module
  rootModule.walk([&](LLVM::GlobalOp globalOp) {
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

    // Create memory block
    MemoryBlock block(size, 64);

    // Check the initializer attribute
    // Most globals are initialized with #llvm.zero or a constant value
    // The MemoryBlock constructor already initializes data to 0, so we just
    // need to mark it as initialized.
    if (globalOp.getValueOrNull()) {
      // Has an initializer - mark as initialized
      // The data is already zeroed from the constructor, which is correct
      // for #llvm.zero initializers
      block.initialized = true;
      LLVM_DEBUG(llvm::dbgs() << "    Initialized to zero\n");
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
  });

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

  // Collect all constructor entries with their priorities
  SmallVector<std::pair<int32_t, StringRef>, 4> ctorEntries;

  rootModule.walk([&](LLVM::GlobalCtorsOp ctorsOp) {
    ArrayAttr ctors = ctorsOp.getCtors();
    ArrayAttr priorities = ctorsOp.getPriorities();

    for (auto [ctorAttr, priorityAttr] : llvm::zip(ctors, priorities)) {
      auto ctorRef = cast<FlatSymbolRefAttr>(ctorAttr);
      auto priority = cast<IntegerAttr>(priorityAttr).getInt();
      ctorEntries.emplace_back(priority, ctorRef.getValue());
      LLVM_DEBUG(llvm::dbgs() << "  Found constructor: " << ctorRef.getValue()
                              << " (priority " << priority << ")\n");
    }
  });

  if (ctorEntries.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No global constructors found\n");
    return success();
  }

  // Sort by priority (lower priority values execute first)
  llvm::sort(ctorEntries,
             [](const auto &a, const auto &b) { return a.first < b.first; });

  // Create a temporary process state for executing constructors
  // We use process ID 0 as a special "global init" process
  ProcessExecutionState tempState;
  ProcessId tempProcId = 0;

  // Check if we already have a process with ID 0; if so, use a different ID
  if (processStates.count(tempProcId)) {
    // Find an unused process ID
    tempProcId = processStates.size() + 1000;
  }
  processStates[tempProcId] = std::move(tempState);

  // Execute each constructor in priority order
  for (auto &[priority, ctorName] : ctorEntries) {
    LLVM_DEBUG(llvm::dbgs() << "  Calling constructor: " << ctorName
                            << " (priority " << priority << ")\n");

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
