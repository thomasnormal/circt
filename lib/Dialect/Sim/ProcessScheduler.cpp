//===- ProcessScheduler.cpp - Process scheduling for simulation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the process scheduler infrastructure for event-driven
// simulation. It manages concurrent processes with sensitivity lists,
// edge detection, and delta cycle semantics following IEEE 1800.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>

using llvm::StringRef;

#define DEBUG_TYPE "sim-process-scheduler"

using namespace circt;
using namespace circt::sim;

namespace {
SignalValue normalizeSignalValueWidth(const SignalValue &value,
                                      uint32_t width) {
  if (value.isUnknown())
    return SignalValue::makeX(width);
  const auto &apInt = value.getAPInt();
  if (apInt.getBitWidth() == width)
    return value;
  return SignalValue(apInt.zextOrTrunc(width));
}

bool isFourStateUnknown(const SignalValue &value) {
  if (value.isUnknown())
    return true;
  uint32_t width = value.getWidth();
  if (width < 2 || (width % 2) != 0)
    return false;
  uint32_t halfWidth = width / 2;
  const auto &bits = value.getAPInt();
  for (uint32_t i = 0; i < halfWidth; ++i) {
    if (bits[i])
      return true;
  }
  return false;
}

bool getFourStateValueLSB(const SignalValue &value) {
  uint32_t width = value.getWidth();
  if (width < 2 || (width % 2) != 0)
    return value.getLSB();
  uint32_t halfWidth = width / 2;
  return value.getAPInt()[halfWidth];
}

EdgeType detectEdgeWithEncoding(const SignalValue &oldVal,
                                const SignalValue &newVal,
                                SignalEncoding encoding) {
  if (encoding != SignalEncoding::FourStateStruct)
    return SignalValue::detectEdge(oldVal, newVal);

  if (!oldVal.isUnknown() && !newVal.isUnknown() &&
      oldVal.getAPInt() == newVal.getAPInt())
    return EdgeType::None;

  bool oldIsX = isFourStateUnknown(oldVal);
  bool newIsX = isFourStateUnknown(newVal);

  if (oldIsX || newIsX) {
    if (oldIsX && newIsX)
      return EdgeType::None;
    if (oldIsX) {
      bool newBit = getFourStateValueLSB(newVal);
      return newBit ? EdgeType::Posedge : EdgeType::Negedge;
    }
    return EdgeType::AnyEdge;
  }

  bool oldBit = getFourStateValueLSB(oldVal);
  bool newBit = getFourStateValueLSB(newVal);

  if (!oldBit && newBit)
    return EdgeType::Posedge;
  if (oldBit && !newBit)
    return EdgeType::Negedge;
  return EdgeType::AnyEdge;
}

bool sensitivityTriggered(const SensitivityList &list, SignalId signalId,
                          EdgeType actualEdge) {
  if (actualEdge == EdgeType::None)
    return false;
  for (const auto &entry : list.getEntries()) {
    if (entry.signalId != signalId)
      continue;
    switch (entry.edge) {
    case EdgeType::None:
      return true;
    case EdgeType::AnyEdge:
      return true;
    case EdgeType::Posedge:
      if (actualEdge == EdgeType::Posedge)
        return true;
      break;
    case EdgeType::Negedge:
      if (actualEdge == EdgeType::Negedge)
        return true;
      break;
    }
  }
  return false;
}

bool isVpiOwnershipSuppressionEnabled() {
  // VPI ownership suppression is ON by default. When enabled, VPI-written
  // signals are protected from stale HDL drives (e.g., seq.firreg presets,
  // module-level llhd.drv) until ownership is cleared at the next time step.
  // The vpiOwnedSignals set is empty when VPI is not loaded, so this has
  // zero overhead for non-VPI simulations.
  // Set CIRCT_SIM_DISABLE_VPI_OWNERSHIP_SUPPRESS=1 to disable.
  static bool disabled = []() {
    const char *env = std::getenv("CIRCT_SIM_DISABLE_VPI_OWNERSHIP_SUPPRESS");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  return !disabled;
}
} // namespace

// Static member initialization
SignalValue ProcessScheduler::unknownSignal = SignalValue::makeX();

//===----------------------------------------------------------------------===//
// ProcessScheduler Implementation
//===----------------------------------------------------------------------===//

ProcessScheduler::ProcessScheduler(Config config)
    : config(config), eventScheduler(std::make_unique<EventScheduler>()) {
  LLVM_DEBUG(llvm::dbgs() << "ProcessScheduler created with maxDeltaCycles="
                          << config.maxDeltaCycles << "\n");
}

ProcessScheduler::~ProcessScheduler() = default;

//===----------------------------------------------------------------------===//
// Process Management
//===----------------------------------------------------------------------===//

ProcessId ProcessScheduler::nextProcessId() { return nextProcId++; }

SignalId ProcessScheduler::nextSignalId() { return nextSigId++; }

ProcessId ProcessScheduler::registerProcess(const std::string &name,
                                            Process::ExecuteCallback callback) {
  if (processes.size() >= config.maxProcesses) {
    LLVM_DEBUG(llvm::dbgs() << "Maximum process count reached\n");
    return InvalidProcessId;
  }

  ProcessId id = nextProcessId();
  auto process = std::make_unique<Process>(id, name, std::move(callback));
  Process *rawPtr = process.get();
  processes[id] = std::move(process);

  // Maintain flat vector for O(1) lookup.
  if (id >= processVec.size())
    processVec.resize(id + 1, nullptr);
  processVec[id] = rawPtr;

  ++stats.processesRegistered;
  LLVM_DEBUG(llvm::dbgs() << "Registered process '" << name << "' with ID "
                          << id << "\n");

  return id;
}

void ProcessScheduler::unregisterProcess(ProcessId id) {
  auto it = processes.find(id);
  if (it == processes.end())
    return;

  Process *proc = it->second.get();

  // Remove from signal mappings (compare by Process*).
  for (auto &[signalId, procList] : signalToProcesses) {
    procList.erase(std::remove(procList.begin(), procList.end(), proc),
                   procList.end());
  }

  // Remove from intrusive ready queues (must unlink before delete).
  if (proc->inReadyQueue) {
    for (auto &queue : readyQueues)
      queue.remove(proc);
  }

  // Clear flat vector entry before erasing ownership.
  if (id < processVec.size())
    processVec[id] = nullptr;

  processes.erase(it);
  LLVM_DEBUG(llvm::dbgs() << "Unregistered process ID " << id << "\n");
}

Process *ProcessScheduler::getProcess(ProcessId id) {
  return id < processVec.size() ? processVec[id] : nullptr;
}

const Process *ProcessScheduler::getProcess(ProcessId id) const {
  return id < processVec.size() ? processVec[id] : nullptr;
}

//===----------------------------------------------------------------------===//
// Sensitivity Management
//===----------------------------------------------------------------------===//

void ProcessScheduler::setSensitivity(ProcessId id,
                                      const SensitivityList &sensitivity) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  // Clear existing mappings
  clearSensitivity(id);

  // Set new sensitivity list
  proc->getSensitivityList() = sensitivity;

  // Update signal-to-process mappings (store Process* for zero-lookup dispatch).
  for (const auto &entry : sensitivity.getEntries()) {
    signalToProcesses[entry.signalId].push_back(proc);
  }

  LLVM_DEBUG(llvm::dbgs() << "Set sensitivity for process " << id << " with "
                          << sensitivity.size() << " entries\n");
}

void ProcessScheduler::addSensitivity(ProcessId id, SignalId signalId,
                                      EdgeType edge) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  proc->getSensitivityList().addEdge(signalId, edge);
  signalToProcesses[signalId].push_back(proc);

  LLVM_DEBUG(llvm::dbgs() << "Added sensitivity for process " << id
                          << " to signal " << signalId << " edge="
                          << getEdgeTypeName(edge) << "\n");
}

void ProcessScheduler::clearSensitivity(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  // Remove from signal mappings (compare by Process*).
  for (const auto &entry : proc->getSensitivityList().getEntries()) {
    auto &procList = signalToProcesses[entry.signalId];
    procList.erase(std::remove(procList.begin(), procList.end(), proc),
                   procList.end());
  }

  proc->getSensitivityList().clear();
}

SignalId ProcessScheduler::registerSignal(const std::string &name,
                                          uint32_t width) {
  return registerSignal(name, width, SignalEncoding::Unknown);
}

SignalId ProcessScheduler::registerSignal(const std::string &name,
                                          uint32_t width,
                                          SignalEncoding encoding) {
  SignalId id = nextSignalId();
  if (id >= signalStates.size())
    signalStates.resize(id + 1);
  signalStates[id] = SignalState(width);
  signalNames[id] = name;
  signalEncodings[id] = encoding;

  // Maintain direct signal memory for narrow signals.
  if (id >= signalMemory.size()) {
    signalMemory.resize(id + 1, 0);
    signalIsDirect.resize(id + 1, false);
  }
  if (width > 0 && width <= 64) {
    signalIsDirect[id] = true;
    signalMemory[id] = 0;
  }

  LLVM_DEBUG(llvm::dbgs() << "Registered signal '" << name << "' with ID " << id
                          << " width=" << width << "\n");

  return id;
}

SignalId ProcessScheduler::registerSignal(const std::string &name,
                                          uint32_t width,
                                          SignalEncoding encoding,
                                          SignalResolution resolution) {
  SignalId id = nextSignalId();
  if (id >= signalStates.size())
    signalStates.resize(id + 1);
  signalStates[id] = SignalState(width, resolution);
  signalNames[id] = name;
  signalEncodings[id] = encoding;

  // Maintain direct signal memory for narrow signals.
  if (id >= signalMemory.size()) {
    signalMemory.resize(id + 1, 0);
    signalIsDirect.resize(id + 1, false);
  }
  if (width > 0 && width <= 64) {
    signalIsDirect[id] = true;
    signalMemory[id] = 0;
  }

  LLVM_DEBUG(llvm::dbgs() << "Registered signal '" << name << "' with ID " << id
                          << " width=" << width
                          << " resolution="
                          << (resolution == SignalResolution::WiredAnd ? "wand"
                              : resolution == SignalResolution::WiredOr ? "wor"
                              : "default")
                          << "\n");

  return id;
}

SignalEncoding ProcessScheduler::getSignalEncoding(SignalId signalId) const {
  auto it = signalEncodings.find(signalId);
  if (it == signalEncodings.end())
    return SignalEncoding::Unknown;
  return it->second;
}

void ProcessScheduler::registerSignalAlias(SignalId signalId,
                                           const std::string &alias) {
  signalAliases[alias] = signalId;
}

void ProcessScheduler::registerInstanceScope(const std::string &instancePath) {
  instanceScopes.insert(instancePath);
}

void ProcessScheduler::setSignalLogicalWidth(SignalId signalId,
                                              uint32_t logicalWidth) {
  signalLogicalWidths[signalId] = logicalWidth;
}

uint32_t ProcessScheduler::getSignalLogicalWidth(SignalId signalId) const {
  auto it = signalLogicalWidths.find(signalId);
  if (it == signalLogicalWidths.end())
    return 0;
  return it->second;
}

void ProcessScheduler::setSignalArrayInfo(SignalId signalId,
                                          const SignalArrayInfo &info) {
  signalArrayInfos[signalId] = info;
}

const ProcessScheduler::SignalArrayInfo *
ProcessScheduler::getSignalArrayInfo(SignalId signalId) const {
  auto it = signalArrayInfos.find(signalId);
  if (it == signalArrayInfos.end())
    return nullptr;
  return &it->second;
}

void ProcessScheduler::setSignalStructFields(
    SignalId signalId, std::vector<SignalStructFieldInfo> fields) {
  signalStructFields[signalId] = std::move(fields);
}

const std::vector<ProcessScheduler::SignalStructFieldInfo> *
ProcessScheduler::getSignalStructFields(SignalId signalId) const {
  auto it = signalStructFields.find(signalId);
  if (it == signalStructFields.end())
    return nullptr;
  return &it->second;
}

void ProcessScheduler::setSignalResolution(SignalId signalId,
                                           SignalResolution resolution) {
  if (signalId < signalStates.size())
    signalStates[signalId].setResolution(resolution);
}

void ProcessScheduler::setMaxDeltaCycles(size_t maxDeltaCycles) {
  config.maxDeltaCycles = maxDeltaCycles;
}

void ProcessScheduler::updateSignal(SignalId signalId,
                                    const SignalValue &newValue) {
  // Suppress non-VPI drives to VPI-owned signals. VPI putValue marks signals
  // as owned after its direct updateSignal call, so this guard only blocks
  // subsequent stale drives (e.g., init-time EventScheduler events).
  if (isVpiOwnershipSuppressionEnabled() && vpiOwnedSignals.count(signalId))
    return;

  if (signalId >= signalStates.size()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: updating unknown signal " << signalId
                            << "\n");
    return;
  }
  auto &sigState = signalStates[signalId];

  uint32_t signalWidth = sigState.getCurrentValue().getWidth();
  SignalValue normalizedValue =
      normalizeSignalValueWidth(newValue, signalWidth);
  SignalValue oldValue = sigState.getCurrentValue();
  (void)sigState.updateValue(normalizedValue);

  // Keep direct signal memory in sync for narrow signals.
  if (signalId < signalIsDirect.size() && signalIsDirect[signalId])
    signalMemory[signalId] = normalizedValue.getValue();

  static bool traceUpdates = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_SIGNAL_UPDATES");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static llvm::StringRef traceFilter = []() -> llvm::StringRef {
    const char *env = std::getenv("CIRCT_SIM_TRACE_SIGNAL_UPDATES_FILTER");
    return env ? llvm::StringRef(env) : llvm::StringRef();
  }();
  if (traceUpdates) {
    auto formatSignalValue = [](const SignalValue &sv) -> std::string {
      if (sv.isUnknown())
        return "X";
      llvm::SmallString<64> bits;
      sv.getAPInt().toString(bits, 16, false);
      return std::string(bits);
    };
    auto nameIt = signalNames.find(signalId);
    llvm::StringRef sigName = nameIt != signalNames.end()
                                  ? llvm::StringRef(nameIt->second)
                                  : llvm::StringRef("<unknown>");
    if (traceFilter.empty() || sigName.contains(traceFilter)) {
      SimTime now = getCurrentTime();
      llvm::errs() << "[SIG-UPD] t=" << now.realTime << " d=" << now.deltaStep
                   << " sig=" << signalId << " (" << sigName << ")"
                   << " old=0x" << formatSignalValue(oldValue)
                   << " new=0x" << formatSignalValue(normalizedValue) << "\n";
    }
  }

  EdgeType edge =
      detectEdgeWithEncoding(oldValue, normalizedValue,
                             getSignalEncoding(signalId));

  ++stats.signalUpdates;

  if (edge != EdgeType::None) {
    ++stats.edgesDetected;
    LLVM_DEBUG(llvm::dbgs() << "Signal " << signalId << " changed: edge="
                            << getEdgeTypeName(edge) << "\n");
    if (signalChangeCallback)
      signalChangeCallback(signalId, normalizedValue);
    triggerSensitiveProcesses(signalId, oldValue, normalizedValue);
    recordSignalChange(signalId);
  }
}

void ProcessScheduler::updateSignalFast(SignalId signalId, uint64_t rawValue,
                                        uint32_t width) {
  auto &sigState = signalStates[signalId];
  ++stats.signalUpdates;

  // Keep direct signal memory in sync for narrow signals.
  if (signalId < signalIsDirect.size() && signalIsDirect[signalId])
    signalMemory[signalId] = rawValue;

  // In-place update: avoids APInt construction entirely. updateValueFast
  // copies current→previous and writes rawValue directly to APInt storage.
  if (sigState.updateValueFast(rawValue)) {
    ++stats.edgesDetected;
    const SignalValue &newVal = sigState.getCurrentValue();
    const SignalValue &oldVal = sigState.getPreviousValue();
    if (signalChangeCallback)
      signalChangeCallback(signalId, newVal);
    triggerSensitiveProcesses(signalId, oldVal, newVal);
    recordSignalChange(signalId);
  }
}

void ProcessScheduler::scheduleProcessDirect(ProcessId id, Process *proc) {
  auto &queue = readyQueues[static_cast<size_t>(SchedulingRegion::Active)];
  // Fast path: skip dedup check — clock processes are single-waiter,
  // so they're never already in the queue when waking from a timed wait.
  queue.push_back(proc);
  proc->clearWaiting();
  proc->inReadyQueue = true;
  proc->setState(ProcessState::Ready);
  recordTriggerTime(id);
}

void ProcessScheduler::updateSignalWithStrength(SignalId signalId,
                                                uint64_t driverId,
                                                const SignalValue &newValue,
                                                DriveStrength strength0,
                                                DriveStrength strength1) {
  // Suppress non-VPI drives to VPI-owned signals.
  if (isVpiOwnershipSuppressionEnabled() && vpiOwnedSignals.count(signalId))
    return;

  if (signalId >= signalStates.size()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: updating unknown signal " << signalId
                            << "\n");
    return;
  }
  auto &sigState = signalStates[signalId];

  uint32_t signalWidth = sigState.getCurrentValue().getWidth();
  SignalValue normalizedValue =
      normalizeSignalValueWidth(newValue, signalWidth);
  SignalValue oldValue = sigState.getCurrentValue();
  SignalEncoding encoding = getSignalEncoding(signalId);

  // Add/update the driver with its strength information
  sigState.addOrUpdateDriver(driverId, normalizedValue, strength0, strength1);

  // Resolve all drivers to get the final signal value
  SignalValue resolvedValue = sigState.resolveDrivers();
  SignalValue normalizedResolved =
      normalizeSignalValueWidth(resolvedValue, signalWidth);

  static bool traceStrength = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_SIGNAL_STRENGTH");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  static llvm::StringRef traceFilter = []() -> llvm::StringRef {
    const char *env = std::getenv("CIRCT_SIM_TRACE_SIGNAL_STRENGTH_FILTER");
    return env ? llvm::StringRef(env) : llvm::StringRef();
  }();
  if (traceStrength) {
    auto formatSignalValue = [](const SignalValue &sv) -> std::string {
      if (sv.isUnknown())
        return "X";
      llvm::SmallString<64> bits;
      sv.getAPInt().toString(bits, 16, false);
      return std::string(bits);
    };
    auto nameIt = signalNames.find(signalId);
    llvm::StringRef sigName = nameIt != signalNames.end()
                                  ? llvm::StringRef(nameIt->second)
                                  : llvm::StringRef("<unknown>");
    if (traceFilter.empty() || sigName.contains(traceFilter)) {
      SimTime now = getCurrentTime();
      llvm::errs() << "[SIG-DRV] t=" << now.realTime
                   << " d=" << now.deltaStep << " sig=" << signalId
                   << " (" << sigName << ")"
                   << " drv=" << driverId
                   << " val=0x" << formatSignalValue(normalizedValue)
                   << " s=(" << getDriveStrengthName(strength0) << ","
                   << getDriveStrengthName(strength1) << ")"
                   << " resolved=0x" << formatSignalValue(normalizedResolved)
                   << "\n";
    }
  }

  // Update the signal with the resolved value
  (void)sigState.updateValue(normalizedResolved);

  // Keep direct signal memory in sync for narrow signals.
  if (signalId < signalIsDirect.size() && signalIsDirect[signalId])
    signalMemory[signalId] = normalizedResolved.getValue();

  EdgeType edge =
      detectEdgeWithEncoding(oldValue, normalizedResolved, encoding);

  ++stats.signalUpdates;

  LLVM_DEBUG({
    llvm::dbgs() << "Signal " << signalId << " driver " << driverId
                 << " updated: value=";
    if (normalizedValue.isUnknown()) {
      llvm::dbgs() << "X";
    } else {
      normalizedValue.getAPInt().print(llvm::dbgs(), /*isSigned=*/false);
    }
    llvm::dbgs() << " strength=(" << getDriveStrengthName(strength0) << ", "
                 << getDriveStrengthName(strength1) << ")";
    if (sigState.hasMultipleDrivers()) {
      llvm::dbgs() << " resolved=";
      if (normalizedResolved.isUnknown()) {
        llvm::dbgs() << "X";
      } else {
        normalizedResolved.getAPInt().print(llvm::dbgs(), /*isSigned=*/false);
      }
    }
    llvm::dbgs() << "\n";
  });

  if (edge != EdgeType::None) {
    ++stats.edgesDetected;
    LLVM_DEBUG(llvm::dbgs() << "Signal " << signalId << " changed: edge="
                            << getEdgeTypeName(edge) << "\n");
    if (signalChangeCallback)
      signalChangeCallback(signalId, normalizedResolved);
    triggerSensitiveProcesses(signalId, oldValue, normalizedResolved);
    recordSignalChange(signalId);
  }
}

void ProcessScheduler::recordSignalChange(SignalId signalId) {
  if (signalId < signalsChangedThisDeltaBits.size()) {
    if (!signalsChangedThisDeltaBits[signalId]) {
      signalsChangedThisDeltaBits[signalId] = true;
      signalsChangedThisDelta.push_back(signalId);
    }
  } else {
    // Signal ID beyond bitvector size — always record (safe fallback).
    signalsChangedThisDelta.push_back(signalId);
  }
}

void ProcessScheduler::recordTriggerSignal(ProcessId id, SignalId signalId) {
  pendingTriggerSignals[id] = signalId;
}

void ProcessScheduler::recordTriggerTime(ProcessId id) {
  pendingTriggerTimes.insert(id);
}

void ProcessScheduler::dumpLastDeltaSignals(llvm::raw_ostream &os) const {
  os << "[circt-sim] Signals changed in last delta:\n";
  if (lastDeltaSignals.empty()) {
    os << "  (none)\n";
    return;
  }

  for (SignalId signalId : lastDeltaSignals) {
    auto nameIt = signalNames.find(signalId);
    llvm::StringRef name = nameIt == signalNames.end()
                               ? llvm::StringRef("<unknown>")
                               : llvm::StringRef(nameIt->second);
    if (signalId >= signalStates.size()) {
      os << "  " << name << " (id=" << signalId << "): <missing>\n";
      continue;
    }

    const SignalValue &value = signalStates[signalId].getCurrentValue();
    if (value.isUnknown() || value.isFourStateX()) {
      os << "  " << name << " (id=" << signalId << ", w=" << value.getWidth()
         << "): X\n";
      continue;
    }

    llvm::SmallString<32> buf;
    value.getAPInt().toString(buf, 16, false);
    os << "  " << name << " (id=" << signalId << ", w=" << value.getWidth()
       << "): 0x" << buf << "\n";
  }
}

void ProcessScheduler::dumpLastDeltaProcesses(llvm::raw_ostream &os) const {
  os << "[circt-sim] Processes executed in last delta:\n";
  if (lastDeltaProcesses.empty()) {
    os << "  (none)\n";
    return;
  }

  for (ProcessId procId : lastDeltaProcesses) {
    Process *proc = procId < processVec.size() ? processVec[procId] : nullptr;
    if (!proc) {
      os << "  proc " << procId << ": <missing>\n";
      continue;
    }
    llvm::StringRef name = proc->getName();
    os << "  proc " << procId << " '" << name << "' state="
       << getProcessStateName(proc->getState());
    if (proc->isCombinational())
      os << " comb";

    auto trigIt = lastDeltaTriggerSignals.find(procId);
    if (trigIt != lastDeltaTriggerSignals.end()) {
      SignalId signalId = trigIt->second;
      os << " trigger=signal(" << signalId;
      auto sigIt = signalNames.find(signalId);
      if (sigIt != signalNames.end())
        os << ":" << sigIt->second;
      os << ")";
    } else if (lastDeltaTriggerTimes.count(procId)) {
      os << " trigger=time";
    }

    if (name.consume_front("cont_assign_")) {
      unsigned signalId = 0;
      if (!name.getAsInteger(10, signalId)) {
        auto sigIt = signalNames.find(signalId);
        if (sigIt != signalNames.end())
          os << " signal=" << sigIt->second;
      }
    }
    os << "\n";
  }
}

const SignalValue &ProcessScheduler::getSignalValue(SignalId signalId) const {
  if (signalId >= signalStates.size())
    return unknownSignal;
  return signalStates[signalId].getCurrentValue();
}

const SignalValue &
ProcessScheduler::getSignalPreviousValue(SignalId signalId) const {
  if (signalId >= signalStates.size())
    return unknownSignal;
  return signalStates[signalId].getPreviousValue();
}

bool ProcessScheduler::isAbortRequested() const {
  return shouldAbortCallback && shouldAbortCallback();
}

void ProcessScheduler::triggerSensitiveProcesses(SignalId signalId,
                                                 const SignalValue &oldVal,
                                                 const SignalValue &newVal) {
  static bool traceI3CSignal31 = []() {
    const char *env = std::getenv("CIRCT_SIM_TRACE_I3C_SIGNAL31");
    return env && env[0] != '\0' && env[0] != '0';
  }();
  auto it = signalToProcesses.find(signalId);
  if (it == signalToProcesses.end())
    return;

  EdgeType actualEdge =
      detectEdgeWithEncoding(oldVal, newVal, getSignalEncoding(signalId));
  if (actualEdge == EdgeType::None)
    return;

  bool traceI3CSignal = traceI3CSignal31 && (signalId == 31 || signalId == 45);
  if (traceI3CSignal) {
    llvm::errs() << "[I3C-SIG31] edge=" << getEdgeTypeName(actualEdge)
                 << " waiters=" << it->second.size() << " [";
    for (size_t idx = 0; idx < it->second.size(); ++idx) {
      if (idx)
        llvm::errs() << ",";
      llvm::errs() << it->second[idx]->getId();
    }
    llvm::errs() << "]\n";
  }

  // Iterate Process* directly — no getProcess() hash lookup per iteration.
  for (Process *proc : it->second) {
    if (!proc)
      continue;
    ProcessId procId = proc->getId();
    if (traceI3CSignal) {
      llvm::errs() << "[I3C-SIG31] check proc=" << procId
                   << " state=" << getProcessStateName(proc->getState())
                   << "\n";
    }

    // Check if process state allows triggering
    ProcessState state = proc->getState();
    if (state != ProcessState::Suspended &&
        state != ProcessState::Waiting &&
        state != ProcessState::Ready)
      continue;

    // For waiting processes, check the waiting sensitivity
    if (state == ProcessState::Waiting) {
      if (sensitivityTriggered(proc->getWaitingSensitivity(), signalId,
                               actualEdge)) {
        proc->clearWaiting();
        recordTriggerSignal(procId, signalId);
        scheduleProcess(procId, proc->getPreferredRegion());
        if (traceI3CSignal)
          llvm::errs() << "[I3C-SIG31] trigger waiting proc=" << procId
                       << "\n";
        LLVM_DEBUG(llvm::dbgs()
                   << "Process " << procId << " triggered from waiting state\n");
      }
      continue;
    }

    // For suspended/ready processes, check the main sensitivity list
    if (sensitivityTriggered(proc->getSensitivityList(), signalId,
                             actualEdge)) {
      recordTriggerSignal(procId, signalId);
      scheduleProcess(procId, proc->getPreferredRegion());
      if (traceI3CSignal)
        llvm::errs() << "[I3C-SIG31] trigger sens proc=" << procId << "\n";
      LLVM_DEBUG(llvm::dbgs() << "Process " << procId << " triggered\n");
    }
  }
}

//===----------------------------------------------------------------------===//
// Process Execution Control
//===----------------------------------------------------------------------===//

void ProcessScheduler::scheduleProcess(ProcessId id, SchedulingRegion region) {
  // O(1) lookup via flat vector instead of DenseMap hash.
  Process *proc = id < processVec.size() ? processVec[id] : nullptr;
  if (!proc) {
    LLVM_DEBUG(llvm::dbgs() << "scheduleProcess(" << id << "): process not found!\n");
    return;
  }

  if (proc->getState() == ProcessState::Terminated) {
    LLVM_DEBUG(llvm::dbgs() << "scheduleProcess(" << id << "): process terminated, skipping\n");
    return;
  }

  // Skip if already in a ready queue — O(1) flag check.
  if (proc->inReadyQueue) {
    LLVM_DEBUG(llvm::dbgs() << "Process " << id << " already in queue\n");
    return;
  }

  auto &queue = readyQueues[static_cast<size_t>(region)];
  queue.push_back(proc);
  proc->inReadyQueue = true;
  proc->setState(ProcessState::Ready);
  LLVM_DEBUG(llvm::dbgs() << "Scheduled process " << id << " ('" << proc->getName()
               << "') in region " << getSchedulingRegionName(region) << "\n");
}

void ProcessScheduler::suspendProcess(ProcessId id, const SimTime &resumeTime) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  proc->setState(ProcessState::Suspended);
  proc->setResumeTime(resumeTime);

  // Schedule a wake-up event
  eventScheduler->schedule(
      resumeTime, proc->getPreferredRegion(),
      Event([this, id]() { resumeProcess(id); }));

  LLVM_DEBUG(llvm::dbgs() << "Suspended process " << id
                          << " until time=" << resumeTime.realTime << "\n");
}

void ProcessScheduler::suspendProcessForEvents(ProcessId id,
                                               const SensitivityList &waitList) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  proc->setWaitingFor(waitList);

  // Also update the main sensitivity list so it persists across wake/sleep cycles.
  // This is critical for LLHD processes that use llhd.wait - if a process wakes
  // but doesn't re-execute to its next wait (due to control flow or errors),
  // it would otherwise become orphaned in Suspended state with no sensitivity.
  proc->getSensitivityList() = waitList;

  // Ensure the signals in the wait list are mapped to this process.
  // Optimization: use a per-process set of registered signal IDs to avoid
  // redundant O(N) linear scans in signalToProcesses on every cycle.
  // RTL always blocks re-suspend on the same signals every cycle, making
  // the original std::find always return true after the first registration.
  for (const auto &entry : waitList.getEntries()) {
    if (proc->registeredSignals.insert(entry.signalId).second) {
      // First time this process registers for this signal.
      signalToProcesses[entry.signalId].push_back(proc);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Process " << id << " waiting for "
                          << waitList.size() << " events\n");
}

void ProcessScheduler::resuspendProcessFast(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;
  // Set state back to Waiting without rebuilding the sensitivity list.
  // The waiting sensitivity and signal-to-process mappings are unchanged
  // from the previous suspend (bytecode processes always wait on the
  // same signals).
  proc->setState(ProcessState::Waiting);
}

void ProcessScheduler::queueSignalUpdateFast(SignalId signalId, uint64_t value,
                                              uint32_t width) {
  pendingFastUpdates.push_back({signalId, value, width});
}

void ProcessScheduler::flushPendingFastUpdates() {
  if (pendingFastUpdates.empty())
    return;
  // Apply all deferred signal updates. This calls updateSignalFast which
  // triggers sensitive processes and adds them to ready queues.
  for (auto &upd : pendingFastUpdates)
    updateSignalFast(upd.signalId, upd.value, upd.width);
  pendingFastUpdates.clear();
}

size_t ProcessScheduler::registerClockDomain(ProcessId processId,
                                              SignalId signalId,
                                              uint64_t toggleMask,
                                              uint64_t halfPeriodFs,
                                              uint64_t firstWakeFs,
                                              uint32_t signalWidth) {
  size_t idx = clockDomains.size();
  clockDomains.push_back(
      {processId, signalId, toggleMask, halfPeriodFs, firstWakeFs, signalWidth});
  processToClockDomain[processId] = idx;
  LLVM_DEBUG(llvm::dbgs() << "Registered clock domain " << idx << " for process "
                          << processId << " signal " << signalId
                          << " period=" << (halfPeriodFs * 2) << "fs\n");
  return idx;
}

bool ProcessScheduler::isClockDomainProcess(ProcessId id) const {
  return processToClockDomain.count(id);
}

bool ProcessScheduler::advanceClockDomains(uint64_t targetTimeFs) {
  bool didWork = false;
  for (auto &cd : clockDomains) {
    if (!cd.active)
      continue;
    if (cd.nextWakeFs > targetTimeFs)
      continue;

    // Toggle the clock signal directly — no Event construction, no TimeWheel.
    uint64_t oldVal = readSignalValueFast(cd.signalId);
    uint64_t newVal = oldVal ^ cd.toggleMask;

    // Update signal. updateSignalFast triggers sensitive processes via
    // the sensitivity list callback, waking any @(posedge clk) waiters.
    updateSignalFast(cd.signalId, newVal, cd.signalWidth);

    // Advance to next half-period.
    cd.nextWakeFs += cd.halfPeriodFs;
    didWork = true;

    // Process one clock domain edge per call so processes can execute
    // between edges (delta cycles run at each edge).
  }
  return didWork;
}

void ProcessScheduler::terminateProcess(ProcessId id) {
  Process *proc = id < processVec.size() ? processVec[id] : nullptr;
  if (!proc)
    return;

  ProcessState oldState = proc->getState();
  proc->setState(ProcessState::Terminated);

  // Decrement active count if leaving an active state.
  // Running is also active (process was in Ready/Waiting, then started executing).
  if (oldState == ProcessState::Suspended ||
      oldState == ProcessState::Waiting ||
      oldState == ProcessState::Ready ||
      oldState == ProcessState::Running) {
    if (activeProcessCount > 0)
      --activeProcessCount;
  }

  // Mark as not in any queue. The stale intrusive-list entry (if any)
  // will be skipped during executeReadyProcesses (state == Terminated check).
  proc->inReadyQueue = false;

  LLVM_DEBUG(llvm::dbgs() << "Terminated process " << id << "\n");
}

void ProcessScheduler::resumeProcess(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  if (proc->getState() == ProcessState::Terminated)
    return;

  proc->clearWaiting();
  recordTriggerTime(id);
  scheduleProcess(id, proc->getPreferredRegion());

  LLVM_DEBUG(llvm::dbgs() << "Resumed process " << id << "\n");
}

//===----------------------------------------------------------------------===//
// Delta Cycle Execution
//===----------------------------------------------------------------------===//

void ProcessScheduler::initialize() {
  if (initialized)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Initializing ProcessScheduler with "
                          << processes.size() << " processes\n");

  // Initialize all processes
  for (auto &[id, proc] : processes) {
    if (proc->getState() == ProcessState::Uninitialized) {
      proc->setState(ProcessState::Ready);
      scheduleProcess(id, SchedulingRegion::Active);
    }
  }

  // All registered processes are now active (Ready state).
  activeProcessCount = processes.size();

  // Pre-allocate hot-path vectors to avoid memmove during simulation.
  // Note: intrusive ready queues need no pre-allocation.
  size_t numProcs = processes.size();
  size_t numSigs = signalStates.size();
  signalsChangedThisDelta.reserve(numSigs);
  signalsChangedThisDeltaBits.resize(numSigs, false);
  lastDeltaSignals.reserve(numSigs);
  processesExecutedThisDelta.reserve(numProcs);
  lastDeltaProcesses.reserve(numProcs);
  pendingFastUpdates.reserve(numProcs);

  initialized = true;
}

bool ProcessScheduler::executeDeltaCycle() {
  if (isAbortRequested())
    return false;
  if (!initialized)
    initialize();

  bool anyExecuted = false;
  currentDeltaCount = 0;
  // Clear bitvector entries only for signals that changed (O(n_changed)
  // instead of O(n_total_signals) from DenseSet::clear()).
  for (SignalId s : signalsChangedThisDelta)
    if (s < signalsChangedThisDeltaBits.size())
      signalsChangedThisDeltaBits[s] = false;
  signalsChangedThisDelta.clear();
  processesExecutedThisDelta.clear();
  triggerSignalsThisDelta.clear();
  triggerTimesThisDelta.clear();

  // Process regions with ready processes, skipping empty ones.
  // Build a bitmask of non-empty ready queues for fast iteration.
  uint16_t readyMask = 0;
  for (size_t i = 0; i < static_cast<size_t>(SchedulingRegion::NumRegions); ++i)
    if (!readyQueues[i].empty())
      readyMask |= (1u << i);

  while (readyMask) {
    unsigned regionIdx = __builtin_ctz(readyMask);
    readyMask &= readyMask - 1; // Clear lowest set bit
    auto region = static_cast<SchedulingRegion>(regionIdx);
    size_t executed = executeReadyProcesses(region);
    anyExecuted = anyExecuted || (executed > 0);
  }

  if (anyExecuted) {
    ++stats.deltaCyclesExecuted;
    ++currentDeltaCount;
    // Swap instead of copy to reuse buffer capacity.
    std::swap(lastDeltaSignals, signalsChangedThisDelta);
    std::swap(lastDeltaProcesses, processesExecutedThisDelta);
    std::swap(lastDeltaTriggerSignals, triggerSignalsThisDelta);
    std::swap(lastDeltaTriggerTimes, triggerTimesThisDelta);

    // Check for infinite loop
    if (currentDeltaCount >= config.maxDeltaCycles) {
      ++stats.maxDeltaCyclesReached;
      LLVM_DEBUG(llvm::dbgs()
                 << "Warning: Max delta cycles reached (" << config.maxDeltaCycles
                 << "). Possible infinite loop.\n");
      return false;
    }
  }

  return anyExecuted;
}

size_t ProcessScheduler::executeReadyProcesses(SchedulingRegion region) {
  if (isAbortRequested())
    return 0;
  auto &queue = readyQueues[static_cast<size_t>(region)];
  if (queue.empty())
    return 0;

  LLVM_DEBUG({
    llvm::dbgs() << "executeReadyProcesses region="
                 << getSchedulingRegionName(region) << "\n";
    for (Process *p = queue.head; p; p = p->readyNext)
      llvm::dbgs() << "  queued: " << p->getId() << " ('" << p->getName()
                   << "')\n";
  });

  // Detach the entire chain so execution can schedule new processes into
  // the now-empty queue. Zero allocation — just pointer moves.
  Process *execHead = queue.takeAll();

  size_t executed = 0;
  for (Process *proc = execHead; proc;) {
    Process *next = proc->readyNext;
    proc->readyNext = nullptr;

    // Clear the queue membership flag now that we've dequeued it.
    proc->inReadyQueue = false;

    if (isAbortRequested())
      break;

    if (proc->getState() == ProcessState::Terminated) {
      proc = next;
      continue;
    }

    ProcessId id = proc->getId();

    auto pendingSignalIt = pendingTriggerSignals.find(id);
    if (pendingSignalIt != pendingTriggerSignals.end()) {
      triggerSignalsThisDelta[id] = pendingSignalIt->second;
      pendingTriggerSignals.erase(pendingSignalIt);
    }
    if (pendingTriggerTimes.erase(id))
      triggerTimesThisDelta.insert(id);

    LLVM_DEBUG(llvm::dbgs() << "Executing process " << id << " ('"
                            << proc->getName() << "') in region "
                            << getSchedulingRegionName(region) << "\n");

    proc->setState(ProcessState::Running);
    proc->execute();
    ++stats.processesExecuted;
    ++executed;
    processesExecutedThisDelta.push_back(id);

    // If process didn't suspend or terminate, mark as suspended
    // (waiting for next sensitivity trigger)
    if (proc->getState() == ProcessState::Running) {
      proc->setState(ProcessState::Suspended);
    }

    proc = next;
  }

  // Invoke post-region callback if registered and processes were executed.
  // This is used for two-phase firreg evaluation: all firregs evaluate
  // with pre-update signal values, then pending updates are applied here.
  if (executed > 0) {
    auto &postCb = postRegionCallbacks[static_cast<size_t>(region)];
    if (postCb)
      postCb();
  }

  return executed;
}

size_t ProcessScheduler::executeRegionsUpTo(SchedulingRegion maxRegion) {
  if (isAbortRequested())
    return 0;
  size_t total = 0;
  for (size_t regionIdx = 0;
       regionIdx <= static_cast<size_t>(maxRegion); ++regionIdx) {
    auto region = static_cast<SchedulingRegion>(regionIdx);
    total += executeReadyProcesses(region);
  }
  return total;
}

size_t ProcessScheduler::executeCurrentTime() {
  if (isAbortRequested())
    return 0;
  size_t totalDeltas = 0;
  static bool traceExecCT =
      std::getenv("CIRCT_SIM_TRACE_VPI_TIMING") != nullptr;
  static uint64_t execCTCount = 0;
  ++execCTCount;
  uint64_t localId = execCTCount;

  while (executeDeltaCycle()) {
    ++totalDeltas;
    if (traceExecCT && (totalDeltas <= 3 || totalDeltas % 100 == 0)) {
      llvm::errs() << "[EXEC-CT] id=" << localId << " delta=" << totalDeltas
                   << " t=" << getCurrentTime().realTime << "\n";
    }
    if (totalDeltas % 1000 == 0) {
      LLVM_DEBUG(llvm::dbgs() << "  [ProcessScheduler] Delta cycle " << totalDeltas
                              << ", processes executed: " << stats.processesExecuted << "\n");
    }
    if (totalDeltas >= config.maxDeltaCycles) {
      LLVM_DEBUG(llvm::dbgs() << "Max delta cycles reached at current time\n");
      if (traceExecCT) {
        llvm::errs() << "[EXEC-CT] id=" << localId
                     << " MAX DELTAS REACHED (" << totalDeltas << ")\n";
      }
      break;
    }
  }

  if (traceExecCT && totalDeltas > 0) {
    llvm::errs() << "[EXEC-CT] id=" << localId << " DONE deltas="
                 << totalDeltas << " t=" << getCurrentTime().realTime << "\n";
  }
  return totalDeltas;
}

bool ProcessScheduler::advanceTime() {
  if (isAbortRequested())
    return false;
  // First, process any ready processes
  executeCurrentTime();

  // Find earliest clock domain wake time (0 = none active).
  uint64_t earliestClockWake = UINT64_MAX;
  for (const auto &cd : clockDomains)
    if (cd.active && cd.nextWakeFs < earliestClockWake)
      earliestClockWake = cd.nextWakeFs;
  bool hasClockWork = (earliestClockWake != UINT64_MAX);

  // Check if there are any events, pending fast updates, or clock domains
  if (eventScheduler->isComplete() && pendingFastUpdates.empty() &&
      !hasClockWork) {
    // Check if any processes are ready
    for (auto &queue : readyQueues) {
      if (!queue.empty())
        return true;
    }
    return false;
  }

  // Track whether we did any work (advanced time or processed events)
  bool didWork = false;

  // Process events at the current time, then advance to the next event time.
  // We process all delta cycles at the current time (events may schedule more
  // events at the same time). Then advance to the next real time and return
  // to the caller so it can run VPI callbacks etc.
  while (!eventScheduler->isComplete() || !pendingFastUpdates.empty() ||
         hasClockWork) {
    if (isAbortRequested())
      return false;

    // Flush pending fast signal updates from bytecode processes alongside
    // Event processing. Both types of deferred updates are applied at the
    // same logical delta boundary, preserving IEEE 1800 semantics.
    bool hadPendingUpdates = !pendingFastUpdates.empty();
    if (hadPendingUpdates)
      flushPendingFastUpdates();

    // Try to step a delta cycle at the current time
    bool hadEvents = eventScheduler->stepDelta();

    if (hadPendingUpdates || hadEvents) {
      didWork = true;
      // Events/updates were processed, check if any processes are now ready
      for (auto &queue : readyQueues) {
        if (!queue.empty())
          return true;
      }
      // No processes ready yet, continue processing deltas at this time
      continue;
    }

    // No events or pending updates at current delta — advance to the next
    // event time. Compare with earliest clock domain wake and advance to
    // whichever is sooner.
    uint64_t currentTimeFs = eventScheduler->getCurrentTime().realTime;

    // Recompute earliest clock domain wake.
    earliestClockWake = UINT64_MAX;
    for (const auto &cd : clockDomains)
      if (cd.active && cd.nextWakeFs < earliestClockWake)
        earliestClockWake = cd.nextWakeFs;
    hasClockWork = (earliestClockWake != UINT64_MAX);

    // Try to advance TimeWheel to its next event.
    bool timeWheelAdvanced = false;
    if (!eventScheduler->isComplete()) {
      // If clock domain is earlier than TimeWheel, advance TimeWheel time
      // pointer to clock wake time (without processing TW events).
      timeWheelAdvanced = eventScheduler->advanceToNextTime();
    }

    // If clock domain wake is at or before current time, process it now.
    if (hasClockWork && earliestClockWake <= currentTimeFs) {
      advanceClockDomains(currentTimeFs);
      didWork = true;
      for (auto &queue : readyQueues)
        if (!queue.empty())
          return true;
      continue;
    }

    // If clock domain wake is earlier than next TimeWheel event (or TW is
    // complete), advance sim time directly to the clock wake time.
    // NOTE: Disabled pending EventScheduler::advanceTimeTo implementation
    // (ClockDomain feature from task #22).
    (void)hasClockWork;
    (void)earliestClockWake;

    if (!timeWheelAdvanced) {
      if (eventScheduler->isComplete() && !hasClockWork)
        break;
      // Neither stepDelta nor advanceToNextTime made progress, but
      // isComplete is false. This means orphaned events exist at past
      // delta steps that are unreachable. Break to avoid spinning.
      LLVM_DEBUG(llvm::dbgs()
                 << "advanceTime: stuck — no delta events, no time advance, "
                 << "but EventScheduler not complete. Breaking.\n");
      break;
    }

    didWork = true;
    LLVM_DEBUG(llvm::dbgs() << "Advanced time to "
                            << eventScheduler->getCurrentTime().realTime << " fs\n");
    // Return after each time advance so the main loop can handle VPI
    // scheduling-region callbacks and simulation control checks.
    return true;
  }

  // Return true if we did any work or if there's still work to do
  return didWork || !isComplete();
}

SimTime ProcessScheduler::runUntil(uint64_t maxTimeFemtoseconds) {
  if (!initialized)
    initialize();

  while (!isComplete()) {
    if (isAbortRequested())
      break;
    // Execute current time delta cycles
    executeCurrentTime();

    // Check time limit
    if (getCurrentTime().realTime >= maxTimeFemtoseconds)
      break;

    // Advance to next event
    if (!advanceTime())
      break;
  }

  return getCurrentTime();
}

bool ProcessScheduler::hasReadyProcesses() const {
  for (const auto &queue : readyQueues) {
    if (!queue.empty())
      return true;
  }
  return false;
}

bool ProcessScheduler::isComplete() const {
  // Fast path: if the event scheduler has pending events and there are
  // active processes that could be woken, simulation is not complete.
  // This avoids iterating all processes and all ready queues on every call.
  if (activeProcessCount > 0 && !eventScheduler->isComplete())
    return false;

  // Check clock domains — if any active, simulation is not complete.
  for (const auto &cd : clockDomains)
    if (cd.active)
      return false;

  // Check if any processes are ready
  for (const auto &queue : readyQueues) {
    if (!queue.empty())
      return false;
  }

  // Check if event scheduler has events
  return eventScheduler->isComplete();
}

void ProcessScheduler::reset() {
  // Clear all processes
  processes.clear();
  processVec.clear();
  nextProcId = 1;
  activeProcessCount = 0;

  // Clear signals
  signalStates.clear();
  signalMemory.clear();
  signalIsDirect.clear();
  signalNames.clear();
  signalEncodings.clear();
  nextSigId = 1;

  // Clear mappings
  signalToProcesses.clear();

  // Clear ready queues
  for (auto &queue : readyQueues)
    queue.clear();

  // Clear timed wait queue
  timedWaitQueue.clear();

  // Reset event scheduler
  eventScheduler->reset();

  // Clear clock domains
  clockDomains.clear();
  processToClockDomain.clear();

  // Reset statistics
  stats = Statistics();

  // Reset flags
  initialized = false;
  currentDeltaCount = 0;
  signalsChangedThisDelta.clear();
  std::fill(signalsChangedThisDeltaBits.begin(),
            signalsChangedThisDeltaBits.end(), false);
  lastDeltaSignals.clear();
  processesExecutedThisDelta.clear();
  lastDeltaProcesses.clear();
  pendingTriggerSignals.clear();
  pendingTriggerTimes.clear();
  triggerSignalsThisDelta.clear();
  lastDeltaTriggerSignals.clear();
  triggerTimesThisDelta.clear();
  lastDeltaTriggerTimes.clear();

  LLVM_DEBUG(llvm::dbgs() << "ProcessScheduler reset\n");
}

//===----------------------------------------------------------------------===//
// CombProcessManager Implementation
//===----------------------------------------------------------------------===//

void CombProcessManager::registerCombProcess(ProcessId id) {
  Process *proc = scheduler.getProcess(id);
  if (!proc)
    return;

  proc->setCombinational(true);
  inferredSignals[id] = {};
}

void CombProcessManager::recordSignalRead(ProcessId id, SignalId signalId) {
  auto it = inferredSignals.find(id);
  if (it == inferredSignals.end())
    return;

  // Add signal if not already present
  auto &signals = it->second;
  if (std::find(signals.begin(), signals.end(), signalId) == signals.end()) {
    signals.push_back(signalId);
  }
}

void CombProcessManager::finalizeSensitivity(ProcessId id) {
  auto it = inferredSignals.find(id);
  if (it == inferredSignals.end())
    return;

  SensitivityList sensitivity;
  sensitivity.setAutoInferred(true);

  for (SignalId signalId : it->second) {
    sensitivity.addLevel(signalId);
  }

  scheduler.setSensitivity(id, sensitivity);

  LLVM_DEBUG(llvm::dbgs() << "Finalized auto-inferred sensitivity for process "
                          << id << " with " << it->second.size()
                          << " signals\n");
}

void CombProcessManager::beginTracking(ProcessId id) {
  auto it = inferredSignals.find(id);
  if (it == inferredSignals.end())
    return;

  currentlyTracking = id;
  it->second.clear();
}

void CombProcessManager::endTracking(ProcessId id) {
  if (currentlyTracking != id)
    return;

  finalizeSensitivity(id);
  currentlyTracking = InvalidProcessId;
}

//===----------------------------------------------------------------------===//
// ForkJoinManager Implementation
//===----------------------------------------------------------------------===//

ForkId ForkJoinManager::createFork(ProcessId parentProcess,
                                   ForkJoinType joinType) {
  ForkId id = getNextForkId();
  auto forkGroup = std::make_unique<ForkGroup>(id, joinType, parentProcess);
  forkGroups[id] = std::move(forkGroup);

  // Track parent to fork relationship
  parentToForks[parentProcess].push_back(id);

  LLVM_DEBUG(llvm::dbgs() << "Created fork group " << id << " for parent "
                          << parentProcess << " with join type "
                          << getForkJoinTypeName(joinType) << "\n");

  return id;
}

void ForkJoinManager::addChildToFork(ForkId forkId, ProcessId childProcess) {
  auto it = forkGroups.find(forkId);
  if (it == forkGroups.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: addChildToFork called with invalid "
                               "fork ID " << forkId << "\n");
    return;
  }

  it->second->childProcesses.push_back(childProcess);
  childToFork[childProcess] = forkId;

  LLVM_DEBUG(llvm::dbgs() << "Added child process " << childProcess
                          << " to fork group " << forkId << "\n");
}

void ForkJoinManager::markChildComplete(ProcessId childProcess) {
  auto it = childToFork.find(childProcess);
  if (it == childToFork.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: markChildComplete called for process "
                            << childProcess << " not in any fork group\n");
    return;
  }

  ForkId forkId = it->second;
  auto groupIt = forkGroups.find(forkId);
  if (groupIt == forkGroups.end())
    return;

  ForkGroup *group = groupIt->second.get();
  ++group->completedCount;

  LLVM_DEBUG(llvm::dbgs() << "Child process " << childProcess
                          << " completed in fork group " << forkId
                          << " (" << group->completedCount << "/"
                          << group->childProcesses.size() << ")\n");

  // Check if the fork is now complete and should resume parent
  if (group->isComplete() && !group->joined) {
    group->joined = true;
    // Resume the parent process if it was waiting
    Process *parent = scheduler.getProcess(group->parentProcess);
    if (parent && parent->getState() == ProcessState::Waiting) {
      scheduler.resumeProcess(group->parentProcess);
      LLVM_DEBUG(llvm::dbgs() << "Resuming parent process "
                              << group->parentProcess << " after fork complete\n");
    }
  }

  // Also check if ALL children have completed - this is needed for processes
  // that are waiting on llhd.halt with active forked children. For join_none
  // forks, isComplete() returns true immediately, but we still need to resume
  // the parent when ALL children actually complete so that the halt can proceed.
  if (group->allComplete()) {
    Process *parent = scheduler.getProcess(group->parentProcess);
    if (parent && parent->getState() == ProcessState::Waiting) {
      scheduler.resumeProcess(group->parentProcess);
      LLVM_DEBUG(llvm::dbgs() << "Resuming parent process "
                              << group->parentProcess
                              << " - all fork children now complete\n");
    }
  }
}

bool ForkJoinManager::join(ForkId forkId) {
  auto it = forkGroups.find(forkId);
  if (it == forkGroups.end())
    return true; // Invalid fork, treat as complete

  ForkGroup *group = it->second.get();

  if (group->isComplete()) {
    group->joined = true;
    return true;
  }

  // Not complete, caller should wait
  return false;
}

bool ForkJoinManager::joinAny(ForkId forkId) {
  auto it = forkGroups.find(forkId);
  if (it == forkGroups.end())
    return true;

  ForkGroup *group = it->second.get();
  return group->completedCount >= 1;
}

ForkGroup *ForkJoinManager::getForkGroup(ForkId forkId) {
  auto it = forkGroups.find(forkId);
  return it != forkGroups.end() ? it->second.get() : nullptr;
}

const ForkGroup *ForkJoinManager::getForkGroup(ForkId forkId) const {
  auto it = forkGroups.find(forkId);
  return it != forkGroups.end() ? it->second.get() : nullptr;
}

ForkGroup *ForkJoinManager::getForkGroupForChild(ProcessId childProcess) {
  auto it = childToFork.find(childProcess);
  if (it == childToFork.end())
    return nullptr;
  return getForkGroup(it->second);
}

bool ForkJoinManager::waitFork(ProcessId parentProcess) {
  auto it = parentToForks.find(parentProcess);
  if (it == parentToForks.end())
    return true; // No forks, nothing to wait for

  // Check if all forks with join_none are complete
  for (ForkId forkId : it->second) {
    ForkGroup *group = getForkGroup(forkId);
    if (!group)
      continue;

    // Only wait for join_none forks (join and join_any already waited)
    if (group->joinType == ForkJoinType::JoinNone && !group->allComplete()) {
      return false;
    }
  }

  return true;
}

void ForkJoinManager::disableFork(ForkId forkId) {
  ForkGroup *group = getForkGroup(forkId);
  if (!group)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Disabling fork group " << forkId << " with "
                          << group->childProcesses.size() << " children\n");

  // Terminate all child processes
  for (ProcessId childId : group->childProcesses) {
    scheduler.terminateProcess(childId);
  }

  // Treat disabled children as completed so wait_fork/hasActiveChildren clears.
  group->completedCount = group->childProcesses.size();
  group->joined = true;

  // Resume the parent if it's waiting on fork completion or halt.
  Process *parent = scheduler.getProcess(group->parentProcess);
  if (parent && parent->getState() == ProcessState::Waiting)
    scheduler.resumeProcess(group->parentProcess);
}

void ForkJoinManager::disableAllForks(ProcessId parentProcess) {
  auto it = parentToForks.find(parentProcess);
  if (it == parentToForks.end())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Disabling all forks for parent process "
                          << parentProcess << "\n");

  for (ForkId forkId : it->second) {
    disableFork(forkId);
  }
}

llvm::SmallVector<ForkId, 4>
ForkJoinManager::getForksForParent(ProcessId parentProcess) const {
  auto it = parentToForks.find(parentProcess);
  if (it == parentToForks.end())
    return {};
  return it->second;
}

bool ForkJoinManager::hasActiveChildren(ProcessId parentProcess) const {
  auto it = parentToForks.find(parentProcess);
  if (it == parentToForks.end())
    return false; // No forks, no active children

  // Check ALL fork groups for any incomplete children
  for (ForkId forkId : it->second) {
    const ForkGroup *group = getForkGroup(forkId);
    if (!group)
      continue;

    // Check if any children are still active (not all completed)
    if (!group->allComplete()) {
      LLVM_DEBUG(llvm::dbgs() << "Fork group " << forkId << " has "
                              << (group->childProcesses.size() - group->completedCount)
                              << " active children\n");
      return true;
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// SyncPrimitivesManager Implementation
//===----------------------------------------------------------------------===//

SemaphoreId SyncPrimitivesManager::createSemaphore(int64_t initialCount) {
  SemaphoreId id = nextSemId++;
  semaphores[id] = std::make_unique<Semaphore>(id, initialCount);

  LLVM_DEBUG(llvm::dbgs() << "Created semaphore " << id
                          << " with initial count " << initialCount << "\n");

  return id;
}

void SyncPrimitivesManager::semaphoreGet(SemaphoreId id, ProcessId caller,
                                         int64_t count) {
  Semaphore *sem = getOrCreateSemaphore(id);
  if (!sem)
    return;

  if (!sem->tryGet(count)) {
    // Block the caller
    sem->addWaiter(caller, count);
    Process *proc = scheduler.getProcess(caller);
    if (proc) {
      proc->setState(ProcessState::Waiting);
    }
    LLVM_DEBUG(llvm::dbgs() << "Process " << caller
                            << " waiting on semaphore " << id << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Process " << caller
                            << " acquired " << count << " keys from semaphore "
                            << id << "\n");
  }
}

bool SyncPrimitivesManager::semaphoreTryGet(SemaphoreId id, int64_t count) {
  Semaphore *sem = getOrCreateSemaphore(id);
  if (!sem)
    return false;
  return sem->tryGet(count);
}

void SyncPrimitivesManager::semaphorePut(SemaphoreId id, int64_t count) {
  Semaphore *sem = getOrCreateSemaphore(id);
  if (!sem)
    return;

  sem->put(count);
  LLVM_DEBUG(llvm::dbgs() << "Released " << count << " keys to semaphore "
                          << id << "\n");

  // Try to satisfy waiting processes
  while (ProcessId waiterId = sem->trySatisfyNextWaiter()) {
    scheduler.resumeProcess(waiterId);
    LLVM_DEBUG(llvm::dbgs() << "Resuming process " << waiterId
                            << " from semaphore wait\n");
  }
}

Semaphore *SyncPrimitivesManager::getSemaphore(SemaphoreId id) {
  auto it = semaphores.find(id);
  return it != semaphores.end() ? it->second.get() : nullptr;
}

Semaphore *SyncPrimitivesManager::getOrCreateSemaphore(SemaphoreId id,
                                                        int64_t initialCount) {
  auto it = semaphores.find(id);
  if (it != semaphores.end())
    return it->second.get();
  // Auto-create a semaphore for this ID (supports address-based IDs)
  semaphores[id] = std::make_unique<Semaphore>(id, initialCount);
  LLVM_DEBUG(llvm::dbgs() << "Auto-created semaphore " << id
                          << " with initial count " << initialCount << "\n");
  return semaphores[id].get();
}

MailboxId SyncPrimitivesManager::createMailbox(int32_t bound) {
  MailboxId id = nextMailboxId++;
  mailboxes[id] = std::make_unique<Mailbox>(id, bound);

  LLVM_DEBUG(llvm::dbgs() << "Created mailbox " << id
                          << (bound > 0 ? " bounded to " + std::to_string(bound)
                                        : " unbounded")
                          << "\n");

  return id;
}

void SyncPrimitivesManager::mailboxPut(MailboxId id, ProcessId caller,
                                       uint64_t message) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return;

  if (mbox->isFull()) {
    // Block the caller
    mbox->addPutWaiter(caller, message);
    Process *proc = scheduler.getProcess(caller);
    if (proc) {
      proc->setState(ProcessState::Waiting);
    }
    LLVM_DEBUG(llvm::dbgs() << "Process " << caller
                            << " waiting to put to mailbox " << id << "\n");
  } else {
    mbox->put(message);
    LLVM_DEBUG(llvm::dbgs() << "Message put to mailbox " << id << "\n");

    // Try to satisfy a get waiter
    uint64_t msg;
    if (ProcessId waiterId = mbox->trySatisfyGetWaiter(msg)) {
      scheduler.resumeProcess(waiterId);
      LLVM_DEBUG(llvm::dbgs() << "Resuming process " << waiterId
                              << " from mailbox get wait\n");
    }
  }
}

bool SyncPrimitivesManager::mailboxTryPut(MailboxId id, uint64_t message) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryPut(message);
}

void SyncPrimitivesManager::mailboxGet(MailboxId id, ProcessId caller) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return;

  uint64_t message;
  if (!mbox->tryGet(message)) {
    // Block the caller
    mbox->addGetWaiter(caller);
    Process *proc = scheduler.getProcess(caller);
    if (proc) {
      proc->setState(ProcessState::Waiting);
    }
    LLVM_DEBUG(llvm::dbgs() << "Process " << caller
                            << " waiting to get from mailbox " << id << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Message retrieved from mailbox " << id << "\n");

    // Try to satisfy a put waiter
    if (ProcessId waiterId = mbox->trySatisfyPutWaiter()) {
      scheduler.resumeProcess(waiterId);
      LLVM_DEBUG(llvm::dbgs() << "Resuming process " << waiterId
                              << " from mailbox put wait\n");
    }
  }
}

bool SyncPrimitivesManager::mailboxTryGet(MailboxId id, uint64_t &message) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryGet(message);
}

bool SyncPrimitivesManager::mailboxPeek(MailboxId id, uint64_t &message) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryPeek(message);
}

size_t SyncPrimitivesManager::mailboxNum(MailboxId id) {
  Mailbox *mbox = getOrCreateMailbox(id);
  if (!mbox)
    return 0;
  return mbox->getMessageCount();
}

Mailbox *SyncPrimitivesManager::getMailbox(MailboxId id) {
  auto it = mailboxes.find(id);
  return it != mailboxes.end() ? it->second.get() : nullptr;
}

Mailbox *SyncPrimitivesManager::getOrCreateMailbox(MailboxId id) {
  auto it = mailboxes.find(id);
  if (it != mailboxes.end())
    return it->second.get();
  // Auto-create an unbounded mailbox for this ID (supports SV mailbox new())
  mailboxes[id] = std::make_unique<Mailbox>(id, 0);
  LLVM_DEBUG(llvm::dbgs() << "Auto-created unbounded mailbox " << id << "\n");
  return mailboxes[id].get();
}
