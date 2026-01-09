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
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "sim-process-scheduler"

using namespace circt;
using namespace circt::sim;

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
  processes[id] = std::move(process);

  ++stats.processesRegistered;
  LLVM_DEBUG(llvm::dbgs() << "Registered process '" << name << "' with ID "
                          << id << "\n");

  return id;
}

void ProcessScheduler::unregisterProcess(ProcessId id) {
  auto it = processes.find(id);
  if (it == processes.end())
    return;

  // Remove from signal mappings
  for (auto &[signalId, procList] : signalToProcesses) {
    procList.erase(std::remove(procList.begin(), procList.end(), id),
                   procList.end());
  }

  // Remove from ready queues
  for (auto &queue : readyQueues) {
    queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());
  }

  processes.erase(it);
  LLVM_DEBUG(llvm::dbgs() << "Unregistered process ID " << id << "\n");
}

Process *ProcessScheduler::getProcess(ProcessId id) {
  auto it = processes.find(id);
  return it != processes.end() ? it->second.get() : nullptr;
}

const Process *ProcessScheduler::getProcess(ProcessId id) const {
  auto it = processes.find(id);
  return it != processes.end() ? it->second.get() : nullptr;
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

  // Update signal-to-process mappings
  for (const auto &entry : sensitivity.getEntries()) {
    signalToProcesses[entry.signalId].push_back(id);
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
  signalToProcesses[signalId].push_back(id);

  LLVM_DEBUG(llvm::dbgs() << "Added sensitivity for process " << id
                          << " to signal " << signalId << " edge="
                          << getEdgeTypeName(edge) << "\n");
}

void ProcessScheduler::clearSensitivity(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  // Remove from signal mappings
  for (const auto &entry : proc->getSensitivityList().getEntries()) {
    auto &procList = signalToProcesses[entry.signalId];
    procList.erase(std::remove(procList.begin(), procList.end(), id),
                   procList.end());
  }

  proc->getSensitivityList().clear();
}

SignalId ProcessScheduler::registerSignal(const std::string &name,
                                          uint32_t width) {
  SignalId id = nextSignalId();
  signalStates[id] = SignalState(width);
  signalNames[id] = name;

  LLVM_DEBUG(llvm::dbgs() << "Registered signal '" << name << "' with ID " << id
                          << " width=" << width << "\n");

  return id;
}

void ProcessScheduler::updateSignal(SignalId signalId,
                                    const SignalValue &newValue) {
  auto it = signalStates.find(signalId);
  if (it == signalStates.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: updating unknown signal " << signalId
                            << "\n");
    return;
  }

  SignalValue oldValue = it->second.getCurrentValue();
  EdgeType edge = it->second.updateValue(newValue);

  ++stats.signalUpdates;

  if (edge != EdgeType::None) {
    ++stats.edgesDetected;
    LLVM_DEBUG(llvm::dbgs() << "Signal " << signalId << " changed: edge="
                            << getEdgeTypeName(edge) << "\n");
    triggerSensitiveProcesses(signalId, oldValue, newValue);
  }
}

const SignalValue &ProcessScheduler::getSignalValue(SignalId signalId) const {
  auto it = signalStates.find(signalId);
  if (it == signalStates.end())
    return unknownSignal;
  return it->second.getCurrentValue();
}

void ProcessScheduler::triggerSensitiveProcesses(SignalId signalId,
                                                 const SignalValue &oldVal,
                                                 const SignalValue &newVal) {
  auto it = signalToProcesses.find(signalId);
  if (it == signalToProcesses.end())
    return;

  for (ProcessId procId : it->second) {
    Process *proc = getProcess(procId);
    if (!proc)
      continue;

    // Check if process state allows triggering
    if (proc->getState() != ProcessState::Suspended &&
        proc->getState() != ProcessState::Waiting &&
        proc->getState() != ProcessState::Ready)
      continue;

    // For waiting processes, check the waiting sensitivity
    if (proc->getState() == ProcessState::Waiting) {
      if (proc->getWaitingSensitivity().isTriggeredBy(signalId, oldVal,
                                                       newVal)) {
        proc->clearWaiting();
        scheduleProcess(procId, proc->getPreferredRegion());
        LLVM_DEBUG(llvm::dbgs()
                   << "Process " << procId << " triggered from waiting state\n");
      }
      continue;
    }

    // For suspended/ready processes, check the main sensitivity list
    if (proc->getSensitivityList().isTriggeredBy(signalId, oldVal, newVal)) {
      scheduleProcess(procId, proc->getPreferredRegion());
      LLVM_DEBUG(llvm::dbgs() << "Process " << procId << " triggered\n");
    }
  }
}

//===----------------------------------------------------------------------===//
// Process Execution Control
//===----------------------------------------------------------------------===//

void ProcessScheduler::scheduleProcess(ProcessId id, SchedulingRegion region) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  if (proc->getState() == ProcessState::Terminated)
    return;

  // Add to ready queue if not already there
  auto &queue = readyQueues[static_cast<size_t>(region)];
  if (std::find(queue.begin(), queue.end(), id) == queue.end()) {
    queue.push_back(id);
    proc->setState(ProcessState::Ready);
    LLVM_DEBUG(llvm::dbgs() << "Scheduled process " << id << " in region "
                            << getSchedulingRegionName(region) << "\n");
  }
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

  // Ensure the signals in the wait list are mapped to this process
  for (const auto &entry : waitList.getEntries()) {
    auto &procList = signalToProcesses[entry.signalId];
    if (std::find(procList.begin(), procList.end(), id) == procList.end()) {
      procList.push_back(id);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Process " << id << " waiting for "
                          << waitList.size() << " events\n");
}

void ProcessScheduler::terminateProcess(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  proc->setState(ProcessState::Terminated);

  // Remove from ready queues
  for (auto &queue : readyQueues) {
    queue.erase(std::remove(queue.begin(), queue.end(), id), queue.end());
  }

  LLVM_DEBUG(llvm::dbgs() << "Terminated process " << id << "\n");
}

void ProcessScheduler::resumeProcess(ProcessId id) {
  Process *proc = getProcess(id);
  if (!proc)
    return;

  if (proc->getState() == ProcessState::Terminated)
    return;

  proc->clearWaiting();
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

  initialized = true;
}

bool ProcessScheduler::executeDeltaCycle() {
  if (!initialized)
    initialize();

  bool anyExecuted = false;
  currentDeltaCount = 0;

  // Process all regions in order
  for (size_t regionIdx = 0;
       regionIdx < static_cast<size_t>(SchedulingRegion::NumRegions);
       ++regionIdx) {
    auto region = static_cast<SchedulingRegion>(regionIdx);
    size_t executed = executeReadyProcesses(region);
    anyExecuted = anyExecuted || (executed > 0);
  }

  if (anyExecuted) {
    ++stats.deltaCyclesExecuted;
    ++currentDeltaCount;

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
  auto &queue = readyQueues[static_cast<size_t>(region)];
  if (queue.empty())
    return 0;

  // Copy the queue since execution may modify it
  std::vector<ProcessId> toExecute = std::move(queue);
  queue.clear();

  size_t executed = 0;
  for (ProcessId id : toExecute) {
    Process *proc = getProcess(id);
    if (!proc)
      continue;

    if (proc->getState() == ProcessState::Terminated)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Executing process " << id << " ('"
                            << proc->getName() << "') in region "
                            << getSchedulingRegionName(region) << "\n");

    proc->setState(ProcessState::Running);
    proc->execute();
    ++stats.processesExecuted;
    ++executed;

    // If process didn't suspend or terminate, mark as suspended
    // (waiting for next sensitivity trigger)
    if (proc->getState() == ProcessState::Running) {
      proc->setState(ProcessState::Suspended);
    }
  }

  return executed;
}

size_t ProcessScheduler::executeCurrentTime() {
  size_t totalDeltas = 0;

  while (executeDeltaCycle()) {
    ++totalDeltas;
    if (totalDeltas >= config.maxDeltaCycles) {
      LLVM_DEBUG(llvm::dbgs() << "Max delta cycles reached at current time\n");
      break;
    }
  }

  return totalDeltas;
}

bool ProcessScheduler::advanceTime() {
  // First, process any ready processes
  executeCurrentTime();

  // Check if there are any events pending
  if (eventScheduler->isComplete()) {
    // Check if any processes are ready
    for (auto &queue : readyQueues) {
      if (!queue.empty())
        return true;
    }
    return false;
  }

  // Advance to the next event
  while (!eventScheduler->isComplete()) {
    if (eventScheduler->stepDelta()) {
      // Event was processed, check if any processes are now ready
      for (auto &queue : readyQueues) {
        if (!queue.empty())
          return true;
      }
    } else {
      // Try to advance real time
      if (!eventScheduler->isComplete()) {
        // Internal event scheduler advancement
        break;
      }
    }
  }

  return !isComplete();
}

SimTime ProcessScheduler::runUntil(uint64_t maxTimeFemtoseconds) {
  if (!initialized)
    initialize();

  while (!isComplete()) {
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

bool ProcessScheduler::isComplete() const {
  // Check if any processes are ready
  for (const auto &queue : readyQueues) {
    if (!queue.empty())
      return false;
  }

  // Check if any processes are suspended (waiting for events)
  for (const auto &[id, proc] : processes) {
    if (proc->getState() == ProcessState::Suspended ||
        proc->getState() == ProcessState::Waiting ||
        proc->getState() == ProcessState::Ready) {
      // Check if there are pending events that could wake them
      if (!eventScheduler->isComplete())
        return false;
    }
  }

  // Check if event scheduler has events
  return eventScheduler->isComplete();
}

void ProcessScheduler::reset() {
  // Clear all processes
  processes.clear();
  nextProcId = 1;

  // Clear signals
  signalStates.clear();
  signalNames.clear();
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

  // Reset statistics
  stats = Statistics();

  // Reset flags
  initialized = false;
  currentDeltaCount = 0;

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
