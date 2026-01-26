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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>

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

  uint32_t signalWidth = it->second.getCurrentValue().getWidth();
  SignalValue normalizedValue =
      normalizeSignalValueWidth(newValue, signalWidth);
  SignalValue oldValue = it->second.getCurrentValue();
  EdgeType edge = it->second.updateValue(normalizedValue);

  ++stats.signalUpdates;

  if (edge != EdgeType::None) {
    ++stats.edgesDetected;
    LLVM_DEBUG(llvm::dbgs() << "Signal " << signalId << " changed: edge="
                            << getEdgeTypeName(edge) << "\n");
    triggerSensitiveProcesses(signalId, oldValue, normalizedValue);
  }
}

void ProcessScheduler::updateSignalWithStrength(SignalId signalId,
                                                uint64_t driverId,
                                                const SignalValue &newValue,
                                                DriveStrength strength0,
                                                DriveStrength strength1) {
  auto it = signalStates.find(signalId);
  if (it == signalStates.end()) {
    LLVM_DEBUG(llvm::dbgs() << "Warning: updating unknown signal " << signalId
                            << "\n");
    return;
  }

  uint32_t signalWidth = it->second.getCurrentValue().getWidth();
  SignalValue normalizedValue =
      normalizeSignalValueWidth(newValue, signalWidth);
  SignalValue oldValue = it->second.getCurrentValue();

  // Add/update the driver with its strength information
  it->second.addOrUpdateDriver(driverId, normalizedValue, strength0, strength1);

  // Resolve all drivers to get the final signal value
  SignalValue resolvedValue = it->second.resolveDrivers();
  SignalValue normalizedResolved =
      normalizeSignalValueWidth(resolvedValue, signalWidth);

  // Update the signal with the resolved value
  EdgeType edge = it->second.updateValue(normalizedResolved);

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
    if (it->second.hasMultipleDrivers()) {
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
    triggerSensitiveProcesses(signalId, oldValue, normalizedResolved);
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

  // Also update the main sensitivity list so it persists across wake/sleep cycles.
  // This is critical for LLHD processes that use llhd.wait - if a process wakes
  // but doesn't re-execute to its next wait (due to control flow or errors),
  // it would otherwise become orphaned in Suspended state with no sensitivity.
  proc->getSensitivityList() = waitList;

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

  // Track whether we did any work (advanced time or processed events)
  bool didWork = false;

  // Advance to the next event time and process ONE time step at a time.
  // This is critical for correct behavior: we must return control to the
  // ProcessScheduler after each event time so it can execute processes
  // that were scheduled by the events.
  while (!eventScheduler->isComplete()) {
    // Try to step a delta cycle at the current time
    if (eventScheduler->stepDelta()) {
      didWork = true;
      // Events were processed, check if any processes are now ready
      for (auto &queue : readyQueues) {
        if (!queue.empty())
          return true;
      }
      // No processes ready yet, continue processing deltas at this time
      continue;
    }

    // No events at current time - advance to the next event time.
    // Use advanceToNextTime which only advances time without processing events.
    SimTime oldTime = eventScheduler->getCurrentTime();
    if (!eventScheduler->advanceToNextTime()) {
      // Could not advance - either no events or already at next event time
      // Check if there are events to process at the "new" current time
      if (eventScheduler->isComplete())
        break;
      // There might be events at current time that weren't cascaded yet
      // Try stepping again
      continue;
    }

    didWork = true;
    LLVM_DEBUG(llvm::dbgs() << "Advanced time from " << oldTime.realTime
                            << " to " << eventScheduler->getCurrentTime().realTime << " fs\n");

    // Time advanced, now process events at the new time
    // Continue to the next iteration which will call stepDelta
  }

  // Return true if we did any work or if there's still work to do
  return didWork || !isComplete();
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

  group->joined = true;
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
  Semaphore *sem = getSemaphore(id);
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
  Semaphore *sem = getSemaphore(id);
  if (!sem)
    return false;
  return sem->tryGet(count);
}

void SyncPrimitivesManager::semaphorePut(SemaphoreId id, int64_t count) {
  Semaphore *sem = getSemaphore(id);
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
  Mailbox *mbox = getMailbox(id);
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
  Mailbox *mbox = getMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryPut(message);
}

void SyncPrimitivesManager::mailboxGet(MailboxId id, ProcessId caller) {
  Mailbox *mbox = getMailbox(id);
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
  Mailbox *mbox = getMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryGet(message);
}

bool SyncPrimitivesManager::mailboxPeek(MailboxId id, uint64_t &message) {
  Mailbox *mbox = getMailbox(id);
  if (!mbox)
    return false;
  return mbox->tryPeek(message);
}

size_t SyncPrimitivesManager::mailboxNum(MailboxId id) {
  Mailbox *mbox = getMailbox(id);
  if (!mbox)
    return 0;
  return mbox->getMessageCount();
}

Mailbox *SyncPrimitivesManager::getMailbox(MailboxId id) {
  auto it = mailboxes.find(id);
  return it != mailboxes.end() ? it->second.get() : nullptr;
}
