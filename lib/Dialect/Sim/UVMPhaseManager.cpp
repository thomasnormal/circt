//===- UVMPhaseManager.cpp - UVM Phase runtime support --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the UVM phase manager infrastructure for managing UVM
// testbench phases and objection handling.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/UVMPhaseManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "uvm-phase-manager"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// UVMPhaseManager Implementation
//===----------------------------------------------------------------------===//

UVMPhaseManager::UVMPhaseManager(ProcessScheduler &scheduler, Config config)
    : scheduler(scheduler), config(std::move(config)) {}

UVMPhaseManager::~UVMPhaseManager() = default;

//===----------------------------------------------------------------------===//
// Component Registration
//===----------------------------------------------------------------------===//

UVMPhaseCallback *UVMPhaseManager::registerComponent(llvm::StringRef name) {
  auto callback = std::make_unique<UVMPhaseCallback>(name);
  auto *ptr = callback.get();
  components[name] = std::move(callback);
  stats.componentsRegistered++;

  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Registered component '" << name
                          << "'\n");
  return ptr;
}

void UVMPhaseManager::unregisterComponent(llvm::StringRef name) {
  // First, drop any objections from this component
  if (auto it = componentToObjection.find(name);
      it != componentToObjection.end()) {
    activeObjections.erase(it->second);
    componentToObjection.erase(it);
  }

  components.erase(name);
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Unregistered component '" << name
                          << "'\n");
}

UVMPhaseCallback *UVMPhaseManager::getComponent(llvm::StringRef name) {
  auto it = components.find(name);
  if (it != components.end())
    return it->second.get();
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Phase Execution
//===----------------------------------------------------------------------===//

void UVMPhaseManager::runPhase(UVMPhase phase) {
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Running phase '"
                          << getUVMPhaseName(phase) << "'\n");

  currentPhase = phase;
  phaseEndRequested = false;
  stats.phasesExecuted++;

  if (isFunctionPhase(phase)) {
    executeFunctionPhase(phase);
  } else {
    executeTimeConsumingPhase(phase);
  }

  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Phase '"
                          << getUVMPhaseName(phase) << "' completed\n");
}

void UVMPhaseManager::runAllPhases() {
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Running all phases\n");

  for (uint8_t i = 0; i < static_cast<uint8_t>(UVMPhase::NumPhases); ++i) {
    UVMPhase phase = static_cast<UVMPhase>(i);
    runPhase(phase);
  }

  currentPhase = UVMPhase::None;
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: All phases completed\n");
}

void UVMPhaseManager::jumpToPhase(UVMPhase phase) {
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Jumping to phase '"
                          << getUVMPhaseName(phase) << "'\n");

  // Clear any active objections
  activeObjections.clear();
  componentToObjection.clear();

  // Set the current phase
  currentPhase = phase;
  phaseEndRequested = false;
}

void UVMPhaseManager::executeFunctionPhase(UVMPhase phase) {
  // Execute phase callbacks for all components in registration order
  // In UVM, this is typically in depth-first, parent-before-child order
  for (auto &entry : components) {
    if (entry.second->hasCallback(phase)) {
      LLVM_DEBUG(llvm::dbgs() << "  Executing " << getUVMPhaseName(phase)
                              << " for component '" << entry.first() << "'\n");
      entry.second->executePhase(phase);
    }
  }
}

void UVMPhaseManager::executeTimeConsumingPhase(UVMPhase phase) {
  // Execute the phase callback for all components
  // This starts the phase but doesn't wait for completion
  for (auto &entry : components) {
    if (entry.second->hasCallback(phase)) {
      LLVM_DEBUG(llvm::dbgs() << "  Starting " << getUVMPhaseName(phase)
                              << " for component '" << entry.first() << "'\n");
      entry.second->executePhase(phase);
    }
  }

  // Wait for all objections to be dropped
  waitForObjections();

  // Apply drain time if configured
  if (config.drainTime > 0 && !activeObjections.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Applying drain time: " << config.drainTime << " fs\n");
    // The actual drain time delay would be scheduled through the scheduler
    // For now, we just note it - integration with ProcessScheduler will handle
    // the actual delay
  }
}

void UVMPhaseManager::waitForObjections() {
  // In a real implementation, this would block until objections are dropped
  // For now, we check if there are any objections and handle the timeout
  if (currentPhase == UVMPhase::Run && config.runPhaseTimeout > 0) {
    // Schedule a timeout event
    SimTime timeout = scheduler.getCurrentTime();
    timeout = timeout.advanceTime(config.runPhaseTimeout);

    LLVM_DEBUG(llvm::dbgs()
               << "  Run phase timeout set for: " << timeout.realTime
               << " fs\n");
  }

  // In actual operation, the simulation kernel runs until objections are
  // dropped The phase completes when:
  // 1. All objections are dropped, AND
  // 2. Drain time has elapsed (if configured)
  // For unit testing, we just check if objections are present
  while (hasObjections() && !phaseEndRequested) {
    // In real implementation, this would yield to the scheduler
    // For testing purposes, we break to avoid infinite loop
    break;
  }
}

bool UVMPhaseManager::canPhaseComplete() const {
  return !hasObjections() || phaseEndRequested;
}

//===----------------------------------------------------------------------===//
// Objection Handling
//===----------------------------------------------------------------------===//

ObjectionId UVMPhaseManager::raiseObjection(llvm::StringRef componentName,
                                            llvm::StringRef description) {
  // Check if component already has an objection
  auto existingIt = componentToObjection.find(componentName);
  if (existingIt != componentToObjection.end()) {
    // Increment the count on existing objection
    auto objIt = activeObjections.find(existingIt->second);
    if (objIt != activeObjections.end()) {
      objIt->second->count++;
      LLVM_DEBUG(llvm::dbgs()
                 << "UVMPhaseManager: Raised objection count for '"
                 << componentName << "' to " << objIt->second->count << "\n");
      return existingIt->second;
    }
  }

  // Create a new objection
  ObjectionId id = nextObjectionId();
  auto objection =
      std::make_unique<UVMObjection>(id, componentName, currentPhase, description);
  activeObjections[id] = std::move(objection);
  componentToObjection[componentName] = id;
  stats.objectionsRaised++;

  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Raised objection from '"
                          << componentName << "' (id=" << id << ")\n");
  return id;
}

void UVMPhaseManager::dropObjection(ObjectionId id) {
  auto it = activeObjections.find(id);
  if (it == activeObjections.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMPhaseManager: Warning - dropping unknown objection id="
               << id << "\n");
    return;
  }

  // Decrement count
  it->second->count--;
  LLVM_DEBUG(llvm::dbgs()
             << "UVMPhaseManager: Dropped objection from '"
             << it->second->componentName << "' (count=" << it->second->count
             << ")\n");

  // Remove if count reaches zero
  if (it->second->count <= 0) {
    componentToObjection.erase(it->second->componentName);
    activeObjections.erase(it);
    stats.objectionsDropped++;

    // Check if all objections are dropped
    if (activeObjections.empty() && allDroppedCallback) {
      LLVM_DEBUG(llvm::dbgs()
                 << "UVMPhaseManager: All objections dropped\n");
      allDroppedCallback();
    }
  }
}

void UVMPhaseManager::dropObjection(llvm::StringRef componentName) {
  auto it = componentToObjection.find(componentName);
  if (it != componentToObjection.end()) {
    dropObjection(it->second);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "UVMPhaseManager: Warning - no objection from '"
               << componentName << "'\n");
  }
}

void UVMPhaseManager::setObjectionCount(ObjectionId id, int32_t count) {
  auto it = activeObjections.find(id);
  if (it != activeObjections.end()) {
    it->second->count = count;
    LLVM_DEBUG(llvm::dbgs()
               << "UVMPhaseManager: Set objection count for '"
               << it->second->componentName << "' to " << count << "\n");

    if (count <= 0) {
      componentToObjection.erase(it->second->componentName);
      activeObjections.erase(it);
      stats.objectionsDropped++;

      if (activeObjections.empty() && allDroppedCallback) {
        allDroppedCallback();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Phase Control
//===----------------------------------------------------------------------===//

void UVMPhaseManager::requestPhaseEnd() {
  phaseEndRequested = true;
  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Phase end requested for '"
                          << getUVMPhaseName(currentPhase) << "'\n");
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

void UVMPhaseManager::reset() {
  currentPhase = UVMPhase::None;
  phaseEndRequested = false;
  activeObjections.clear();
  componentToObjection.clear();
  nextObjId = 0;
  stats = Statistics();

  LLVM_DEBUG(llvm::dbgs() << "UVMPhaseManager: Reset\n");
}
