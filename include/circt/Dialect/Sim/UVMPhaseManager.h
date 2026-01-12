//===- UVMPhaseManager.h - UVM Phase runtime support -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the UVM phase manager infrastructure for managing UVM
// testbench phases and objection handling. It provides runtime support for
// the UVM phasing mechanism as described in the UVM Reference Manual.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_UVMPHASEMANAGER_H
#define CIRCT_DIALECT_SIM_UVMPHASEMANAGER_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// UVMPhase - UVM phase enumeration
//===----------------------------------------------------------------------===//

/// UVM phases following the UVM specification.
/// The phases execute in order from Build to Final.
enum class UVMPhase : uint8_t {
  /// Build phase: Create and configure component hierarchy
  Build = 0,

  /// Connect phase: Connect TLM ports and exports
  Connect = 1,

  /// End of elaboration phase: Final configuration before simulation
  EndOfElaboration = 2,

  /// Start of simulation phase: Prepare for simulation start
  StartOfSimulation = 3,

  /// Run phase: Main simulation time-consuming phase
  Run = 4,

  /// Extract phase: Extract data from simulation (coverage, etc.)
  Extract = 5,

  /// Check phase: Check for DUT correctness
  Check = 6,

  /// Report phase: Report results
  Report = 7,

  /// Final phase: Final cleanup
  Final = 8,

  /// Number of phases (for array sizing)
  NumPhases = 9,

  /// Special value indicating no phase
  None = 255
};

/// Get the name of a UVM phase.
inline const char *getUVMPhaseName(UVMPhase phase) {
  switch (phase) {
  case UVMPhase::Build:
    return "build";
  case UVMPhase::Connect:
    return "connect";
  case UVMPhase::EndOfElaboration:
    return "end_of_elaboration";
  case UVMPhase::StartOfSimulation:
    return "start_of_simulation";
  case UVMPhase::Run:
    return "run";
  case UVMPhase::Extract:
    return "extract";
  case UVMPhase::Check:
    return "check";
  case UVMPhase::Report:
    return "report";
  case UVMPhase::Final:
    return "final";
  default:
    return "none";
  }
}

/// Parse a UVM phase from a string.
inline UVMPhase parseUVMPhase(llvm::StringRef str) {
  if (str == "build")
    return UVMPhase::Build;
  if (str == "connect")
    return UVMPhase::Connect;
  if (str == "end_of_elaboration")
    return UVMPhase::EndOfElaboration;
  if (str == "start_of_simulation")
    return UVMPhase::StartOfSimulation;
  if (str == "run")
    return UVMPhase::Run;
  if (str == "extract")
    return UVMPhase::Extract;
  if (str == "check")
    return UVMPhase::Check;
  if (str == "report")
    return UVMPhase::Report;
  if (str == "final")
    return UVMPhase::Final;
  return UVMPhase::None;
}

/// Check if a phase is a time-consuming phase.
/// Time-consuming phases can consume simulation time, while
/// function phases complete in zero simulation time.
inline bool isTimeConsumingPhase(UVMPhase phase) {
  // Only the run phase is time-consuming in the standard UVM phases
  return phase == UVMPhase::Run;
}

/// Check if a phase is a function phase.
inline bool isFunctionPhase(UVMPhase phase) {
  return !isTimeConsumingPhase(phase);
}

//===----------------------------------------------------------------------===//
// UVMObjection - Objection tracking for phase completion
//===----------------------------------------------------------------------===//

/// Unique identifier for an objection.
using ObjectionId = uint64_t;

/// Invalid objection ID constant.
constexpr ObjectionId InvalidObjectionId = 0;

/// Represents an objection raised by a component to prevent phase completion.
struct UVMObjection {
  /// Unique identifier for this objection.
  ObjectionId id;

  /// Name of the component that raised the objection.
  std::string componentName;

  /// Optional description of why the objection was raised.
  std::string description;

  /// The phase this objection applies to.
  UVMPhase phase;

  /// Count for this objection (can be raised multiple times).
  int32_t count;

  UVMObjection(ObjectionId id, llvm::StringRef component, UVMPhase phase,
               llvm::StringRef desc = "")
      : id(id), componentName(component.str()), description(desc.str()),
        phase(phase), count(1) {}
};

//===----------------------------------------------------------------------===//
// UVMPhaseCallback - Callback interface for phase events
//===----------------------------------------------------------------------===//

/// Interface for components to receive phase callbacks.
class UVMPhaseCallback {
public:
  using PhaseFunc = std::function<void(UVMPhase)>;

  UVMPhaseCallback() = default;
  explicit UVMPhaseCallback(llvm::StringRef name) : componentName(name.str()) {}

  virtual ~UVMPhaseCallback() = default;

  /// Get the component name.
  const std::string &getName() const { return componentName; }

  /// Set a callback for a specific phase.
  void setPhaseCallback(UVMPhase phase, PhaseFunc callback) {
    phaseCallbacks[static_cast<size_t>(phase)] = std::move(callback);
  }

  /// Execute the callback for a phase.
  void executePhase(UVMPhase phase) {
    if (auto &cb = phaseCallbacks[static_cast<size_t>(phase)])
      cb(phase);
  }

  /// Check if a callback is registered for a phase.
  bool hasCallback(UVMPhase phase) const {
    return phaseCallbacks[static_cast<size_t>(phase)] != nullptr;
  }

private:
  std::string componentName;
  PhaseFunc phaseCallbacks[static_cast<size_t>(UVMPhase::NumPhases)];
};

//===----------------------------------------------------------------------===//
// UVMPhaseManager - Main phase management infrastructure
//===----------------------------------------------------------------------===//

/// Manages UVM phases, phase callbacks, and objection handling.
/// This is the runtime component that coordinates UVM testbench execution.
class UVMPhaseManager {
public:
  /// Configuration for the phase manager.
  struct Config {
    /// Timeout for the run phase in femtoseconds (0 = no timeout).
    uint64_t runPhaseTimeout;

    /// Enable debug output.
    bool debug;

    /// Default drain time after all objections dropped (femtoseconds).
    uint64_t drainTime;

    Config() : runPhaseTimeout(0), debug(false), drainTime(0) {}
  };

  UVMPhaseManager(ProcessScheduler &scheduler, Config config = Config());
  ~UVMPhaseManager();

  //===--------------------------------------------------------------------===//
  // Component Registration
  //===--------------------------------------------------------------------===//

  /// Register a component with the phase manager.
  /// Returns a callback interface for the component to set phase callbacks.
  UVMPhaseCallback *registerComponent(llvm::StringRef name);

  /// Unregister a component.
  void unregisterComponent(llvm::StringRef name);

  /// Get a registered component.
  UVMPhaseCallback *getComponent(llvm::StringRef name);

  /// Get all registered components.
  const llvm::StringMap<std::unique_ptr<UVMPhaseCallback>> &
  getComponents() const {
    return components;
  }

  //===--------------------------------------------------------------------===//
  // Phase Execution
  //===--------------------------------------------------------------------===//

  /// Run a specific phase for all registered components.
  void runPhase(UVMPhase phase);

  /// Run all phases in sequence.
  void runAllPhases();

  /// Get the current phase.
  UVMPhase getCurrentPhase() const { return currentPhase; }

  /// Check if currently in a specific phase.
  bool isInPhase(UVMPhase phase) const { return currentPhase == phase; }

  /// Jump to a specific phase (for phase debugging/testing).
  void jumpToPhase(UVMPhase phase);

  //===--------------------------------------------------------------------===//
  // Objection Handling
  //===--------------------------------------------------------------------===//

  /// Raise an objection to prevent the current phase from completing.
  ObjectionId raiseObjection(llvm::StringRef componentName,
                             llvm::StringRef description = "");

  /// Drop a previously raised objection.
  void dropObjection(ObjectionId id);

  /// Drop an objection by component name.
  void dropObjection(llvm::StringRef componentName);

  /// Set the objection count for a component (allows raising multiple times).
  void setObjectionCount(ObjectionId id, int32_t count);

  /// Get the current objection count.
  size_t getObjectionCount() const { return activeObjections.size(); }

  /// Check if there are any active objections.
  bool hasObjections() const { return !activeObjections.empty(); }

  /// Get all active objections.
  const llvm::DenseMap<ObjectionId, std::unique_ptr<UVMObjection>> &
  getActiveObjections() const {
    return activeObjections;
  }

  /// Set a callback to be called when all objections are dropped.
  void setAllDroppedCallback(std::function<void()> callback) {
    allDroppedCallback = std::move(callback);
  }

  //===--------------------------------------------------------------------===//
  // Phase Control
  //===--------------------------------------------------------------------===//

  /// Set the drain time for after objections are dropped.
  void setDrainTime(uint64_t femtoseconds) { config.drainTime = femtoseconds; }

  /// Get the current drain time.
  uint64_t getDrainTime() const { return config.drainTime; }

  /// Set the run phase timeout.
  void setRunPhaseTimeout(uint64_t femtoseconds) {
    config.runPhaseTimeout = femtoseconds;
  }

  /// Get the run phase timeout.
  uint64_t getRunPhaseTimeout() const { return config.runPhaseTimeout; }

  /// Request the current phase to end (when objections allow).
  void requestPhaseEnd();

  /// Check if phase end has been requested.
  bool isPhaseEndRequested() const { return phaseEndRequested; }

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t phasesExecuted = 0;
    size_t objectionsRaised = 0;
    size_t objectionsDropped = 0;
    size_t componentsRegistered = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Reset the phase manager.
  void reset();

private:
  /// Execute callbacks for a function phase.
  void executeFunctionPhase(UVMPhase phase);

  /// Execute a time-consuming phase.
  void executeTimeConsumingPhase(UVMPhase phase);

  /// Wait for all objections to be dropped.
  void waitForObjections();

  /// Check if the phase can complete.
  bool canPhaseComplete() const;

  /// Generate the next objection ID.
  ObjectionId nextObjectionId() { return ++nextObjId; }

  ProcessScheduler &scheduler;
  Config config;

  // Component management
  llvm::StringMap<std::unique_ptr<UVMPhaseCallback>> components;

  // Phase state
  UVMPhase currentPhase = UVMPhase::None;
  bool phaseEndRequested = false;

  // Objection management
  llvm::DenseMap<ObjectionId, std::unique_ptr<UVMObjection>> activeObjections;
  llvm::StringMap<ObjectionId> componentToObjection;
  ObjectionId nextObjId = 0;

  // Callbacks
  std::function<void()> allDroppedCallback;

  // Statistics
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// UVMPhaseJumpException - Exception for phase jumps
//===----------------------------------------------------------------------===//

/// Exception thrown when a phase jump is requested.
class UVMPhaseJumpException : public std::exception {
public:
  UVMPhaseJumpException(UVMPhase targetPhase)
      : target(targetPhase),
        message("Phase jump to " + std::string(getUVMPhaseName(targetPhase))) {}

  UVMPhase getTargetPhase() const { return target; }

  const char *what() const noexcept override { return message.c_str(); }

private:
  UVMPhase target;
  std::string message;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_UVMPHASEMANAGER_H
