//===- ProcessScheduler.h - Process scheduling for simulation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the process scheduler infrastructure for event-driven
// simulation. It manages concurrent processes with sensitivity lists,
// edge detection, and delta cycle semantics following IEEE 1800.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_PROCESSSCHEDULER_H
#define CIRCT_DIALECT_SIM_PROCESSSCHEDULER_H

#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// ProcessState - Process execution states
//===----------------------------------------------------------------------===//

/// States that a simulation process can be in.
enum class ProcessState : uint8_t {
  /// Process has not yet been initialized.
  Uninitialized = 0,

  /// Process is ready to run but not currently executing.
  Ready = 1,

  /// Process is currently executing.
  Running = 2,

  /// Process is suspended waiting for events or delay.
  Suspended = 3,

  /// Process is waiting for a specific event condition.
  Waiting = 4,

  /// Process has completed execution and will not run again.
  Terminated = 5,
};

/// Get the name of a process state for debugging.
inline const char *getProcessStateName(ProcessState state) {
  switch (state) {
  case ProcessState::Uninitialized:
    return "Uninitialized";
  case ProcessState::Ready:
    return "Ready";
  case ProcessState::Running:
    return "Running";
  case ProcessState::Suspended:
    return "Suspended";
  case ProcessState::Waiting:
    return "Waiting";
  case ProcessState::Terminated:
    return "Terminated";
  }
  return "Unknown";
}

//===----------------------------------------------------------------------===//
// EdgeType - Types of signal edge transitions
//===----------------------------------------------------------------------===//

/// Types of signal edge transitions for edge-sensitive triggers.
enum class EdgeType : uint8_t {
  /// No edge detection, level-sensitive.
  None = 0,

  /// Rising edge: 0->1, 0->X, X->1 (simplified: old=0 && new=1).
  Posedge = 1,

  /// Falling edge: 1->0, 1->X, X->0 (simplified: old=1 && new=0).
  Negedge = 2,

  /// Any edge: any value change.
  AnyEdge = 3,
};

/// Get the name of an edge type for debugging.
inline const char *getEdgeTypeName(EdgeType edge) {
  switch (edge) {
  case EdgeType::None:
    return "none";
  case EdgeType::Posedge:
    return "posedge";
  case EdgeType::Negedge:
    return "negedge";
  case EdgeType::AnyEdge:
    return "anyedge";
  }
  return "unknown";
}

/// Parse an edge type from a string.
inline EdgeType parseEdgeType(llvm::StringRef str) {
  if (str == "posedge")
    return EdgeType::Posedge;
  if (str == "negedge")
    return EdgeType::Negedge;
  if (str == "anyedge" || str == "any")
    return EdgeType::AnyEdge;
  return EdgeType::None;
}

//===----------------------------------------------------------------------===//
// SignalValue - Signal value representation for edge detection
//===----------------------------------------------------------------------===//

/// Represents a signal value that can track changes for edge detection.
/// Supports multi-bit values with special handling for single-bit edge detection.
/// Uses llvm::APInt to support arbitrary bit widths beyond 64 bits.
class SignalValue {
public:
  /// Default constructor creates an X (unknown) value.
  SignalValue() : value(1, 0), isX(true) {}

  /// Construct from an integer value (for widths ≤ 64 bits).
  explicit SignalValue(uint64_t val, uint32_t w = 1)
      : value(w, val), isX(false) {}

  /// Construct from an APInt value (for arbitrary widths).
  explicit SignalValue(llvm::APInt val)
      : value(std::move(val)), isX(false) {}

  /// Construct an X (unknown) value of the given width.
  static SignalValue makeX(uint32_t w = 1) {
    SignalValue sv;
    sv.value = llvm::APInt(w, 0);
    sv.isX = true;
    return sv;
  }

  /// Get the numeric value as uint64_t.
  /// For values wider than 64 bits, returns the lower 64 bits.
  /// Prefer getAPInt() for arbitrary-width values.
  uint64_t getValue() const {
    if (value.getBitWidth() <= 64)
      return value.getZExtValue();
    // For values wider than 64 bits, extract the lower 64 bits
    return value.trunc(64).getZExtValue();
  }

  /// Get the numeric value as APInt (supports arbitrary widths).
  const llvm::APInt &getAPInt() const { return value; }

  /// Get the bit width.
  uint32_t getWidth() const { return value.getBitWidth(); }

  /// Check if the value is unknown (X).
  bool isUnknown() const { return isX; }

  /// Get the LSB (for single-bit edge detection).
  bool getLSB() const { return value[0]; }

  /// Check if this is a 4-state X value using struct encoding.
  /// 4-state encoding uses {value: iN, unknown: iN} flattened to 2N bits:
  /// - Lower N bits: value bits
  /// - Upper N bits: unknown flags (1 = unknown/X)
  /// Returns true if ALL unknown bits are set (fully X value).
  bool isFourStateX() const {
    // First check the explicit isX flag
    if (isX)
      return true;
    // Check for 4-state struct encoding
    uint32_t w = value.getBitWidth();
    // Must have even width >= 2 for 4-state encoding
    if (w < 2 || (w % 2) != 0)
      return false;
    // Check if all upper-half bits (unknown flags) are set
    uint32_t halfWidth = w / 2;
    for (uint32_t i = halfWidth; i < w; ++i) {
      if (!value[i])
        return false;
    }
    return true;
  }

  /// Check if two values are equal.
  /// Handles both explicit isX and 4-state struct encoding.
  bool operator==(const SignalValue &other) const {
    // Check if either value is X (explicit flag or 4-state encoding)
    bool thisIsX = isX || isFourStateXInternal();
    bool otherIsX = other.isX || other.isFourStateXInternal();
    // Two X values are equal
    if (thisIsX && otherIsX)
      return true;
    // X and non-X are not equal
    if (thisIsX || otherIsX)
      return false;
    // Both are known values, compare directly
    return value == other.value;
  }

private:
  /// Internal helper to check 4-state encoding without checking isX flag.
  /// Used by operator== to avoid double-checking isX.
  bool isFourStateXInternal() const {
    uint32_t w = value.getBitWidth();
    // Must have even width >= 2 for 4-state encoding
    if (w < 2 || (w % 2) != 0)
      return false;
    // Check if all upper-half bits (unknown flags) are set
    uint32_t halfWidth = w / 2;
    for (uint32_t i = halfWidth; i < w; ++i) {
      if (!value[i])
        return false;
    }
    return true;
  }

public:

  bool operator!=(const SignalValue &other) const { return !(*this == other); }

  /// Detect edge between old and new values.
  /// Handles both explicit isX and 4-state struct encoding.
  static EdgeType detectEdge(const SignalValue &oldVal,
                             const SignalValue &newVal) {
    // No edge if values are the same (handles X==X via operator==)
    if (oldVal == newVal)
      return EdgeType::None;

    // Check for unknown values using both explicit flag and 4-state encoding
    bool oldIsX = oldVal.isUnknown() || oldVal.isFourStateX();
    bool newIsX = newVal.isUnknown() || newVal.isFourStateX();

    // For unknown values, detect edges conservatively per IEEE 1800.
    if (oldIsX || newIsX) {
      if (oldIsX && newIsX)
        return EdgeType::None;
      if (oldIsX) {
        // X -> known: check if going to 1 (posedge) or 0 (negedge)
        if (!newIsX && newVal.getLSB())
          return EdgeType::Posedge;
        if (!newIsX && !newVal.getLSB())
          return EdgeType::Negedge;
        return EdgeType::AnyEdge;
      }
      // known -> X: use AnyEdge since we can't determine direction
      return EdgeType::AnyEdge;
    }

    bool oldBit = oldVal.getLSB();
    bool newBit = newVal.getLSB();

    if (!oldBit && newBit)
      return EdgeType::Posedge;
    if (oldBit && !newBit)
      return EdgeType::Negedge;
    return EdgeType::AnyEdge;
  }

private:
  llvm::APInt value;
  bool isX;
};

//===----------------------------------------------------------------------===//
// SensitivityEntry - A single entry in a sensitivity list
//===----------------------------------------------------------------------===//

/// Identifier for a signal (can be extended to include hierarchical paths).
using SignalId = uint64_t;

/// A single entry in a sensitivity list, representing a trigger condition.
struct SensitivityEntry {
  /// The signal being monitored.
  SignalId signalId;

  /// The type of edge that triggers this entry (None for level-sensitive).
  EdgeType edge;

  /// Constructor for level-sensitive entry.
  explicit SensitivityEntry(SignalId id)
      : signalId(id), edge(EdgeType::AnyEdge) {}

  /// Constructor for edge-sensitive entry.
  SensitivityEntry(SignalId id, EdgeType e) : signalId(id), edge(e) {}

  bool operator==(const SensitivityEntry &other) const {
    return signalId == other.signalId && edge == other.edge;
  }
};

//===----------------------------------------------------------------------===//
// SensitivityList - Collection of trigger conditions
//===----------------------------------------------------------------------===//

/// A collection of sensitivity entries representing a process's trigger
/// conditions. Corresponds to the @(...) construct in Verilog.
class SensitivityList {
public:
  SensitivityList() = default;

  /// Add a level-sensitive entry (any change triggers).
  void addLevel(SignalId signalId) {
    entries.emplace_back(signalId, EdgeType::AnyEdge);
  }

  /// Add an edge-sensitive entry.
  void addEdge(SignalId signalId, EdgeType edge) {
    entries.emplace_back(signalId, edge);
  }

  /// Add a posedge entry.
  void addPosedge(SignalId signalId) {
    entries.emplace_back(signalId, EdgeType::Posedge);
  }

  /// Add a negedge entry.
  void addNegedge(SignalId signalId) {
    entries.emplace_back(signalId, EdgeType::Negedge);
  }

  /// Check if this sensitivity list is empty.
  bool empty() const { return entries.empty(); }

  /// Get the number of entries.
  size_t size() const { return entries.size(); }

  /// Clear all entries.
  void clear() { entries.clear(); }

  /// Get the entries.
  const llvm::SmallVector<SensitivityEntry, 4> &getEntries() const {
    return entries;
  }

  /// Check if a signal change would trigger this sensitivity list.
  bool isTriggeredBy(SignalId signalId, const SignalValue &oldVal,
                     const SignalValue &newVal) const {
    if (oldVal == newVal)
      return false;

    for (const auto &entry : entries) {
      if (entry.signalId != signalId)
        continue;

      EdgeType actualEdge = SignalValue::detectEdge(oldVal, newVal);

      switch (entry.edge) {
      case EdgeType::None:
        // Level-sensitive, should not be in list normally
        return true;
      case EdgeType::AnyEdge:
        // Any change triggers
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

  /// Mark this as an auto-inferred sensitivity list (like always_comb).
  void setAutoInferred(bool inferred) { autoInferred = inferred; }

  /// Check if this sensitivity list was auto-inferred.
  bool isAutoInferred() const { return autoInferred; }

private:
  llvm::SmallVector<SensitivityEntry, 4> entries;
  bool autoInferred = false;
};

//===----------------------------------------------------------------------===//
// ProcessId - Unique identifier for a process
//===----------------------------------------------------------------------===//

using ProcessId = uint64_t;

/// Invalid process ID constant.
constexpr ProcessId InvalidProcessId = 0;

//===----------------------------------------------------------------------===//
// Process - Represents a simulation process
//===----------------------------------------------------------------------===//

/// A simulation process that can be scheduled and executed.
class Process {
public:
  using ExecuteCallback = std::function<void()>;

  Process(ProcessId id, std::string name, ExecuteCallback callback)
      : id(id), name(std::move(name)), callback(std::move(callback)),
        state(ProcessState::Uninitialized) {}

  /// Get the process ID.
  ProcessId getId() const { return id; }

  /// Get the process name.
  const std::string &getName() const { return name; }

  /// Get the current state.
  ProcessState getState() const { return state; }

  /// Set the process state.
  void setState(ProcessState newState) { state = newState; }

  /// Get the sensitivity list.
  SensitivityList &getSensitivityList() { return sensitivity; }
  const SensitivityList &getSensitivityList() const { return sensitivity; }

  /// Execute the process.
  void execute() {
    if (callback)
      callback();
  }

  /// Update the process callback.
  void setCallback(ExecuteCallback newCallback) {
    callback = std::move(newCallback);
  }

  /// Check if this is a combinational process (auto-inferred sensitivity).
  bool isCombinational() const { return sensitivity.isAutoInferred(); }

  /// Set whether this is a combinational process.
  void setCombinational(bool comb) { sensitivity.setAutoInferred(comb); }

  /// Get the scheduling region preference for this process.
  SchedulingRegion getPreferredRegion() const { return preferredRegion; }

  /// Set the scheduling region preference.
  void setPreferredRegion(SchedulingRegion region) { preferredRegion = region; }

  /// Get the resume time (for delayed processes).
  const SimTime &getResumeTime() const { return resumeTime; }

  /// Set the resume time.
  void setResumeTime(const SimTime &time) { resumeTime = time; }

  /// Mark the process as waiting for a specific event.
  void setWaitingFor(const SensitivityList &waitList) {
    waitingSensitivity = waitList;
    state = ProcessState::Waiting;
  }

  /// Get the sensitivity list this process is waiting for.
  const SensitivityList &getWaitingSensitivity() const {
    return waitingSensitivity;
  }

  /// Clear the waiting state.
  void clearWaiting() {
    waitingSensitivity.clear();
    if (state == ProcessState::Waiting)
      state = ProcessState::Ready;
  }

private:
  ProcessId id;
  std::string name;
  ExecuteCallback callback;
  ProcessState state;
  SensitivityList sensitivity;
  SensitivityList waitingSensitivity;
  SchedulingRegion preferredRegion = SchedulingRegion::Active;
  SimTime resumeTime;
};

//===----------------------------------------------------------------------===//
// DriveStrength - Signal drive strength per IEEE 1800-2017 § 7.9
//===----------------------------------------------------------------------===//

/// Signal drive strength levels (lower values = stronger).
enum class DriveStrength : uint8_t {
  Supply = 0, // Strongest (power/ground)
  Strong = 1, // Normal drive
  Pull = 2,   // Pull-up/pull-down
  Weak = 3,   // Weak drive
  HighZ = 4,  // High impedance (no driver)
};

/// Get the name of a drive strength for debugging.
inline const char *getDriveStrengthName(DriveStrength strength) {
  switch (strength) {
  case DriveStrength::Supply:
    return "supply";
  case DriveStrength::Strong:
    return "strong";
  case DriveStrength::Pull:
    return "pull";
  case DriveStrength::Weak:
    return "weak";
  case DriveStrength::HighZ:
    return "highz";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// SignalDriver - Represents a single driver on a signal
//===----------------------------------------------------------------------===//

/// Represents a single driver contributing to a signal value.
struct SignalDriver {
  /// Unique identifier for this driver (e.g., process ID or drive operation).
  uint64_t driverId;

  /// The value being driven.
  SignalValue value;

  /// Drive strength when driving 0.
  DriveStrength strength0;

  /// Drive strength when driving 1.
  DriveStrength strength1;

  /// Whether this driver is currently active.
  bool active;

  SignalDriver()
      : driverId(0), strength0(DriveStrength::Strong),
        strength1(DriveStrength::Strong), active(false) {}

  SignalDriver(uint64_t id, const SignalValue &val, DriveStrength s0,
               DriveStrength s1)
      : driverId(id), value(val), strength0(s0), strength1(s1), active(true) {}

  /// Get the effective strength based on the driven value.
  DriveStrength getEffectiveStrength() const {
    if (value.isUnknown())
      return DriveStrength::HighZ; // X values treated as weak
    return value.getLSB() ? strength1 : strength0;
  }
};

//===----------------------------------------------------------------------===//
// SignalState - Tracks signal values for edge detection
//===----------------------------------------------------------------------===//

/// Tracks the current and previous values of a signal for edge detection.
/// Also tracks multiple drivers with their strengths for signal resolution.
class SignalState {
public:
  SignalState() = default;
  explicit SignalState(uint32_t width) : current(SignalValue::makeX(width)) {}

  /// Get the current value.
  const SignalValue &getCurrentValue() const { return current; }

  /// Get the previous value.
  const SignalValue &getPreviousValue() const { return previous; }

  /// Update the signal value, returning the detected edge.
  EdgeType updateValue(const SignalValue &newValue) {
    previous = current;
    current = newValue;
    return SignalValue::detectEdge(previous, current);
  }

  /// Check if the signal has changed.
  bool hasChanged() const { return current != previous; }

  /// Get the detected edge type.
  EdgeType getDetectedEdge() const {
    return SignalValue::detectEdge(previous, current);
  }

  /// Add or update a driver with strength information.
  void addOrUpdateDriver(uint64_t driverId, const SignalValue &value,
                         DriveStrength strength0, DriveStrength strength1) {
    // Find existing driver or add new one
    for (auto &driver : drivers) {
      if (driver.driverId == driverId) {
        driver.value = value;
        driver.strength0 = strength0;
        driver.strength1 = strength1;
        driver.active = true;
        return;
      }
    }
    // Add new driver
    drivers.emplace_back(driverId, value, strength0, strength1);
  }

  /// Resolve all drivers to compute the signal value.
  /// Returns the resolved value based on IEEE 1800-2017 strength rules.
  SignalValue resolveDrivers() const {
    if (drivers.empty())
      return current;

    // Count active drivers
    size_t activeCount = 0;
    for (const auto &d : drivers)
      if (d.active)
        ++activeCount;

    if (activeCount == 0)
      return current;

    if (activeCount == 1) {
      // Single driver - use its value directly
      for (const auto &d : drivers)
        if (d.active)
          return d.value;
    }

    // Multiple drivers - need strength-based resolution
    // Separate drivers by their driven value (0 or 1)
    DriveStrength strongestFor0 = DriveStrength::HighZ;
    DriveStrength strongestFor1 = DriveStrength::HighZ;
    bool has0Driver = false;
    bool has1Driver = false;

    for (const auto &d : drivers) {
      if (!d.active)
        continue;
      if (d.value.isUnknown())
        continue; // Skip X drivers

      if (d.value.getLSB()) {
        // Driving 1
        has1Driver = true;
        if (d.strength1 < strongestFor1)
          strongestFor1 = d.strength1;
      } else {
        // Driving 0
        has0Driver = true;
        if (d.strength0 < strongestFor0)
          strongestFor0 = d.strength0;
      }
    }

    // Resolution rules:
    // 1. If only one value is being driven, use that value
    // 2. If both 0 and 1 are driven, stronger one wins
    // 3. If equal strength, result is X (unknown)
    if (has0Driver && !has1Driver) {
      return SignalValue(0, current.getWidth());
    }
    if (has1Driver && !has0Driver) {
      return SignalValue(1, current.getWidth());
    }
    if (has0Driver && has1Driver) {
      if (strongestFor0 < strongestFor1) {
        // 0 driver is stronger
        return SignalValue(0, current.getWidth());
      } else if (strongestFor1 < strongestFor0) {
        // 1 driver is stronger
        return SignalValue(1, current.getWidth());
      } else {
        // Equal strength - result is X
        return SignalValue::makeX(current.getWidth());
      }
    }

    // No active non-X drivers
    return SignalValue::makeX(current.getWidth());
  }

  /// Check if this signal has multiple drivers.
  bool hasMultipleDrivers() const {
    size_t activeCount = 0;
    for (const auto &d : drivers)
      if (d.active)
        ++activeCount;
    return activeCount > 1;
  }

  /// Get the list of drivers (for debugging).
  const std::vector<SignalDriver> &getDrivers() const { return drivers; }

private:
  SignalValue current;
  SignalValue previous;
  std::vector<SignalDriver> drivers;
};

//===----------------------------------------------------------------------===//
// ProcessScheduler - Main process scheduling infrastructure
//===----------------------------------------------------------------------===//

/// The main process scheduler that manages concurrent process execution,
/// sensitivity lists, and event handling following IEEE 1800 semantics.
class ProcessScheduler {
public:
  /// Configuration for the process scheduler.
  struct Config {
    /// Maximum number of delta cycles before declaring infinite loop.
    size_t maxDeltaCycles;

    /// Maximum number of processes that can be registered.
    size_t maxProcesses;

    /// Enable debug output.
    bool debug;

    Config() : maxDeltaCycles(1000), maxProcesses(10000), debug(false) {}
  };

  ProcessScheduler(Config config = Config());
  ~ProcessScheduler();

  //===--------------------------------------------------------------------===//
  // Process Management
  //===--------------------------------------------------------------------===//

  /// Register a new process with the scheduler.
  ProcessId registerProcess(const std::string &name,
                            Process::ExecuteCallback callback);

  /// Unregister a process from the scheduler.
  void unregisterProcess(ProcessId id);

  /// Get a process by ID.
  Process *getProcess(ProcessId id);
  const Process *getProcess(ProcessId id) const;

  /// Get all registered processes.
  const llvm::DenseMap<ProcessId, std::unique_ptr<Process>> &
  getProcesses() const {
    return processes;
  }

  //===--------------------------------------------------------------------===//
  // Sensitivity Management
  //===--------------------------------------------------------------------===//

  /// Set the sensitivity list for a process.
  void setSensitivity(ProcessId id, const SensitivityList &sensitivity);

  /// Add a sensitivity entry to a process.
  void addSensitivity(ProcessId id, SignalId signalId,
                      EdgeType edge = EdgeType::AnyEdge);

  /// Clear the sensitivity list for a process.
  void clearSensitivity(ProcessId id);

  /// Register a signal for tracking.
  SignalId registerSignal(const std::string &name, uint32_t width = 1);

  /// Set the maximum delta cycles to execute at a single time.
  void setMaxDeltaCycles(size_t maxDeltaCycles);

  /// Update a signal value, triggering sensitive processes.
  void updateSignal(SignalId signalId, const SignalValue &newValue);

  /// Update a signal value with strength information for multi-driver resolution.
  /// Uses IEEE 1800-2017 § 7.9 strength-based resolution when multiple drivers
  /// exist on the same signal.
  void updateSignalWithStrength(SignalId signalId, uint64_t driverId,
                                const SignalValue &newValue,
                                DriveStrength strength0,
                                DriveStrength strength1);

  /// Get the current value of a signal.
  const SignalValue &getSignalValue(SignalId signalId) const;

  //===--------------------------------------------------------------------===//
  // Process Execution Control
  //===--------------------------------------------------------------------===//

  /// Mark a process as ready to run.
  void scheduleProcess(ProcessId id, SchedulingRegion region);

  /// Suspend a process until the specified time or events.
  void suspendProcess(ProcessId id, const SimTime &resumeTime);

  /// Suspend a process waiting for specific events.
  void suspendProcessForEvents(ProcessId id,
                               const SensitivityList &waitList);

  /// Terminate a process.
  void terminateProcess(ProcessId id);

  /// Resume a suspended process.
  void resumeProcess(ProcessId id);

  //===--------------------------------------------------------------------===//
  // Delta Cycle Execution
  //===--------------------------------------------------------------------===//

  /// Initialize all processes (run initial blocks).
  void initialize();

  /// Execute one delta cycle.
  /// Returns true if any processes were executed.
  bool executeDeltaCycle();

  /// Execute all delta cycles at the current time.
  /// Returns the number of delta cycles executed.
  size_t executeCurrentTime();

  /// Advance simulation time to the next scheduled event.
  ///
  /// This method first processes any ready processes at the current time,
  /// then checks the EventScheduler for pending delayed events. If events
  /// are found (e.g., from llhd.wait with delay), it advances time to execute
  /// them and resumes any processes that were waiting for that time.
  ///
  /// Returns false if there are no more events or processes to run.
  bool advanceTime();

  /// Check if any processes are ready to run.
  bool hasReadyProcesses() const;

  /// Run the simulation until completion or time limit.
  SimTime runUntil(uint64_t maxTimeFemtoseconds);

  /// Check if the simulation is complete.
  bool isComplete() const;

  //===--------------------------------------------------------------------===//
  // Integration with EventScheduler
  //===--------------------------------------------------------------------===//

  /// Get the underlying event scheduler.
  EventScheduler &getEventScheduler() { return *eventScheduler; }
  const EventScheduler &getEventScheduler() const { return *eventScheduler; }

  /// Get the current simulation time.
  const SimTime &getCurrentTime() const {
    return eventScheduler->getCurrentTime();
  }

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t processesRegistered = 0;
    size_t processesExecuted = 0;
    size_t deltaCyclesExecuted = 0;
    size_t signalUpdates = 0;
    size_t edgesDetected = 0;
    size_t maxDeltaCyclesReached = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Dump signals that changed in the last executed delta cycle.
  void dumpLastDeltaSignals(llvm::raw_ostream &os) const;

  /// Dump processes executed in the last delta cycle.
  void dumpLastDeltaProcesses(llvm::raw_ostream &os) const;

  /// Reset the scheduler to initial state.
  void reset();

private:
  /// Schedule processes triggered by a signal change.
  void triggerSensitiveProcesses(SignalId signalId, const SignalValue &oldVal,
                                 const SignalValue &newVal);

  /// Record a signal change for delta-cycle diagnostics.
  void recordSignalChange(SignalId signalId);

  /// Execute processes in the ready queue for a specific region.
  size_t executeReadyProcesses(SchedulingRegion region);

  /// Generate the next process ID.
  ProcessId nextProcessId();

  /// Generate the next signal ID.
  SignalId nextSignalId();

  Config config;
  std::unique_ptr<EventScheduler> eventScheduler;

  // Process management
  llvm::DenseMap<ProcessId, std::unique_ptr<Process>> processes;
  ProcessId nextProcId = 1;

  // Signal management
  llvm::DenseMap<SignalId, SignalState> signalStates;
  llvm::DenseMap<SignalId, std::string> signalNames;
  SignalId nextSigId = 1;

  // Maps signals to processes sensitive to them
  llvm::DenseMap<SignalId, llvm::SmallVector<ProcessId, 8>>
      signalToProcesses;

  // Ready queues per scheduling region
  std::vector<ProcessId>
      readyQueues[static_cast<size_t>(SchedulingRegion::NumRegions)];

  // Processes waiting for timed events
  std::vector<std::pair<SimTime, ProcessId>> timedWaitQueue;

  // Statistics
  Statistics stats;

  // Initialization flag
  bool initialized = false;

  // Current delta cycle count (for infinite loop detection)
  size_t currentDeltaCount = 0;

  // Signals updated during the current delta cycle.
  llvm::SmallVector<SignalId, 32> signalsChangedThisDelta;
  llvm::DenseSet<SignalId> signalsChangedThisDeltaSet;
  llvm::SmallVector<SignalId, 32> lastDeltaSignals;

  llvm::SmallVector<ProcessId, 32> processesExecutedThisDelta;
  llvm::SmallVector<ProcessId, 32> lastDeltaProcesses;

  // Static signal for unknown values
  static SignalValue unknownSignal;
};

//===----------------------------------------------------------------------===//
// EdgeDetector - Utility class for edge detection
//===----------------------------------------------------------------------===//

/// Utility class for detecting edges on signals.
class EdgeDetector {
public:
  EdgeDetector() = default;

  /// Record a new value and detect the edge.
  EdgeType recordValue(const SignalValue &newValue) {
    EdgeType edge = SignalValue::detectEdge(previousValue, newValue);
    previousValue = newValue;
    return edge;
  }

  /// Get the previous value.
  const SignalValue &getPreviousValue() const { return previousValue; }

  /// Check if a specific edge occurred.
  bool hasPosedge(const SignalValue &newValue) const {
    return SignalValue::detectEdge(previousValue, newValue) == EdgeType::Posedge;
  }

  bool hasNegedge(const SignalValue &newValue) const {
    return SignalValue::detectEdge(previousValue, newValue) == EdgeType::Negedge;
  }

  bool hasAnyEdge(const SignalValue &newValue) const {
    return previousValue != newValue;
  }

private:
  SignalValue previousValue;
};

//===----------------------------------------------------------------------===//
// CombProcessManager - Manages combinational process inference
//===----------------------------------------------------------------------===//

/// Manages automatic sensitivity inference for combinational processes.
class CombProcessManager {
public:
  CombProcessManager(ProcessScheduler &scheduler) : scheduler(scheduler) {}

  /// Register a combinational process that should have auto-inferred sensitivity.
  void registerCombProcess(ProcessId id);

  /// Record a signal read during combinational process execution.
  void recordSignalRead(ProcessId id, SignalId signalId);

  /// Finalize sensitivity inference after process execution.
  void finalizeSensitivity(ProcessId id);

  /// Begin tracking for a process.
  void beginTracking(ProcessId id);

  /// End tracking and update sensitivity.
  void endTracking(ProcessId id);

private:
  ProcessScheduler &scheduler;
  llvm::DenseMap<ProcessId, llvm::SmallVector<SignalId, 8>> inferredSignals;
  ProcessId currentlyTracking = InvalidProcessId;
};

//===----------------------------------------------------------------------===//
// ForkHandle - Handle for managing forked processes
//===----------------------------------------------------------------------===//

/// Unique identifier for a fork group.
using ForkId = uint64_t;

/// Invalid fork ID constant.
constexpr ForkId InvalidForkId = 0;

/// Types of fork join semantics.
enum class ForkJoinType : uint8_t {
  /// Wait for all processes (fork...join)
  Join = 0,

  /// Wait for any one process (fork...join_any)
  JoinAny = 1,

  /// Don't wait (fork...join_none)
  JoinNone = 2,
};

/// Get the name of a fork join type for debugging.
inline const char *getForkJoinTypeName(ForkJoinType type) {
  switch (type) {
  case ForkJoinType::Join:
    return "join";
  case ForkJoinType::JoinAny:
    return "join_any";
  case ForkJoinType::JoinNone:
    return "join_none";
  }
  return "unknown";
}

/// Parse a fork join type from a string.
inline ForkJoinType parseForkJoinType(llvm::StringRef str) {
  if (str == "join_any")
    return ForkJoinType::JoinAny;
  if (str == "join_none")
    return ForkJoinType::JoinNone;
  return ForkJoinType::Join;
}

/// Represents a group of forked processes.
struct ForkGroup {
  /// Unique identifier for this fork group.
  ForkId id;

  /// The type of join semantics.
  ForkJoinType joinType;

  /// Parent process that created this fork.
  ProcessId parentProcess;

  /// Child processes in this fork group.
  llvm::SmallVector<ProcessId, 4> childProcesses;

  /// Number of child processes that have completed.
  size_t completedCount = 0;

  /// Whether the fork has been joined.
  bool joined = false;

  ForkGroup(ForkId id, ForkJoinType joinType, ProcessId parent)
      : id(id), joinType(joinType), parentProcess(parent) {}

  /// Check if the fork is complete based on join type.
  bool isComplete() const {
    switch (joinType) {
    case ForkJoinType::Join:
      return completedCount >= childProcesses.size();
    case ForkJoinType::JoinAny:
      return completedCount >= 1;
    case ForkJoinType::JoinNone:
      return true; // Never waits
    }
    return true;
  }

  /// Check if all child processes have completed.
  bool allComplete() const {
    return completedCount >= childProcesses.size();
  }
};

//===----------------------------------------------------------------------===//
// ForkJoinManager - Manages fork/join process groups
//===----------------------------------------------------------------------===//

/// Manages fork/join process groups and synchronization.
class ForkJoinManager {
public:
  ForkJoinManager(ProcessScheduler &scheduler) : scheduler(scheduler) {}

  /// Create a new fork group.
  ForkId createFork(ProcessId parentProcess, ForkJoinType joinType);

  /// Add a child process to a fork group.
  void addChildToFork(ForkId forkId, ProcessId childProcess);

  /// Mark a child process as completed.
  void markChildComplete(ProcessId childProcess);

  /// Wait for a fork to complete (join semantics).
  /// Returns true if fork is already complete, false if waiting.
  bool join(ForkId forkId);

  /// Wait for any process in a fork to complete (join_any semantics).
  bool joinAny(ForkId forkId);

  /// Get the fork group for a fork ID.
  ForkGroup *getForkGroup(ForkId forkId);
  const ForkGroup *getForkGroup(ForkId forkId) const;

  /// Get the fork group that a child process belongs to.
  ForkGroup *getForkGroupForChild(ProcessId childProcess);

  /// Wait for all child processes of the current process to complete.
  bool waitFork(ProcessId parentProcess);

  /// Disable all child processes of a fork group.
  void disableFork(ForkId forkId);

  /// Disable all child processes of the current process.
  void disableAllForks(ProcessId parentProcess);

  /// Get all fork groups for a parent process.
  llvm::SmallVector<ForkId, 4> getForksForParent(ProcessId parentProcess) const;

private:
  ProcessScheduler &scheduler;
  llvm::DenseMap<ForkId, std::unique_ptr<ForkGroup>> forkGroups;
  llvm::DenseMap<ProcessId, ForkId> childToFork;
  llvm::DenseMap<ProcessId, llvm::SmallVector<ForkId, 4>> parentToForks;
  ForkId nextForkId = 1;

  ForkId getNextForkId() { return nextForkId++; }
};

//===----------------------------------------------------------------------===//
// Semaphore - Counting semaphore for synchronization
//===----------------------------------------------------------------------===//

/// Unique identifier for a semaphore.
using SemaphoreId = uint64_t;

/// Invalid semaphore ID constant.
constexpr SemaphoreId InvalidSemaphoreId = 0;

/// A counting semaphore for process synchronization.
class Semaphore {
public:
  Semaphore(SemaphoreId id, int64_t initialCount)
      : id(id), keyCount(initialCount) {}

  /// Get the semaphore ID.
  SemaphoreId getId() const { return id; }

  /// Get the current key count.
  int64_t getKeyCount() const { return keyCount; }

  /// Try to get keys (non-blocking).
  bool tryGet(int64_t count = 1) {
    if (keyCount >= count) {
      keyCount -= count;
      return true;
    }
    return false;
  }

  /// Put keys back.
  void put(int64_t count = 1) { keyCount += count; }

  /// Add a process to the wait queue.
  void addWaiter(ProcessId id, int64_t requestedKeys) {
    waitQueue.push_back({id, requestedKeys});
  }

  /// Check if there are waiting processes.
  bool hasWaiters() const { return !waitQueue.empty(); }

  /// Try to satisfy the next waiter.
  /// Returns the process ID if satisfied, InvalidProcessId otherwise.
  ProcessId trySatisfyNextWaiter() {
    if (waitQueue.empty())
      return InvalidProcessId;

    auto &waiter = waitQueue.front();
    if (keyCount >= waiter.second) {
      keyCount -= waiter.second;
      ProcessId id = waiter.first;
      waitQueue.erase(waitQueue.begin());
      return id;
    }
    return InvalidProcessId;
  }

private:
  SemaphoreId id;
  int64_t keyCount;
  std::vector<std::pair<ProcessId, int64_t>> waitQueue;
};

//===----------------------------------------------------------------------===//
// Mailbox - Inter-process communication channel
//===----------------------------------------------------------------------===//

/// Unique identifier for a mailbox.
using MailboxId = uint64_t;

/// Invalid mailbox ID constant.
constexpr MailboxId InvalidMailboxId = 0;

/// A mailbox for inter-process message passing.
/// Messages are stored as opaque 64-bit values (handles to actual data).
class Mailbox {
public:
  Mailbox(MailboxId id, int32_t bound = 0) : id(id), boundSize(bound) {}

  /// Get the mailbox ID.
  MailboxId getId() const { return id; }

  /// Get the current message count.
  size_t getMessageCount() const { return messages.size(); }

  /// Check if the mailbox is bounded.
  bool isBounded() const { return boundSize > 0; }

  /// Check if the mailbox is full (for bounded mailboxes).
  bool isFull() const {
    return isBounded() && messages.size() >= static_cast<size_t>(boundSize);
  }

  /// Check if the mailbox is empty.
  bool isEmpty() const { return messages.empty(); }

  /// Try to put a message (non-blocking).
  bool tryPut(uint64_t message) {
    if (isFull())
      return false;
    messages.push_back(message);
    return true;
  }

  /// Put a message (for unbounded or blocking after space available).
  void put(uint64_t message) { messages.push_back(message); }

  /// Try to get a message (non-blocking).
  bool tryGet(uint64_t &message) {
    if (isEmpty())
      return false;
    message = messages.front();
    messages.erase(messages.begin());
    return true;
  }

  /// Peek at the front message without removing.
  bool tryPeek(uint64_t &message) const {
    if (isEmpty())
      return false;
    message = messages.front();
    return true;
  }

  /// Add a process waiting to put.
  void addPutWaiter(ProcessId id, uint64_t message) {
    putWaitQueue.push_back({id, message});
  }

  /// Add a process waiting to get.
  void addGetWaiter(ProcessId id) { getWaitQueue.push_back(id); }

  /// Try to satisfy a get waiter.
  ProcessId trySatisfyGetWaiter(uint64_t &message) {
    if (getWaitQueue.empty() || isEmpty())
      return InvalidProcessId;

    ProcessId id = getWaitQueue.front();
    message = messages.front();
    messages.erase(messages.begin());
    getWaitQueue.erase(getWaitQueue.begin());
    return id;
  }

  /// Try to satisfy a put waiter.
  ProcessId trySatisfyPutWaiter() {
    if (putWaitQueue.empty() || isFull())
      return InvalidProcessId;

    auto waiter = putWaitQueue.front();
    messages.push_back(waiter.second);
    putWaitQueue.erase(putWaitQueue.begin());
    return waiter.first;
  }

private:
  MailboxId id;
  int32_t boundSize;
  std::vector<uint64_t> messages;
  std::vector<std::pair<ProcessId, uint64_t>> putWaitQueue;
  std::vector<ProcessId> getWaitQueue;
};

//===----------------------------------------------------------------------===//
// SyncPrimitivesManager - Manages semaphores and mailboxes
//===----------------------------------------------------------------------===//

/// Manages synchronization primitives (semaphores, mailboxes, events).
class SyncPrimitivesManager {
public:
  SyncPrimitivesManager(ProcessScheduler &scheduler) : scheduler(scheduler) {}

  //===------------------------------------------------------------------===//
  // Semaphore Management
  //===------------------------------------------------------------------===//

  /// Create a new semaphore with initial key count.
  SemaphoreId createSemaphore(int64_t initialCount);

  /// Get keys from a semaphore (blocking).
  void semaphoreGet(SemaphoreId id, ProcessId caller, int64_t count = 1);

  /// Try to get keys from a semaphore (non-blocking).
  bool semaphoreTryGet(SemaphoreId id, int64_t count = 1);

  /// Put keys back to a semaphore.
  void semaphorePut(SemaphoreId id, int64_t count = 1);

  /// Get a semaphore by ID.
  Semaphore *getSemaphore(SemaphoreId id);

  //===------------------------------------------------------------------===//
  // Mailbox Management
  //===------------------------------------------------------------------===//

  /// Create a new mailbox.
  MailboxId createMailbox(int32_t bound = 0);

  /// Put a message into a mailbox (blocking).
  void mailboxPut(MailboxId id, ProcessId caller, uint64_t message);

  /// Try to put a message into a mailbox (non-blocking).
  bool mailboxTryPut(MailboxId id, uint64_t message);

  /// Get a message from a mailbox (blocking).
  void mailboxGet(MailboxId id, ProcessId caller);

  /// Try to get a message from a mailbox (non-blocking).
  bool mailboxTryGet(MailboxId id, uint64_t &message);

  /// Peek at a message in a mailbox.
  bool mailboxPeek(MailboxId id, uint64_t &message);

  /// Get the message count in a mailbox.
  size_t mailboxNum(MailboxId id);

  /// Get a mailbox by ID.
  Mailbox *getMailbox(MailboxId id);

private:
  ProcessScheduler &scheduler;
  llvm::DenseMap<SemaphoreId, std::unique_ptr<Semaphore>> semaphores;
  llvm::DenseMap<MailboxId, std::unique_ptr<Mailbox>> mailboxes;
  SemaphoreId nextSemId = 1;
  MailboxId nextMailboxId = 1;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_PROCESSSCHEDULER_H
