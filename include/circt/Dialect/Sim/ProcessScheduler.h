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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
class SignalValue {
public:
  /// Default constructor creates an X (unknown) value.
  SignalValue() : value(0), width(1), isX(true) {}

  /// Construct from an integer value.
  explicit SignalValue(uint64_t val, uint32_t w = 1)
      : value(val), width(w), isX(false) {}

  /// Construct an X (unknown) value of the given width.
  static SignalValue makeX(uint32_t w = 1) {
    SignalValue sv;
    sv.width = w;
    sv.isX = true;
    return sv;
  }

  /// Get the numeric value.
  uint64_t getValue() const { return value; }

  /// Get the bit width.
  uint32_t getWidth() const { return width; }

  /// Check if the value is unknown (X).
  bool isUnknown() const { return isX; }

  /// Get the LSB (for single-bit edge detection).
  bool getLSB() const { return (value & 1) != 0; }

  /// Check if two values are equal.
  bool operator==(const SignalValue &other) const {
    if (isX && other.isX)
      return true;
    if (isX || other.isX)
      return false;
    return value == other.value && width == other.width;
  }

  bool operator!=(const SignalValue &other) const { return !(*this == other); }

  /// Detect edge between old and new values.
  static EdgeType detectEdge(const SignalValue &oldVal,
                             const SignalValue &newVal) {
    // No edge if values are the same
    if (oldVal == newVal)
      return EdgeType::None;

    // For unknown values, we can detect edges conservatively
    bool oldBit = oldVal.isUnknown() ? false : oldVal.getLSB();
    bool newBit = newVal.isUnknown() ? false : newVal.getLSB();

    if (!oldBit && newBit)
      return EdgeType::Posedge;
    if (oldBit && !newBit)
      return EdgeType::Negedge;
    return EdgeType::AnyEdge;
  }

private:
  uint64_t value;
  uint32_t width;
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
// SignalState - Tracks signal values for edge detection
//===----------------------------------------------------------------------===//

/// Tracks the current and previous values of a signal for edge detection.
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

private:
  SignalValue current;
  SignalValue previous;
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
    size_t maxDeltaCycles = 1000;

    /// Maximum number of processes that can be registered.
    size_t maxProcesses = 10000;

    /// Enable debug output.
    bool debug = false;
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

  /// Update a signal value, triggering sensitive processes.
  void updateSignal(SignalId signalId, const SignalValue &newValue);

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

  /// Step to the next event time.
  /// Returns false if there are no more events.
  bool advanceTime();

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

  /// Reset the scheduler to initial state.
  void reset();

private:
  /// Schedule processes triggered by a signal change.
  void triggerSensitiveProcesses(SignalId signalId, const SignalValue &oldVal,
                                 const SignalValue &newVal);

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

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_PROCESSSCHEDULER_H
