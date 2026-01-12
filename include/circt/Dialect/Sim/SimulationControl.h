//===- SimulationControl.h - Simulation control and diagnostics -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines simulation control infrastructure including:
// - $finish, $stop handling
// - Timeout and watchdog support
// - Error/warning counting and limits
// - Message reporting system
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMULATIONCONTROL_H
#define CIRCT_DIALECT_SIM_SIMULATIONCONTROL_H

#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace sim {

// Forward declarations
class ProcessScheduler;

//===----------------------------------------------------------------------===//
// SimulationStatus - Simulation completion status
//===----------------------------------------------------------------------===//

/// Status of simulation completion.
enum class SimulationStatus : uint8_t {
  /// Simulation is still running.
  Running = 0,

  /// Simulation completed normally (no events, or time limit reached).
  Completed = 1,

  /// Simulation stopped by $stop.
  Stopped = 2,

  /// Simulation finished by $finish.
  Finished = 3,

  /// Simulation terminated due to error limit.
  ErrorLimit = 4,

  /// Simulation terminated due to timeout.
  Timeout = 5,

  /// Simulation terminated due to fatal error.
  Fatal = 6,

  /// Simulation aborted by user or external signal.
  Aborted = 7
};

/// Get the name of a simulation status.
inline const char *getSimulationStatusName(SimulationStatus status) {
  switch (status) {
  case SimulationStatus::Running:
    return "running";
  case SimulationStatus::Completed:
    return "completed";
  case SimulationStatus::Stopped:
    return "stopped";
  case SimulationStatus::Finished:
    return "finished";
  case SimulationStatus::ErrorLimit:
    return "error_limit";
  case SimulationStatus::Timeout:
    return "timeout";
  case SimulationStatus::Fatal:
    return "fatal";
  case SimulationStatus::Aborted:
    return "aborted";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// MessageSeverity - Severity levels for diagnostic messages
//===----------------------------------------------------------------------===//

/// Severity levels for simulation messages.
enum class MessageSeverity : uint8_t {
  /// Informational message.
  Info = 0,

  /// Warning message.
  Warning = 1,

  /// Error message.
  Error = 2,

  /// Fatal error - terminates simulation immediately.
  Fatal = 3
};

/// Get the name of a message severity.
inline const char *getMessageSeverityName(MessageSeverity severity) {
  switch (severity) {
  case MessageSeverity::Info:
    return "INFO";
  case MessageSeverity::Warning:
    return "WARNING";
  case MessageSeverity::Error:
    return "ERROR";
  case MessageSeverity::Fatal:
    return "FATAL";
  }
  return "UNKNOWN";
}

//===----------------------------------------------------------------------===//
// SimulationMessage - A diagnostic message
//===----------------------------------------------------------------------===//

/// Represents a diagnostic message from the simulation.
struct SimulationMessage {
  /// Severity of the message.
  MessageSeverity severity;

  /// Message ID (for filtering).
  std::string id;

  /// The message text.
  std::string text;

  /// File name where the message originated.
  std::string fileName;

  /// Line number where the message originated.
  int lineNumber;

  /// Simulation time when the message was generated.
  SimTime time;

  /// Component/scope that generated the message.
  std::string scope;

  SimulationMessage(MessageSeverity sev, llvm::StringRef id,
                    llvm::StringRef text)
      : severity(sev), id(id.str()), text(text.str()), lineNumber(0) {}

  SimulationMessage(MessageSeverity sev, llvm::StringRef id,
                    llvm::StringRef text, llvm::StringRef file, int line)
      : severity(sev), id(id.str()), text(text.str()), fileName(file.str()),
        lineNumber(line) {}
};

//===----------------------------------------------------------------------===//
// MessageAction - Actions to take on messages
//===----------------------------------------------------------------------===//

/// Actions that can be taken when a message is reported.
enum class MessageAction : uint8_t {
  /// Display the message.
  Display = 0,

  /// Count the message (for limits).
  Count = 1,

  /// Log the message to file.
  Log = 2,

  /// Stop simulation ($stop).
  Stop = 3,

  /// Exit simulation ($finish).
  Exit = 4,

  /// Call a callback function.
  Callback = 5
};

//===----------------------------------------------------------------------===//
// MessageFilter - Filter for message handling
//===----------------------------------------------------------------------===//

/// Filter configuration for message handling.
struct MessageFilter {
  /// Message ID pattern (can be wildcard).
  std::string idPattern;

  /// Severity to match (or all severities if not set).
  std::optional<MessageSeverity> severity;

  /// Action to take when filter matches.
  MessageAction action;

  /// Whether to stop processing other filters.
  bool stopOnMatch;

  MessageFilter(llvm::StringRef pattern, MessageAction action)
      : idPattern(pattern.str()), action(action), stopOnMatch(true) {}

  MessageFilter(llvm::StringRef pattern, MessageSeverity sev,
                MessageAction action)
      : idPattern(pattern.str()), severity(sev), action(action),
        stopOnMatch(true) {}
};

//===----------------------------------------------------------------------===//
// Watchdog - Timeout monitor for simulation hangs
//===----------------------------------------------------------------------===//

/// Watchdog timer for detecting simulation hangs.
class Watchdog {
public:
  using TimeoutCallback = std::function<void()>;

  Watchdog() : enabled(false), timeout(0), lastActivity(0) {}

  /// Enable the watchdog with a timeout in femtoseconds.
  void enable(uint64_t timeoutFs) {
    timeout = timeoutFs;
    enabled = true;
    lastActivity = 0;
  }

  /// Disable the watchdog.
  void disable() { enabled = false; }

  /// Check if the watchdog is enabled.
  bool isEnabled() const { return enabled; }

  /// Get the timeout value.
  uint64_t getTimeout() const { return timeout; }

  /// Reset the watchdog (call when activity occurs).
  void kick(uint64_t currentTime) { lastActivity = currentTime; }

  /// Check if the watchdog has timed out.
  bool hasTimedOut(uint64_t currentTime) const {
    if (!enabled)
      return false;
    return (currentTime - lastActivity) > timeout;
  }

  /// Set the timeout callback.
  void setTimeoutCallback(TimeoutCallback callback) {
    timeoutCallback = std::move(callback);
  }

  /// Trigger the timeout.
  void triggerTimeout() {
    if (timeoutCallback)
      timeoutCallback();
  }

private:
  bool enabled;
  uint64_t timeout;
  uint64_t lastActivity;
  TimeoutCallback timeoutCallback;
};

//===----------------------------------------------------------------------===//
// SimulationControl - Main simulation control class
//===----------------------------------------------------------------------===//

/// Main simulation control class that handles $finish, $stop, timeouts,
/// and diagnostic message management.
class SimulationControl {
public:
  /// Configuration for simulation control.
  struct Config {
    /// Maximum number of errors before termination (0 = unlimited).
    size_t maxErrors;

    /// Maximum number of warnings before termination (0 = unlimited).
    size_t maxWarnings;

    /// Global simulation timeout in femtoseconds (0 = no timeout).
    uint64_t globalTimeout;

    /// Watchdog timeout in femtoseconds (0 = disabled).
    uint64_t watchdogTimeout;

    /// Default verbosity level (0-4).
    int verbosity;

    /// Whether to show simulation time in messages.
    bool showTime;

    /// Output stream for messages.
    llvm::raw_ostream *messageOutput;

    Config()
        : maxErrors(0), maxWarnings(0), globalTimeout(0), watchdogTimeout(0),
          verbosity(2), showTime(true), messageOutput(&llvm::errs()) {}
  };

  SimulationControl(Config config = Config());
  ~SimulationControl();

  //===--------------------------------------------------------------------===//
  // Simulation Control Operations
  //===--------------------------------------------------------------------===//

  /// Request simulation finish (like $finish).
  void finish(int exitCode = 0);

  /// Request simulation stop (like $stop).
  void stop();

  /// Abort simulation immediately.
  void abort();

  /// Get the current simulation status.
  SimulationStatus getStatus() const { return status; }

  /// Check if simulation should continue.
  bool shouldContinue() const { return status == SimulationStatus::Running; }

  /// Get the exit code.
  int getExitCode() const { return exitCode; }

  /// Set the exit code.
  void setExitCode(int code) { exitCode = code; }

  //===--------------------------------------------------------------------===//
  // Message Reporting
  //===--------------------------------------------------------------------===//

  /// Report an informational message.
  void info(llvm::StringRef id, llvm::StringRef message);

  /// Report a warning.
  void warning(llvm::StringRef id, llvm::StringRef message);

  /// Report an error.
  void error(llvm::StringRef id, llvm::StringRef message);

  /// Report a fatal error.
  void fatal(llvm::StringRef id, llvm::StringRef message);

  /// Report a message with full details.
  void report(const SimulationMessage &message);

  /// Report a message with severity.
  void report(MessageSeverity severity, llvm::StringRef id,
              llvm::StringRef message);

  //===--------------------------------------------------------------------===//
  // Message Filtering
  //===--------------------------------------------------------------------===//

  /// Add a message filter.
  void addFilter(const MessageFilter &filter);

  /// Remove filters matching a pattern.
  void removeFilter(llvm::StringRef idPattern);

  /// Clear all filters.
  void clearFilters();

  /// Set the action for a message ID.
  void setMessageAction(llvm::StringRef id, MessageAction action);

  /// Set severity override for a message ID.
  void setSeverityOverride(llvm::StringRef id, MessageSeverity severity);

  //===--------------------------------------------------------------------===//
  // Message History
  //===--------------------------------------------------------------------===//

  /// Get the message history.
  const std::vector<SimulationMessage> &getMessageHistory() const {
    return messageHistory;
  }

  /// Clear the message history.
  void clearMessageHistory() { messageHistory.clear(); }

  /// Set maximum history size (0 = unlimited).
  void setMaxHistorySize(size_t size) { maxHistorySize = size; }

  //===--------------------------------------------------------------------===//
  // Error/Warning Counts
  //===--------------------------------------------------------------------===//

  /// Get the error count.
  size_t getErrorCount() const { return errorCount; }

  /// Get the warning count.
  size_t getWarningCount() const { return warningCount; }

  /// Get the info count.
  size_t getInfoCount() const { return infoCount; }

  /// Reset message counts.
  void resetCounts();

  /// Set maximum errors before termination.
  void setMaxErrors(size_t max) { config.maxErrors = max; }

  /// Set maximum warnings before termination.
  void setMaxWarnings(size_t max) { config.maxWarnings = max; }

  //===--------------------------------------------------------------------===//
  // Timeout and Watchdog
  //===--------------------------------------------------------------------===//

  /// Set the global simulation timeout.
  void setGlobalTimeout(uint64_t femtoseconds) {
    config.globalTimeout = femtoseconds;
  }

  /// Get the global timeout.
  uint64_t getGlobalTimeout() const { return config.globalTimeout; }

  /// Check if global timeout has been reached.
  bool hasTimedOut(const SimTime &currentTime) const {
    if (config.globalTimeout == 0)
      return false;
    return currentTime.realTime >= config.globalTimeout;
  }

  /// Get the watchdog.
  Watchdog &getWatchdog() { return watchdog; }
  const Watchdog &getWatchdog() const { return watchdog; }

  /// Enable watchdog with specified timeout.
  void enableWatchdog(uint64_t femtoseconds);

  /// Disable the watchdog.
  void disableWatchdog() { watchdog.disable(); }

  /// Kick the watchdog (signal activity).
  void kickWatchdog(uint64_t currentTime) { watchdog.kick(currentTime); }

  //===--------------------------------------------------------------------===//
  // Callbacks
  //===--------------------------------------------------------------------===//

  /// Set callback for finish events.
  void setFinishCallback(std::function<void(int)> callback) {
    finishCallback = std::move(callback);
  }

  /// Set callback for stop events.
  void setStopCallback(std::function<void()> callback) {
    stopCallback = std::move(callback);
  }

  /// Set callback for message events.
  void setMessageCallback(std::function<void(const SimulationMessage &)> callback) {
    messageCallback = std::move(callback);
  }

  //===--------------------------------------------------------------------===//
  // Verbosity Control
  //===--------------------------------------------------------------------===//

  /// Set the verbosity level (0-4).
  void setVerbosity(int level) { config.verbosity = level; }

  /// Get the verbosity level.
  int getVerbosity() const { return config.verbosity; }

  /// Check if a message at the given verbosity level should be displayed.
  bool shouldDisplay(int messageVerbosity) const {
    return messageVerbosity <= config.verbosity;
  }

  //===--------------------------------------------------------------------===//
  // Output Control
  //===--------------------------------------------------------------------===//

  /// Set the output stream for messages.
  void setOutputStream(llvm::raw_ostream &os) { config.messageOutput = &os; }

  /// Enable/disable showing simulation time in messages.
  void setShowTime(bool show) { config.showTime = show; }

  /// Set the current simulation time (for message timestamps).
  void setCurrentTime(const SimTime &time) { currentTime = time; }

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  struct Statistics {
    size_t messagesReported = 0;
    size_t messagesFiltered = 0;
    size_t finishCalls = 0;
    size_t stopCalls = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Reset the simulation control.
  void reset();

  //===--------------------------------------------------------------------===//
  // UVM Compatibility
  //===--------------------------------------------------------------------===//

  /// UVM report methods (for compatibility).
  void uvmReportInfo(llvm::StringRef id, llvm::StringRef message,
                     int verbosity = 2);
  void uvmReportWarning(llvm::StringRef id, llvm::StringRef message);
  void uvmReportError(llvm::StringRef id, llvm::StringRef message);
  void uvmReportFatal(llvm::StringRef id, llvm::StringRef message);

private:
  /// Format and output a message.
  void outputMessage(const SimulationMessage &message);

  /// Check if message limits have been reached.
  void checkLimits();

  /// Apply filters to a message and get the action.
  MessageAction applyFilters(const SimulationMessage &message);

  Config config;
  SimulationStatus status;
  int exitCode;

  // Message counting
  size_t errorCount;
  size_t warningCount;
  size_t infoCount;

  // Message history
  std::vector<SimulationMessage> messageHistory;
  size_t maxHistorySize;

  // Message filters
  std::vector<MessageFilter> filters;
  llvm::StringMap<MessageAction> idActions;
  llvm::StringMap<MessageSeverity> severityOverrides;

  // Timeout management
  Watchdog watchdog;
  SimTime currentTime;

  // Callbacks
  std::function<void(int)> finishCallback;
  std::function<void()> stopCallback;
  std::function<void(const SimulationMessage &)> messageCallback;

  // Statistics
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// SimulationControlGuard - RAII guard for message filtering
//===----------------------------------------------------------------------===//

/// RAII guard for temporarily modifying message handling.
class SimulationControlGuard {
public:
  /// Create a guard that temporarily suppresses messages matching pattern.
  SimulationControlGuard(SimulationControl &ctrl, llvm::StringRef pattern);

  /// Create a guard that temporarily changes message action.
  SimulationControlGuard(SimulationControl &ctrl, llvm::StringRef pattern,
                         MessageAction action);

  ~SimulationControlGuard();

  // Non-copyable
  SimulationControlGuard(const SimulationControlGuard &) = delete;
  SimulationControlGuard &operator=(const SimulationControlGuard &) = delete;

private:
  SimulationControl &control;
  std::string pattern;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_SIMULATIONCONTROL_H
