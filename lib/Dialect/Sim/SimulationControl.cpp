//===- SimulationControl.cpp - Simulation control and diagnostics ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements simulation control infrastructure including $finish,
// $stop handling, timeout support, and diagnostic message management.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimulationControl.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "simulation-control"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// SimulationControl Implementation
//===----------------------------------------------------------------------===//

SimulationControl::SimulationControl(Config config)
    : config(std::move(config)), status(SimulationStatus::Running), exitCode(0),
      errorCount(0), warningCount(0), infoCount(0), maxHistorySize(1000) {}

SimulationControl::~SimulationControl() = default;

//===----------------------------------------------------------------------===//
// Simulation Control Operations
//===----------------------------------------------------------------------===//

void SimulationControl::finish(int code) {
  if (status != SimulationStatus::Running)
    return;

  exitCode = code;
  status = SimulationStatus::Finished;
  stats.finishCalls++;

  LLVM_DEBUG(llvm::dbgs() << "SimulationControl: $finish called with code "
                          << code << "\n");

  if (finishCallback)
    finishCallback(code);
}

void SimulationControl::stop() {
  if (status != SimulationStatus::Running)
    return;

  status = SimulationStatus::Stopped;
  stats.stopCalls++;

  LLVM_DEBUG(llvm::dbgs() << "SimulationControl: $stop called\n");

  if (stopCallback)
    stopCallback();
}

void SimulationControl::abort() {
  status = SimulationStatus::Aborted;
  LLVM_DEBUG(llvm::dbgs() << "SimulationControl: Simulation aborted\n");
}

//===----------------------------------------------------------------------===//
// Message Reporting
//===----------------------------------------------------------------------===//

void SimulationControl::info(llvm::StringRef id, llvm::StringRef message) {
  report(MessageSeverity::Info, id, message);
}

void SimulationControl::warning(llvm::StringRef id, llvm::StringRef message) {
  report(MessageSeverity::Warning, id, message);
}

void SimulationControl::error(llvm::StringRef id, llvm::StringRef message) {
  report(MessageSeverity::Error, id, message);
}

void SimulationControl::fatal(llvm::StringRef id, llvm::StringRef message) {
  report(MessageSeverity::Fatal, id, message);
}

void SimulationControl::report(MessageSeverity severity, llvm::StringRef id,
                               llvm::StringRef message) {
  SimulationMessage msg(severity, id, message);
  msg.time = currentTime;
  report(msg);
}

void SimulationControl::report(const SimulationMessage &message) {
  stats.messagesReported++;

  // Apply severity override if configured
  MessageSeverity effectiveSeverity = message.severity;
  auto overrideIt = severityOverrides.find(message.id);
  if (overrideIt != severityOverrides.end()) {
    effectiveSeverity = overrideIt->second;
  }

  // Apply filters
  MessageAction action = applyFilters(message);

  // Update counts based on severity
  switch (effectiveSeverity) {
  case MessageSeverity::Info:
    infoCount++;
    break;
  case MessageSeverity::Warning:
    warningCount++;
    break;
  case MessageSeverity::Error:
    errorCount++;
    break;
  case MessageSeverity::Fatal:
    errorCount++;
    break;
  }

  // Execute action
  switch (action) {
  case MessageAction::Display:
    outputMessage(message);
    break;
  case MessageAction::Count:
    stats.messagesFiltered++;
    // Don't display, just count
    break;
  case MessageAction::Log:
    // Add to history only
    if (maxHistorySize > 0 && messageHistory.size() >= maxHistorySize) {
      messageHistory.erase(messageHistory.begin());
    }
    messageHistory.push_back(message);
    break;
  case MessageAction::Stop:
    outputMessage(message);
    stop();
    break;
  case MessageAction::Exit:
    outputMessage(message);
    finish(1);
    break;
  case MessageAction::Callback:
    if (messageCallback)
      messageCallback(message);
    break;
  }

  // Add to history
  if (maxHistorySize > 0) {
    if (messageHistory.size() >= maxHistorySize) {
      messageHistory.erase(messageHistory.begin());
    }
    messageHistory.push_back(message);
  }

  // Handle fatal messages
  if (effectiveSeverity == MessageSeverity::Fatal) {
    status = SimulationStatus::Fatal;
    if (finishCallback)
      finishCallback(1);
    return;
  }

  // Check limits
  checkLimits();
}

void SimulationControl::outputMessage(const SimulationMessage &message) {
  if (!config.messageOutput)
    return;

  llvm::raw_ostream &os = *config.messageOutput;

  // Format: [TIME] SEVERITY(ID): message
  if (config.showTime) {
    os << "[" << message.time.realTime << "fs";
    if (message.time.deltaStep > 0)
      os << " d" << message.time.deltaStep;
    os << "] ";
  }

  os << getMessageSeverityName(message.severity) << "(" << message.id << "): "
     << message.text;

  if (!message.fileName.empty()) {
    os << " [" << message.fileName;
    if (message.lineNumber > 0)
      os << ":" << message.lineNumber;
    os << "]";
  }

  if (!message.scope.empty()) {
    os << " @ " << message.scope;
  }

  os << "\n";
}

//===----------------------------------------------------------------------===//
// Message Filtering
//===----------------------------------------------------------------------===//

void SimulationControl::addFilter(const MessageFilter &filter) {
  filters.push_back(filter);
}

void SimulationControl::removeFilter(llvm::StringRef idPattern) {
  filters.erase(std::remove_if(filters.begin(), filters.end(),
                               [&](const MessageFilter &f) {
                                 return f.idPattern == idPattern.str();
                               }),
                filters.end());
}

void SimulationControl::clearFilters() { filters.clear(); }

void SimulationControl::setMessageAction(llvm::StringRef id,
                                         MessageAction action) {
  idActions[id] = action;
}

void SimulationControl::setSeverityOverride(llvm::StringRef id,
                                            MessageSeverity severity) {
  severityOverrides[id] = severity;
}

MessageAction
SimulationControl::applyFilters(const SimulationMessage &message) {
  // First check direct ID action mapping
  auto actionIt = idActions.find(message.id);
  if (actionIt != idActions.end())
    return actionIt->second;

  // Then apply filters in order
  for (const auto &filter : filters) {
    // Simple pattern matching (exact match or wildcard)
    bool matches = false;
    if (filter.idPattern == "*") {
      matches = true;
    } else if (filter.idPattern.back() == '*') {
      // Prefix match
      llvm::StringRef prefix(filter.idPattern.data(),
                             filter.idPattern.size() - 1);
      matches = llvm::StringRef(message.id).starts_with(prefix);
    } else {
      matches = filter.idPattern == message.id;
    }

    // Check severity if specified
    if (matches && filter.severity.has_value()) {
      matches = message.severity == filter.severity.value();
    }

    if (matches) {
      if (filter.stopOnMatch)
        return filter.action;
    }
  }

  // Default action: display
  return MessageAction::Display;
}

//===----------------------------------------------------------------------===//
// Error/Warning Limits
//===----------------------------------------------------------------------===//

void SimulationControl::resetCounts() {
  errorCount = 0;
  warningCount = 0;
  infoCount = 0;
}

void SimulationControl::checkLimits() {
  if (config.maxErrors > 0 && errorCount >= config.maxErrors) {
    LLVM_DEBUG(llvm::dbgs() << "SimulationControl: Error limit reached ("
                            << errorCount << ")\n");
    status = SimulationStatus::ErrorLimit;
    if (finishCallback)
      finishCallback(1);
  }

  if (config.maxWarnings > 0 && warningCount >= config.maxWarnings) {
    LLVM_DEBUG(llvm::dbgs() << "SimulationControl: Warning limit reached ("
                            << warningCount << ")\n");
    // Warnings don't terminate by default, but we can log it
  }
}

//===----------------------------------------------------------------------===//
// Watchdog
//===----------------------------------------------------------------------===//

void SimulationControl::enableWatchdog(uint64_t femtoseconds) {
  watchdog.enable(femtoseconds);
  watchdog.setTimeoutCallback([this]() {
    error("WATCHDOG", "Watchdog timeout - simulation appears hung");
    status = SimulationStatus::Timeout;
  });
}

//===----------------------------------------------------------------------===//
// UVM Compatibility
//===----------------------------------------------------------------------===//

void SimulationControl::uvmReportInfo(llvm::StringRef id,
                                      llvm::StringRef message, int verbosity) {
  if (shouldDisplay(verbosity)) {
    info(id, message);
  }
}

void SimulationControl::uvmReportWarning(llvm::StringRef id,
                                         llvm::StringRef message) {
  warning(id, message);
}

void SimulationControl::uvmReportError(llvm::StringRef id,
                                       llvm::StringRef message) {
  error(id, message);
}

void SimulationControl::uvmReportFatal(llvm::StringRef id,
                                       llvm::StringRef message) {
  fatal(id, message);
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

void SimulationControl::reset() {
  status = SimulationStatus::Running;
  exitCode = 0;
  errorCount = 0;
  warningCount = 0;
  infoCount = 0;
  messageHistory.clear();
  filters.clear();
  idActions.clear();
  severityOverrides.clear();
  watchdog.disable();
  currentTime = SimTime();
  stats = Statistics();

  LLVM_DEBUG(llvm::dbgs() << "SimulationControl: Reset\n");
}

//===----------------------------------------------------------------------===//
// SimulationControlGuard Implementation
//===----------------------------------------------------------------------===//

SimulationControlGuard::SimulationControlGuard(SimulationControl &ctrl,
                                               llvm::StringRef pattern)
    : control(ctrl), pattern(pattern.str()) {
  // Suppress messages matching pattern
  control.addFilter(MessageFilter(pattern, MessageAction::Count));
}

SimulationControlGuard::SimulationControlGuard(SimulationControl &ctrl,
                                               llvm::StringRef pattern,
                                               MessageAction action)
    : control(ctrl), pattern(pattern.str()) {
  control.addFilter(MessageFilter(pattern, action));
}

SimulationControlGuard::~SimulationControlGuard() {
  control.removeFilter(pattern);
}
