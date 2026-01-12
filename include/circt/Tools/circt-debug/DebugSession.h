//===- DebugSession.h - CIRCT Debug Session ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the debug session management for the CIRCT
// interactive debugger, including simulation control and command processing.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_DEBUG_DEBUGSESSION_H
#define CIRCT_TOOLS_CIRCT_DEBUG_DEBUGSESSION_H

#include "circt/Tools/circt-debug/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace debug {

//===----------------------------------------------------------------------===//
// Debug Session Configuration
//===----------------------------------------------------------------------===//

/// Configuration options for a debug session.
struct DebugConfig {
  /// Path to the design file.
  std::string designFile;

  /// Top module name.
  std::string topModule;

  /// Output directory for waveforms and logs.
  std::string outputDir = ".";

  /// VCD file path (empty = no VCD output).
  std::string vcdFile;

  /// Enable automatic waveform capture around breakpoints.
  bool captureWaveformsOnBreak = false;

  /// Number of cycles to capture before a breakpoint.
  unsigned waveformPreTrigger = 10;

  /// Number of cycles to capture after a breakpoint.
  unsigned waveformPostTrigger = 10;

  /// Maximum simulation cycles (0 = unlimited).
  uint64_t maxCycles = 0;

  /// Maximum simulation time.
  SimTime maxTime;

  /// Verbose output.
  bool verbose = false;
};

//===----------------------------------------------------------------------===//
// Simulation Backend Interface
//===----------------------------------------------------------------------===//

/// Abstract interface for simulation backends.
class SimulationBackend {
public:
  virtual ~SimulationBackend() = default;

  /// Initialize the simulation.
  virtual bool initialize(const DebugConfig &config) = 0;

  /// Reset the simulation to initial state.
  virtual bool reset() = 0;

  /// Step one delta cycle.
  virtual bool stepDelta() = 0;

  /// Step to the next clock edge.
  virtual bool stepClock() = 0;

  /// Run for specified number of clock cycles.
  virtual bool run(uint64_t cycles) = 0;

  /// Run until a time limit.
  virtual bool runUntil(const SimTime &time) = 0;

  /// Get current simulation state.
  virtual SimState &getState() = 0;
  virtual const SimState &getState() const = 0;

  /// Set a signal value (for forcing).
  virtual bool setSignal(StringRef path, const SignalValue &value) = 0;

  /// Release a forced signal.
  virtual bool releaseSignal(StringRef path) = 0;

  /// Check if simulation has finished.
  virtual bool isFinished() const = 0;

  /// Get error message if any operation failed.
  virtual StringRef getLastError() const = 0;
};

//===----------------------------------------------------------------------===//
// Stop Reason
//===----------------------------------------------------------------------===//

/// Reason why simulation stopped.
enum class StopReason {
  None,          // Not stopped
  Breakpoint,    // Hit a breakpoint
  Watchpoint,    // Signal changed (watched)
  Step,          // Completed step command
  Finished,      // Simulation finished
  Error,         // Error occurred
  UserInterrupt, // User interrupted (Ctrl+C)
  MaxCycles,     // Reached max cycles limit
  MaxTime        // Reached max time limit
};

/// Get string representation of stop reason.
StringRef toString(StopReason reason);

//===----------------------------------------------------------------------===//
// Debug Session
//===----------------------------------------------------------------------===//

/// Manages an interactive debug session.
class DebugSession {
public:
  DebugSession(std::unique_ptr<SimulationBackend> backend,
               const DebugConfig &config);
  ~DebugSession();

  /// Start the debug session (initialize simulation).
  bool start();

  /// Stop the debug session.
  void stop();

  //==========================================================================
  // Simulation Control
  //==========================================================================

  /// Run simulation for a number of clock cycles.
  StopReason run(uint64_t cycles = 0);

  /// Step one clock cycle.
  StopReason step(uint64_t n = 1);

  /// Step one delta cycle.
  StopReason stepDelta();

  /// Continue until breakpoint or end.
  StopReason continueExec();

  /// Run until a specific time.
  StopReason runUntil(const SimTime &time);

  /// Run until a specific cycle.
  StopReason runUntilCycle(uint64_t cycle);

  /// Reset simulation to initial state.
  bool reset();

  //==========================================================================
  // Breakpoint Management
  //==========================================================================

  BreakpointManager &getBreakpointManager() { return breakpointMgr; }
  const BreakpointManager &getBreakpointManager() const {
    return breakpointMgr;
  }

  //==========================================================================
  // Signal Access
  //==========================================================================

  /// Get a signal value.
  EvalResult getSignal(StringRef path) const;

  /// Set a signal value (force).
  bool setSignal(StringRef path, const SignalValue &value);

  /// Release a forced signal.
  bool releaseSignal(StringRef path);

  /// List signals in current scope.
  std::vector<SignalInfo> listSignals() const;

  /// List signals matching a pattern.
  std::vector<SignalInfo> findSignals(StringRef pattern) const;

  //==========================================================================
  // Scope Navigation
  //==========================================================================

  /// Get current scope path.
  std::string getCurrentScope() const;

  /// Change to a different scope.
  bool changeScope(StringRef path);

  /// Go up one level in scope hierarchy.
  bool scopeUp();

  /// List child scopes.
  std::vector<std::string> listScopes() const;

  //==========================================================================
  // Expression Evaluation
  //==========================================================================

  /// Evaluate an expression.
  EvalResult evaluate(StringRef expr) const;

  //==========================================================================
  // Waveform Output
  //==========================================================================

  /// Start VCD dumping.
  bool startVCDDump(StringRef filename);

  /// Stop VCD dumping.
  void stopVCDDump();

  /// Dump current signal values to VCD.
  void dumpVCDValues();

  /// Save a waveform snippet around the current time.
  bool saveWaveformSnippet(StringRef filename, unsigned preTime = 10,
                           unsigned postTime = 10);

  //==========================================================================
  // State Access
  //==========================================================================

  /// Get simulation state.
  SimState &getState();
  const SimState &getState() const;

  /// Get configuration.
  const DebugConfig &getConfig() const { return config; }

  /// Get last stop reason.
  StopReason getLastStopReason() const { return lastStopReason; }

  /// Get triggered breakpoints from last stop.
  const llvm::SmallVector<Breakpoint *> &getTriggeredBreakpoints() const {
    return triggeredBreakpoints;
  }

  /// Is session running?
  bool isRunning() const { return running; }

  /// Is simulation finished?
  bool isFinished() const;

  //==========================================================================
  // Callbacks
  //==========================================================================

  /// Callback types for session events.
  using StopCallback = std::function<void(StopReason, DebugSession &)>;
  using BreakCallback = std::function<void(Breakpoint &, DebugSession &)>;
  using WatchCallback = std::function<void(Watchpoint &, DebugSession &)>;
  using OutputCallback = std::function<void(StringRef)>;

  /// Set callbacks.
  void setStopCallback(StopCallback cb) { stopCallback = std::move(cb); }
  void setBreakCallback(BreakCallback cb) { breakCallback = std::move(cb); }
  void setWatchCallback(WatchCallback cb) { watchCallback = std::move(cb); }
  void setOutputCallback(OutputCallback cb) { outputCallback = std::move(cb); }

private:
  /// Check limits and breakpoints, update state.
  StopReason checkStopConditions();

  /// Handle a stop.
  void handleStop(StopReason reason);

  /// Output message.
  void output(StringRef msg);

  std::unique_ptr<SimulationBackend> backend;
  DebugConfig config;
  BreakpointManager breakpointMgr;

  bool running = false;
  StopReason lastStopReason = StopReason::None;
  llvm::SmallVector<Breakpoint *> triggeredBreakpoints;

  // VCD output state
  std::unique_ptr<llvm::raw_fd_ostream> vcdStream;
  bool vcdActive = false;

  // Callbacks
  StopCallback stopCallback;
  BreakCallback breakCallback;
  WatchCallback watchCallback;
  OutputCallback outputCallback;
};

//===----------------------------------------------------------------------===//
// Debug Command Interface
//===----------------------------------------------------------------------===//

/// Result of executing a debug command.
struct CommandResult {
  bool success = true;
  std::string output;
  bool shouldContinue = true; // false = exit debugger

  static CommandResult ok(StringRef msg = "") {
    return CommandResult{true, msg.str(), true};
  }
  static CommandResult error(StringRef msg) {
    return CommandResult{false, msg.str(), true};
  }
  static CommandResult quit() { return CommandResult{true, "", false}; }
};

/// Parses and executes debug commands.
class CommandProcessor {
public:
  CommandProcessor(DebugSession &session, llvm::raw_ostream &out,
                   llvm::raw_ostream &err);

  /// Execute a single command.
  CommandResult execute(StringRef command);

  /// Get help text for all commands.
  std::string getHelp() const;

  /// Get help for a specific command.
  std::string getHelp(StringRef command) const;

  /// Set the prompt string.
  void setPrompt(StringRef p) { prompt = p.str(); }
  StringRef getPrompt() const { return prompt; }

  /// Enable/disable command echo.
  void setEcho(bool e) { echo = e; }

private:
  /// Command handlers.
  CommandResult cmdRun(const std::vector<std::string> &args);
  CommandResult cmdStep(const std::vector<std::string> &args);
  CommandResult cmdStepDelta(const std::vector<std::string> &args);
  CommandResult cmdContinue(const std::vector<std::string> &args);
  CommandResult cmdBreak(const std::vector<std::string> &args);
  CommandResult cmdWatch(const std::vector<std::string> &args);
  CommandResult cmdDelete(const std::vector<std::string> &args);
  CommandResult cmdEnable(const std::vector<std::string> &args);
  CommandResult cmdDisable(const std::vector<std::string> &args);
  CommandResult cmdPrint(const std::vector<std::string> &args);
  CommandResult cmdSet(const std::vector<std::string> &args);
  CommandResult cmdInfo(const std::vector<std::string> &args);
  CommandResult cmdScope(const std::vector<std::string> &args);
  CommandResult cmdList(const std::vector<std::string> &args);
  CommandResult cmdDump(const std::vector<std::string> &args);
  CommandResult cmdHelp(const std::vector<std::string> &args);
  CommandResult cmdQuit(const std::vector<std::string> &args);
  CommandResult cmdReset(const std::vector<std::string> &args);

  /// Parse command line into command and arguments.
  std::pair<std::string, std::vector<std::string>>
  parseCommand(StringRef line) const;

  /// Output helpers.
  void printOutput(StringRef msg);
  void printError(StringRef msg);

  DebugSession &session;
  llvm::raw_ostream &out;
  llvm::raw_ostream &err;
  std::string prompt = "(circt-debug) ";
  bool echo = false;

  /// Command aliases.
  llvm::StringMap<std::string> aliases;
};

//===----------------------------------------------------------------------===//
// Interactive CLI
//===----------------------------------------------------------------------===//

/// Runs an interactive command-line debug session.
class InteractiveCLI {
public:
  InteractiveCLI(DebugSession &session);
  ~InteractiveCLI();

  /// Run the interactive loop.
  int run();

  /// Set input/output streams.
  void setInput(std::istream &in);
  void setOutput(llvm::raw_ostream &out);
  void setError(llvm::raw_ostream &err);

private:
  /// Read a line from input.
  std::optional<std::string> readLine();

  /// Handle Ctrl+C.
  void setupSignalHandlers();

  DebugSession &session;
  std::unique_ptr<CommandProcessor> cmdProcessor;

  std::istream *input = nullptr;
  llvm::raw_ostream *output = nullptr;
  llvm::raw_ostream *error = nullptr;

  std::string lastCommand;
  bool useReadline = false;
};

} // namespace debug
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_DEBUG_DEBUGSESSION_H
