//===- DebugSession.cpp - CIRCT Debug Session Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-debug/DebugSession.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <iostream>
#include <sstream>

using namespace circt;
using namespace circt::debug;

//===----------------------------------------------------------------------===//
// StopReason
//===----------------------------------------------------------------------===//

StringRef circt::debug::toString(StopReason reason) {
  switch (reason) {
  case StopReason::None:
    return "none";
  case StopReason::Breakpoint:
    return "breakpoint";
  case StopReason::Watchpoint:
    return "watchpoint";
  case StopReason::Step:
    return "step";
  case StopReason::Finished:
    return "finished";
  case StopReason::Error:
    return "error";
  case StopReason::UserInterrupt:
    return "user interrupt";
  case StopReason::MaxCycles:
    return "max cycles";
  case StopReason::MaxTime:
    return "max time";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// DebugSession Implementation
//===----------------------------------------------------------------------===//

DebugSession::DebugSession(std::unique_ptr<SimulationBackend> backend,
                           const DebugConfig &config)
    : backend(std::move(backend)), config(config) {}

DebugSession::~DebugSession() { stop(); }

bool DebugSession::start() {
  if (running)
    return true;

  if (!backend->initialize(config)) {
    output("Error: Failed to initialize simulation\n");
    return false;
  }

  running = true;

  // Start VCD dump if configured
  if (!config.vcdFile.empty()) {
    startVCDDump(config.vcdFile);
  }

  return true;
}

void DebugSession::stop() {
  if (!running)
    return;

  stopVCDDump();
  running = false;
}

StopReason DebugSession::run(uint64_t cycles) {
  if (!running)
    return StopReason::Error;

  uint64_t targetCycle =
      cycles > 0 ? getState().getCycle() + cycles : UINT64_MAX;

  while (getState().getCycle() < targetCycle) {
    if (!backend->stepClock()) {
      lastStopReason = StopReason::Error;
      handleStop(lastStopReason);
      return lastStopReason;
    }

    // Update watchpoints
    breakpointMgr.updateWatchpoints(getState());

    // Check stop conditions
    lastStopReason = checkStopConditions();
    if (lastStopReason != StopReason::None) {
      handleStop(lastStopReason);
      return lastStopReason;
    }

    // VCD dump
    if (vcdActive)
      dumpVCDValues();
  }

  if (cycles > 0)
    lastStopReason = StopReason::Step;
  else
    lastStopReason = StopReason::Finished;

  handleStop(lastStopReason);
  return lastStopReason;
}

StopReason DebugSession::step(uint64_t n) {
  if (!running)
    return StopReason::Error;

  for (uint64_t i = 0; i < n; ++i) {
    if (!backend->stepClock()) {
      lastStopReason = StopReason::Error;
      handleStop(lastStopReason);
      return lastStopReason;
    }

    breakpointMgr.updateWatchpoints(getState());

    if (vcdActive)
      dumpVCDValues();
  }

  lastStopReason = StopReason::Step;
  handleStop(lastStopReason);
  return lastStopReason;
}

StopReason DebugSession::stepDelta() {
  if (!running)
    return StopReason::Error;

  if (!backend->stepDelta()) {
    lastStopReason = StopReason::Error;
    handleStop(lastStopReason);
    return lastStopReason;
  }

  breakpointMgr.updateWatchpoints(getState());

  lastStopReason = StopReason::Step;
  handleStop(lastStopReason);
  return lastStopReason;
}

StopReason DebugSession::continueExec() {
  if (!running)
    return StopReason::Error;

  while (!backend->isFinished()) {
    if (!backend->stepClock()) {
      lastStopReason = StopReason::Error;
      handleStop(lastStopReason);
      return lastStopReason;
    }

    breakpointMgr.updateWatchpoints(getState());

    lastStopReason = checkStopConditions();
    if (lastStopReason != StopReason::None) {
      handleStop(lastStopReason);
      return lastStopReason;
    }

    if (vcdActive)
      dumpVCDValues();
  }

  lastStopReason = StopReason::Finished;
  handleStop(lastStopReason);
  return lastStopReason;
}

StopReason DebugSession::runUntil(const SimTime &time) {
  if (!running)
    return StopReason::Error;

  while (getState().getTime() < time && !backend->isFinished()) {
    if (!backend->stepClock()) {
      lastStopReason = StopReason::Error;
      handleStop(lastStopReason);
      return lastStopReason;
    }

    breakpointMgr.updateWatchpoints(getState());

    lastStopReason = checkStopConditions();
    if (lastStopReason != StopReason::None) {
      handleStop(lastStopReason);
      return lastStopReason;
    }

    if (vcdActive)
      dumpVCDValues();
  }

  lastStopReason = StopReason::Step;
  handleStop(lastStopReason);
  return lastStopReason;
}

StopReason DebugSession::runUntilCycle(uint64_t cycle) {
  if (!running)
    return StopReason::Error;

  while (getState().getCycle() < cycle && !backend->isFinished()) {
    if (!backend->stepClock()) {
      lastStopReason = StopReason::Error;
      handleStop(lastStopReason);
      return lastStopReason;
    }

    breakpointMgr.updateWatchpoints(getState());

    lastStopReason = checkStopConditions();
    if (lastStopReason != StopReason::None) {
      handleStop(lastStopReason);
      return lastStopReason;
    }

    if (vcdActive)
      dumpVCDValues();
  }

  lastStopReason = StopReason::Step;
  handleStop(lastStopReason);
  return lastStopReason;
}

bool DebugSession::reset() {
  if (!backend->reset()) {
    output("Error: Failed to reset simulation\n");
    return false;
  }
  return true;
}

EvalResult DebugSession::getSignal(StringRef path) const {
  ExpressionEvaluator eval(getState());
  return eval.evaluate(path);
}

bool DebugSession::setSignal(StringRef path, const SignalValue &value) {
  return backend->setSignal(path, value);
}

bool DebugSession::releaseSignal(StringRef path) {
  return backend->releaseSignal(path);
}

std::vector<SignalInfo> DebugSession::listSignals() const {
  const Scope *scope = getState().getCurrentScope();
  if (!scope)
    return {};
  return scope->getSignals();
}

std::vector<SignalInfo> DebugSession::findSignals(StringRef pattern) const {
  // Simple wildcard matching for now
  std::vector<SignalInfo> results;

  std::function<void(const Scope *)> searchScope = [&](const Scope *scope) {
    for (const auto &sig : scope->getSignals()) {
      // Check if name matches pattern (simple contains for now)
      if (pattern.empty() || sig.fullPath.find(pattern.str()) != std::string::npos ||
          sig.name.find(pattern.str()) != std::string::npos) {
        results.push_back(sig);
      }
    }
    for (const auto &child : scope->getChildren()) {
      searchScope(child.get());
    }
  };

  const Scope *root = getState().getRootScope();
  if (root)
    searchScope(root);

  return results;
}

std::string DebugSession::getCurrentScope() const {
  const Scope *scope = getState().getCurrentScope();
  return scope ? scope->getFullPath() : "";
}

bool DebugSession::changeScope(StringRef path) {
  Scope *root = backend->getState().getRootScope();
  if (!root)
    return false;

  // Handle absolute vs relative path
  Scope *target = root;
  StringRef remaining = path;

  if (path.starts_with("/") || path.starts_with(root->getName())) {
    remaining = path.drop_front(path.starts_with("/") ? 1 : root->getName().size());
    if (remaining.starts_with("."))
      remaining = remaining.drop_front(1);
  } else {
    // Relative path from current scope
    target = backend->getState().getCurrentScope();
    if (!target)
      target = root;
  }

  // Navigate the path
  while (!remaining.empty()) {
    auto dotPos = remaining.find('.');
    StringRef component =
        dotPos == StringRef::npos ? remaining : remaining.take_front(dotPos);
    remaining =
        dotPos == StringRef::npos ? "" : remaining.drop_front(dotPos + 1);

    if (component == "..")  {
      if (target->getParent())
        target = target->getParent();
    } else if (!component.empty() && component != ".") {
      Scope *child = target->findChild(component);
      if (!child)
        return false;
      target = child;
    }
  }

  backend->getState().setCurrentScope(target);
  return true;
}

bool DebugSession::scopeUp() {
  Scope *current = backend->getState().getCurrentScope();
  if (!current || !current->getParent())
    return false;
  backend->getState().setCurrentScope(current->getParent());
  return true;
}

std::vector<std::string> DebugSession::listScopes() const {
  std::vector<std::string> results;
  const Scope *scope = getState().getCurrentScope();
  if (!scope)
    return results;

  for (const auto &child : scope->getChildren()) {
    results.push_back(child->getName().str());
  }
  return results;
}

EvalResult DebugSession::evaluate(StringRef expr) const {
  ExpressionEvaluator eval(getState());
  return eval.evaluate(expr);
}

bool DebugSession::startVCDDump(StringRef filename) {
  std::error_code ec;
  vcdStream =
      std::make_unique<llvm::raw_fd_ostream>(filename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    output("Error: Cannot open VCD file: " + ec.message() + "\n");
    return false;
  }

  vcdActive = true;

  // Write VCD header
  *vcdStream << "$date\n    " << "Generated by CIRCT Debug" << "\n$end\n";
  *vcdStream << "$version\n    CIRCT Debug 1.0\n$end\n";
  *vcdStream << "$timescale 1ns $end\n";

  // Write signal definitions
  const Scope *root = getState().getRootScope();
  if (root) {
    std::function<void(const Scope *)> writeScope = [&](const Scope *scope) {
      *vcdStream << "$scope module " << scope->getName() << " $end\n";
      for (const auto &sig : scope->getSignals()) {
        *vcdStream << "$var " << sig.getTypeString() << " " << sig.width << " "
                   << sig.fullPath << " " << sig.name << " $end\n";
      }
      for (const auto &child : scope->getChildren()) {
        writeScope(child.get());
      }
      *vcdStream << "$upscope $end\n";
    };
    writeScope(root);
  }

  *vcdStream << "$enddefinitions $end\n";
  *vcdStream << "$dumpvars\n";

  // Write initial values
  dumpVCDValues();

  *vcdStream << "$end\n";

  return true;
}

void DebugSession::stopVCDDump() {
  if (vcdActive) {
    vcdStream->flush();
    vcdStream.reset();
    vcdActive = false;
  }
}

void DebugSession::dumpVCDValues() {
  if (!vcdActive || !vcdStream)
    return;

  *vcdStream << "#" << getState().getTime().value << "\n";

  // Dump all signal values
  for (const auto &change : getState().getRecentChanges()) {
    *vcdStream << "b" << change.newValue.toBinaryString() << " " << change.path
               << "\n";
  }

  // Clear recent changes
  backend->getState().clearChanges();
}

bool DebugSession::saveWaveformSnippet(StringRef filename, unsigned preTime,
                                       unsigned postTime) {
  // Implementation would use VCDRingBuffer - simplified for now
  return startVCDDump(filename);
}

SimState &DebugSession::getState() { return backend->getState(); }

const SimState &DebugSession::getState() const { return backend->getState(); }

bool DebugSession::isFinished() const { return backend->isFinished(); }

StopReason DebugSession::checkStopConditions() {
  // Check simulation finished
  if (backend->isFinished())
    return StopReason::Finished;

  // Check max cycles
  if (config.maxCycles > 0 && getState().getCycle() >= config.maxCycles)
    return StopReason::MaxCycles;

  // Check max time
  if (config.maxTime.value > 0 && getState().getTime() >= config.maxTime)
    return StopReason::MaxTime;

  // Check breakpoints
  triggeredBreakpoints = breakpointMgr.getTriggeredBreakpoints(getState());
  if (!triggeredBreakpoints.empty()) {
    for (auto *bp : triggeredBreakpoints) {
      bp->incrementHitCount();
    }
    return StopReason::Breakpoint;
  }

  return StopReason::None;
}

void DebugSession::handleStop(StopReason reason) {
  if (stopCallback)
    stopCallback(reason, *this);

  if (reason == StopReason::Breakpoint) {
    for (auto *bp : triggeredBreakpoints) {
      if (breakCallback)
        breakCallback(*bp, *this);
    }
  }
}

void DebugSession::output(StringRef msg) {
  if (outputCallback)
    outputCallback(msg);
  else
    llvm::outs() << msg;
}

//===----------------------------------------------------------------------===//
// CommandProcessor Implementation
//===----------------------------------------------------------------------===//

CommandProcessor::CommandProcessor(DebugSession &session, llvm::raw_ostream &out,
                                   llvm::raw_ostream &err)
    : session(session), out(out), err(err) {
  // Set up aliases
  aliases["r"] = "run";
  aliases["s"] = "step";
  aliases["si"] = "stepi";
  aliases["c"] = "continue";
  aliases["b"] = "break";
  aliases["w"] = "watch";
  aliases["d"] = "delete";
  aliases["p"] = "print";
  aliases["i"] = "info";
  aliases["l"] = "list";
  aliases["h"] = "help";
  aliases["q"] = "quit";
  aliases["x"] = "dump";
}

CommandResult CommandProcessor::execute(StringRef command) {
  auto [cmd, args] = parseCommand(command);

  if (cmd.empty())
    return CommandResult::ok();

  // Check for alias
  auto aliasIt = aliases.find(cmd);
  if (aliasIt != aliases.end())
    cmd = aliasIt->second;

  // Dispatch to handler
  if (cmd == "run")
    return cmdRun(args);
  if (cmd == "step")
    return cmdStep(args);
  if (cmd == "stepi")
    return cmdStepDelta(args);
  if (cmd == "continue")
    return cmdContinue(args);
  if (cmd == "break")
    return cmdBreak(args);
  if (cmd == "watch")
    return cmdWatch(args);
  if (cmd == "delete")
    return cmdDelete(args);
  if (cmd == "enable")
    return cmdEnable(args);
  if (cmd == "disable")
    return cmdDisable(args);
  if (cmd == "print")
    return cmdPrint(args);
  if (cmd == "set")
    return cmdSet(args);
  if (cmd == "info")
    return cmdInfo(args);
  if (cmd == "scope")
    return cmdScope(args);
  if (cmd == "list")
    return cmdList(args);
  if (cmd == "dump")
    return cmdDump(args);
  if (cmd == "help")
    return cmdHelp(args);
  if (cmd == "quit" || cmd == "exit")
    return cmdQuit(args);
  if (cmd == "reset")
    return cmdReset(args);

  return CommandResult::error("Unknown command: " + cmd +
                              ". Type 'help' for a list of commands.");
}

std::string CommandProcessor::getHelp() const {
  std::string help;
  help += "CIRCT Debug Commands:\n\n";
  help += "Execution:\n";
  help += "  run [cycles]        - Run simulation for N cycles (0 = until end)\n";
  help += "  step [n]            - Step N clock cycles (default: 1)\n";
  help += "  stepi               - Step one delta cycle\n";
  help += "  continue            - Continue until breakpoint or end\n";
  help += "  reset               - Reset simulation to initial state\n\n";
  help += "Breakpoints:\n";
  help += "  break <file:line>   - Set breakpoint at source location\n";
  help += "  break -sig <signal> - Break on signal change\n";
  help += "  break -cond <expr>  - Break when condition is true\n";
  help += "  break -cycle <n>    - Break at cycle N\n";
  help += "  watch <signal>      - Watch signal for changes\n";
  help += "  delete [id]         - Delete breakpoint/watchpoint\n";
  help += "  enable <id>         - Enable breakpoint/watchpoint\n";
  help += "  disable <id>        - Disable breakpoint/watchpoint\n\n";
  help += "Inspection:\n";
  help += "  print <expr>        - Print signal/expression value\n";
  help += "  info signals        - List signals in current scope\n";
  help += "  info breakpoints    - List all breakpoints\n";
  help += "  info watchpoints    - List all watchpoints\n";
  help += "  info scope          - Show current scope\n";
  help += "  info time           - Show current simulation time\n\n";
  help += "Navigation:\n";
  help += "  scope <path>        - Change to scope\n";
  help += "  scope ..            - Go up one level\n";
  help += "  list                - List child scopes and signals\n\n";
  help += "Waveforms:\n";
  help += "  dump vcd <file>     - Start VCD dump to file\n";
  help += "  dump stop           - Stop VCD dump\n\n";
  help += "Other:\n";
  help += "  set <signal> <val>  - Force signal to value\n";
  help += "  help [command]      - Show help\n";
  help += "  quit                - Exit debugger\n\n";
  help += "Aliases: r=run, s=step, si=stepi, c=continue, b=break, w=watch,\n";
  help += "         d=delete, p=print, i=info, l=list, h=help, q=quit\n";
  return help;
}

std::string CommandProcessor::getHelp(StringRef command) const {
  // Command-specific help
  std::string cmd = command.str();
  auto aliasIt = aliases.find(cmd);
  if (aliasIt != aliases.end())
    cmd = aliasIt->second;

  if (cmd == "break") {
    return "break - Set a breakpoint\n\n"
           "Usage:\n"
           "  break <file>:<line>      - Break at source location\n"
           "  break -sig <signal>      - Break on signal change\n"
           "  break -sig <signal> <val> - Break when signal equals value\n"
           "  break -cond <expr>       - Break when expression is true\n"
           "  break -cycle <n>         - Break at cycle N\n"
           "  break -time <t>          - Break at time T\n\n"
           "Examples:\n"
           "  break top.v:42\n"
           "  break -sig clk\n"
           "  break -sig counter 8'hFF\n"
           "  break -cond \"counter == 100\"\n";
  }

  if (cmd == "print") {
    return "print - Print a value\n\n"
           "Usage:\n"
           "  print <signal>           - Print signal value\n"
           "  print <expr>             - Evaluate and print expression\n\n"
           "Formats:\n"
           "  print/x <signal>         - Print as hex\n"
           "  print/b <signal>         - Print as binary\n"
           "  print/d <signal>         - Print as decimal\n";
  }

  return getHelp();
}

std::pair<std::string, std::vector<std::string>>
CommandProcessor::parseCommand(StringRef line) const {
  std::vector<std::string> parts;
  StringRef remaining = line.trim();

  while (!remaining.empty()) {
    // Skip whitespace
    remaining = remaining.ltrim();
    if (remaining.empty())
      break;

    // Handle quoted strings
    if (remaining.starts_with("\"")) {
      auto endQuote = remaining.find('"', 1);
      if (endQuote == StringRef::npos) {
        parts.push_back(remaining.drop_front(1).str());
        break;
      }
      parts.push_back(remaining.slice(1, endQuote).str());
      remaining = remaining.drop_front(endQuote + 1);
    } else {
      // Find next whitespace
      auto space = remaining.find_first_of(" \t");
      if (space == StringRef::npos) {
        parts.push_back(remaining.str());
        break;
      }
      parts.push_back(remaining.take_front(space).str());
      remaining = remaining.drop_front(space);
    }
  }

  if (parts.empty())
    return {"", {}};

  std::string cmd = parts[0];
  parts.erase(parts.begin());
  return {cmd, parts};
}

void CommandProcessor::printOutput(StringRef msg) { out << msg; }

void CommandProcessor::printError(StringRef msg) {
  err << "Error: " << msg << "\n";
}

CommandResult CommandProcessor::cmdRun(const std::vector<std::string> &args) {
  uint64_t cycles = 0;
  if (!args.empty()) {
    if (llvm::StringRef(args[0]).getAsInteger(10, cycles))
      return CommandResult::error("Invalid cycle count: " + args[0]);
  }

  auto reason = session.run(cycles);

  std::string output;
  output += "Stopped: " + std::string(toString(reason)) + "\n";
  output += "Time: " + session.getState().getTime().toString() + "\n";
  output += "Cycle: " + std::to_string(session.getState().getCycle()) + "\n";

  if (reason == StopReason::Breakpoint) {
    for (auto *bp : session.getTriggeredBreakpoints()) {
      output += "Breakpoint " + std::to_string(bp->getId()) + ": " +
                bp->getDescription() + "\n";
    }
  }

  return CommandResult::ok(output);
}

CommandResult CommandProcessor::cmdStep(const std::vector<std::string> &args) {
  uint64_t n = 1;
  if (!args.empty()) {
    if (llvm::StringRef(args[0]).getAsInteger(10, n))
      return CommandResult::error("Invalid step count: " + args[0]);
  }

  session.step(n);

  std::string output;
  output += "Time: " + session.getState().getTime().toString() + "\n";
  output += "Cycle: " + std::to_string(session.getState().getCycle()) + "\n";

  return CommandResult::ok(output);
}

CommandResult
CommandProcessor::cmdStepDelta(const std::vector<std::string> &args) {
  session.stepDelta();

  std::string output;
  output += "Time: " + session.getState().getTime().toString() + "\n";
  output += "Cycle: " + std::to_string(session.getState().getCycle()) + "\n";
  output += "Delta: " + std::to_string(session.getState().getDeltaCycle()) + "\n";

  return CommandResult::ok(output);
}

CommandResult
CommandProcessor::cmdContinue(const std::vector<std::string> &args) {
  auto reason = session.continueExec();

  std::string output;
  output += "Stopped: " + std::string(toString(reason)) + "\n";
  output += "Time: " + session.getState().getTime().toString() + "\n";
  output += "Cycle: " + std::to_string(session.getState().getCycle()) + "\n";

  if (reason == StopReason::Breakpoint) {
    for (auto *bp : session.getTriggeredBreakpoints()) {
      output += "Breakpoint " + std::to_string(bp->getId()) + ": " +
                bp->getDescription() + "\n";
    }
  }

  return CommandResult::ok(output);
}

CommandResult CommandProcessor::cmdBreak(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: break <location> or break -sig <signal>");

  auto &mgr = session.getBreakpointManager();
  unsigned id;

  if (args[0] == "-sig" || args[0] == "--signal") {
    if (args.size() < 2)
      return CommandResult::error("Missing signal name");

    if (args.size() >= 3) {
      // break -sig signal value
      auto val = SignalValue::fromString(args[2], 32);
      if (!val)
        return CommandResult::error("Invalid value: " + args[2]);
      id = mgr.addSignalBreakpoint(args[1], *val);
    } else {
      id = mgr.addSignalBreakpoint(args[1]);
    }
  } else if (args[0] == "-cond" || args[0] == "--condition") {
    if (args.size() < 2)
      return CommandResult::error("Missing condition expression");
    // Join remaining args as expression
    std::string expr;
    for (size_t i = 1; i < args.size(); ++i) {
      if (i > 1)
        expr += " ";
      expr += args[i];
    }
    id = mgr.addConditionBreakpoint(expr);
  } else if (args[0] == "-cycle") {
    if (args.size() < 2)
      return CommandResult::error("Missing cycle number");
    uint64_t cycle;
    if (llvm::StringRef(args[1]).getAsInteger(10, cycle))
      return CommandResult::error("Invalid cycle: " + args[1]);
    id = mgr.addCycleBreakpoint(cycle);
  } else if (args[0] == "-time") {
    if (args.size() < 2)
      return CommandResult::error("Missing time value");
    uint64_t time;
    if (llvm::StringRef(args[1]).getAsInteger(10, time))
      return CommandResult::error("Invalid time: " + args[1]);
    id = mgr.addTimeBreakpoint(SimTime(time));
  } else {
    // Assume file:line format
    auto colonPos = args[0].find(':');
    if (colonPos == std::string::npos)
      return CommandResult::error(
          "Invalid breakpoint location. Use file:line format.");

    std::string file = args[0].substr(0, colonPos);
    unsigned line;
    if (llvm::StringRef(args[0].substr(colonPos + 1)).getAsInteger(10, line))
      return CommandResult::error("Invalid line number");

    id = mgr.addLineBreakpoint(file, line);
  }

  auto *bp = mgr.getBreakpoint(id);
  return CommandResult::ok("Breakpoint " + std::to_string(id) + " set: " +
                           bp->getDescription() + "\n");
}

CommandResult CommandProcessor::cmdWatch(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: watch <signal>");

  unsigned id = session.getBreakpointManager().addWatchpoint(args[0]);
  return CommandResult::ok("Watchpoint " + std::to_string(id) +
                           " set on: " + args[0] + "\n");
}

CommandResult
CommandProcessor::cmdDelete(const std::vector<std::string> &args) {
  auto &mgr = session.getBreakpointManager();

  if (args.empty()) {
    // Delete all
    mgr.removeAllBreakpoints();
    mgr.removeAllWatchpoints();
    return CommandResult::ok("All breakpoints and watchpoints deleted.\n");
  }

  unsigned id;
  if (llvm::StringRef(args[0]).getAsInteger(10, id))
    return CommandResult::error("Invalid ID: " + args[0]);

  if (mgr.removeBreakpoint(id))
    return CommandResult::ok("Breakpoint " + std::to_string(id) + " deleted.\n");

  if (mgr.removeWatchpoint(id))
    return CommandResult::ok("Watchpoint " + std::to_string(id) + " deleted.\n");

  return CommandResult::error("No breakpoint or watchpoint with ID " +
                              std::to_string(id));
}

CommandResult
CommandProcessor::cmdEnable(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: enable <id>");

  unsigned id;
  if (llvm::StringRef(args[0]).getAsInteger(10, id))
    return CommandResult::error("Invalid ID: " + args[0]);

  auto &mgr = session.getBreakpointManager();
  if (mgr.enableBreakpoint(id, true))
    return CommandResult::ok("Breakpoint " + std::to_string(id) + " enabled.\n");

  if (mgr.enableWatchpoint(id, true))
    return CommandResult::ok("Watchpoint " + std::to_string(id) + " enabled.\n");

  return CommandResult::error("No breakpoint or watchpoint with ID " +
                              std::to_string(id));
}

CommandResult
CommandProcessor::cmdDisable(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: disable <id>");

  unsigned id;
  if (llvm::StringRef(args[0]).getAsInteger(10, id))
    return CommandResult::error("Invalid ID: " + args[0]);

  auto &mgr = session.getBreakpointManager();
  if (mgr.enableBreakpoint(id, false))
    return CommandResult::ok("Breakpoint " + std::to_string(id) + " disabled.\n");

  if (mgr.enableWatchpoint(id, false))
    return CommandResult::ok("Watchpoint " + std::to_string(id) + " disabled.\n");

  return CommandResult::error("No breakpoint or watchpoint with ID " +
                              std::to_string(id));
}

CommandResult CommandProcessor::cmdPrint(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: print <signal or expression>");

  // Check for format specifier
  std::string format = "hex";
  std::string expr;

  if (!args[0].empty() && args[0][0] == '/') {
    if (args[0] == "/x")
      format = "hex";
    else if (args[0] == "/b")
      format = "bin";
    else if (args[0] == "/d")
      format = "dec";

    if (args.size() < 2)
      return CommandResult::error("Missing expression");

    for (size_t i = 1; i < args.size(); ++i) {
      if (i > 1)
        expr += " ";
      expr += args[i];
    }
  } else {
    for (const auto &arg : args) {
      if (!expr.empty())
        expr += " ";
      expr += arg;
    }
  }

  auto result = session.evaluate(expr);
  if (!result.succeeded)
    return CommandResult::error(result.error);

  std::string output = expr + " = ";
  if (format == "hex")
    output += "0x" + result.value->toHexString();
  else if (format == "bin")
    output += "0b" + result.value->toBinaryString();
  else
    output += result.value->toString(10);

  output += " (" + std::to_string(result.value->getWidth()) + " bits)\n";

  return CommandResult::ok(output);
}

CommandResult CommandProcessor::cmdSet(const std::vector<std::string> &args) {
  if (args.size() < 2)
    return CommandResult::error("Usage: set <signal> <value>");

  auto val = SignalValue::fromString(args[1], 32);
  if (!val)
    return CommandResult::error("Invalid value: " + args[1]);

  if (!session.setSignal(args[0], *val))
    return CommandResult::error("Failed to set signal: " + args[0]);

  return CommandResult::ok(args[0] + " = " + val->toHexString() + "\n");
}

CommandResult CommandProcessor::cmdInfo(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error(
        "Usage: info <signals|breakpoints|watchpoints|scope|time>");

  std::string output;

  if (args[0] == "signals" || args[0] == "sig") {
    auto signals = session.listSignals();
    if (signals.empty()) {
      output = "No signals in current scope.\n";
    } else {
      for (const auto &sig : signals) {
        auto val = session.getSignal(sig.fullPath);
        output += "  " + sig.name + " [" + std::to_string(sig.width) + "] " +
                  std::string(sig.getTypeString());
        if (val.succeeded && val.value)
          output += " = 0x" + val.value->toHexString();
        output += "\n";
      }
    }
  } else if (args[0] == "breakpoints" || args[0] == "break" || args[0] == "b") {
    auto &mgr = session.getBreakpointManager();
    const auto &bps = mgr.getBreakpoints();
    if (bps.empty()) {
      output = "No breakpoints.\n";
    } else {
      for (const auto &bp : bps) {
        output += "  " + std::to_string(bp->getId()) + ": " +
                  bp->getDescription();
        if (!bp->isEnabled())
          output += " [disabled]";
        if (bp->getHitCount() > 0)
          output += " (hit " + std::to_string(bp->getHitCount()) + " times)";
        output += "\n";
      }
    }
  } else if (args[0] == "watchpoints" || args[0] == "watch" || args[0] == "w") {
    auto &mgr = session.getBreakpointManager();
    const auto &wps = mgr.getWatchpoints();
    if (wps.empty()) {
      output = "No watchpoints.\n";
    } else {
      for (const auto &wp : wps) {
        output += "  " + std::to_string(wp->getId()) + ": " +
                  wp->getSignal().str();
        if (!wp->isEnabled())
          output += " [disabled]";
        output += "\n";
      }
    }
  } else if (args[0] == "scope") {
    output = "Current scope: " + session.getCurrentScope() + "\n";
    auto scopes = session.listScopes();
    if (!scopes.empty()) {
      output += "Child scopes:\n";
      for (const auto &s : scopes) {
        output += "  " + s + "/\n";
      }
    }
  } else if (args[0] == "time") {
    output = "Time: " + session.getState().getTime().toString() + "\n";
    output += "Cycle: " + std::to_string(session.getState().getCycle()) + "\n";
    output +=
        "Delta: " + std::to_string(session.getState().getDeltaCycle()) + "\n";
  } else {
    return CommandResult::error("Unknown info type: " + args[0]);
  }

  return CommandResult::ok(output);
}

CommandResult CommandProcessor::cmdScope(const std::vector<std::string> &args) {
  if (args.empty()) {
    // Show current scope
    return CommandResult::ok("Current scope: " + session.getCurrentScope() +
                             "\n");
  }

  if (args[0] == "..") {
    if (!session.scopeUp())
      return CommandResult::error("Already at root scope");
    return CommandResult::ok("Scope: " + session.getCurrentScope() + "\n");
  }

  if (!session.changeScope(args[0]))
    return CommandResult::error("Scope not found: " + args[0]);

  return CommandResult::ok("Scope: " + session.getCurrentScope() + "\n");
}

CommandResult CommandProcessor::cmdList(const std::vector<std::string> &args) {
  std::string output;

  // List child scopes
  auto scopes = session.listScopes();
  for (const auto &s : scopes) {
    output += "  " + s + "/\n";
  }

  // List signals
  auto signals = session.listSignals();
  for (const auto &sig : signals) {
    output += "  " + sig.name + " [" + std::to_string(sig.width) + "] " +
              std::string(sig.getTypeString()) + "\n";
  }

  if (output.empty())
    output = "Empty scope.\n";

  return CommandResult::ok(output);
}

CommandResult CommandProcessor::cmdDump(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::error("Usage: dump vcd <file> | dump stop");

  if (args[0] == "vcd") {
    if (args.size() < 2)
      return CommandResult::error("Usage: dump vcd <filename>");

    if (!session.startVCDDump(args[1]))
      return CommandResult::error("Failed to start VCD dump");

    return CommandResult::ok("VCD dump started: " + args[1] + "\n");
  }

  if (args[0] == "stop") {
    session.stopVCDDump();
    return CommandResult::ok("VCD dump stopped.\n");
  }

  return CommandResult::error("Unknown dump command: " + args[0]);
}

CommandResult CommandProcessor::cmdHelp(const std::vector<std::string> &args) {
  if (args.empty())
    return CommandResult::ok(getHelp());
  return CommandResult::ok(getHelp(args[0]));
}

CommandResult CommandProcessor::cmdQuit(const std::vector<std::string> &args) {
  return CommandResult::quit();
}

CommandResult CommandProcessor::cmdReset(const std::vector<std::string> &args) {
  if (!session.reset())
    return CommandResult::error("Failed to reset simulation");
  return CommandResult::ok("Simulation reset to initial state.\n");
}

//===----------------------------------------------------------------------===//
// InteractiveCLI Implementation
//===----------------------------------------------------------------------===//

InteractiveCLI::InteractiveCLI(DebugSession &session)
    : session(session), output(&llvm::outs()), error(&llvm::errs()) {
  cmdProcessor =
      std::make_unique<CommandProcessor>(session, *output, *error);
}

InteractiveCLI::~InteractiveCLI() = default;

int InteractiveCLI::run() {
  // Print welcome message
  *output << "CIRCT Debug - Interactive Hardware Debugger\n";
  *output << "Type 'help' for a list of commands.\n\n";

  // Start the session
  if (!session.start()) {
    *error << "Failed to start debug session.\n";
    return 1;
  }

  // Main loop
  while (true) {
    *output << cmdProcessor->getPrompt();
    output->flush();

    auto line = readLine();
    if (!line) {
      *output << "\n";
      break;
    }

    // Handle empty line - repeat last command
    std::string cmd = *line;
    if (cmd.empty()) {
      if (!lastCommand.empty())
        cmd = lastCommand;
      else
        continue;
    } else {
      lastCommand = cmd;
    }

    auto result = cmdProcessor->execute(cmd);

    if (!result.output.empty())
      *output << result.output;

    if (!result.success && result.output.empty())
      *error << "Command failed.\n";

    if (!result.shouldContinue)
      break;
  }

  session.stop();
  return 0;
}

void InteractiveCLI::setInput(std::istream &in) { input = &in; }

void InteractiveCLI::setOutput(llvm::raw_ostream &out) {
  output = &out;
  cmdProcessor =
      std::make_unique<CommandProcessor>(session, *output, *error);
}

void InteractiveCLI::setError(llvm::raw_ostream &err) {
  error = &err;
  cmdProcessor =
      std::make_unique<CommandProcessor>(session, *output, *error);
}

std::optional<std::string> InteractiveCLI::readLine() {
  std::string line;
  if (input) {
    if (!std::getline(*input, line))
      return std::nullopt;
  } else {
    if (!std::getline(std::cin, line))
      return std::nullopt;
  }
  return line;
}

void InteractiveCLI::setupSignalHandlers() {
  // TODO: Set up Ctrl+C handling
}

} // namespace debug
} // namespace circt
