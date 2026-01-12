//===- DAPServer.cpp - Debug Adapter Protocol Server Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-debug/DAPServer.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include <iostream>
#include <sstream>

using namespace circt;
using namespace circt::debug;
using namespace circt::debug::dap;

//===----------------------------------------------------------------------===//
// Message Implementation
//===----------------------------------------------------------------------===//

llvm::json::Object Message::toJSON() const {
  return llvm::json::Object{{"seq", seq}, {"type", type}};
}

llvm::json::Object Request::toJSON() const {
  auto obj = Message::toJSON();
  obj["command"] = command;
  if (!arguments.empty())
    obj["arguments"] = llvm::json::Object(arguments);
  return obj;
}

Response::Response(const Request &req, bool success)
    : success(success), command(req.command) {
  type = "response";
  requestSeq = req.seq;
}

llvm::json::Object Response::toJSON() const {
  auto obj = Message::toJSON();
  obj["request_seq"] = requestSeq;
  obj["success"] = success;
  obj["command"] = command;
  if (!message.empty())
    obj["message"] = message;
  if (!body.empty())
    obj["body"] = llvm::json::Object(body);
  return obj;
}

Event::Event(StringRef eventName) : event(eventName.str()) { type = "event"; }

llvm::json::Object Event::toJSON() const {
  auto obj = Message::toJSON();
  obj["event"] = event;
  if (!body.empty())
    obj["body"] = llvm::json::Object(body);
  return obj;
}

//===----------------------------------------------------------------------===//
// Capabilities Implementation
//===----------------------------------------------------------------------===//

llvm::json::Object Capabilities::toJSON() const {
  return llvm::json::Object{
      {"supportsConfigurationDoneRequest", supportsConfigurationDoneRequest},
      {"supportsFunctionBreakpoints", supportsFunctionBreakpoints},
      {"supportsConditionalBreakpoints", supportsConditionalBreakpoints},
      {"supportsHitConditionalBreakpoints", supportsHitConditionalBreakpoints},
      {"supportsEvaluateForHovers", supportsEvaluateForHovers},
      {"supportsSetVariable", supportsSetVariable},
      {"supportsCompletionsRequest", supportsCompletionsRequest},
      {"supportsModulesRequest", supportsModulesRequest},
      {"supportsRestartRequest", supportsRestartRequest},
      {"supportsValueFormattingOptions", supportsValueFormattingOptions},
      {"supportTerminateDebuggee", supportTerminateDebuggee},
      {"supportsTerminateRequest", supportsTerminateRequest},
      {"supportsDataBreakpoints", supportsDataBreakpoints},
      {"supportsSteppingGranularity", supportsSteppingGranularity},
  };
}

//===----------------------------------------------------------------------===//
// VariableReferenceManager Implementation
//===----------------------------------------------------------------------===//

VariableReferenceManager::VariableReferenceManager() = default;

int VariableReferenceManager::createReference(StringRef path) {
  auto it = pathToReference.find(path);
  if (it != pathToReference.end())
    return it->second;

  int ref = nextReference++;
  referenceToPath[ref] = path.str();
  pathToReference[path] = ref;
  return ref;
}

std::optional<std::string>
VariableReferenceManager::getPath(int reference) const {
  auto it = referenceToPath.find(reference);
  if (it == referenceToPath.end())
    return std::nullopt;
  return it->second;
}

void VariableReferenceManager::clear() {
  referenceToPath.clear();
  pathToReference.clear();
  nextReference = 1;
}

std::vector<VariableReferenceManager::Scope>
VariableReferenceManager::getScopes(const SimState &state, int frameId) {
  std::vector<Scope> scopes;

  // Current scope
  const debug::Scope *currentScope = state.getCurrentScope();
  if (currentScope) {
    Scope local;
    local.type = ScopeType::Local;
    local.name = "Signals";
    local.path = currentScope->getFullPath();
    local.variablesReference = createReference(local.path);
    scopes.push_back(local);
  }

  // Root scope (hierarchy)
  const debug::Scope *rootScope = state.getRootScope();
  if (rootScope) {
    Scope hierarchy;
    hierarchy.type = ScopeType::Hierarchy;
    hierarchy.name = "Hierarchy";
    hierarchy.path = rootScope->getFullPath();
    hierarchy.variablesReference = createReference("__hierarchy__");
    scopes.push_back(hierarchy);
  }

  return scopes;
}

std::vector<VariableReferenceManager::Variable>
VariableReferenceManager::getVariables(const SimState &state, int reference) {
  std::vector<Variable> variables;

  auto pathOpt = getPath(reference);
  if (!pathOpt)
    return variables;

  std::string path = *pathOpt;

  // Special handling for hierarchy root
  if (path == "__hierarchy__") {
    const debug::Scope *root = state.getRootScope();
    if (root) {
      for (const auto &child : root->getChildren()) {
        Variable var;
        var.name = child->getName();
        var.value = "[scope]";
        var.type = "scope";
        var.variablesReference = createReference(child->getFullPath());
        var.evaluateName = child->getFullPath();
        variables.push_back(var);
      }
      for (const auto &sig : root->getSignals()) {
        Variable var;
        var.name = sig.name;
        auto val = state.getSignalValue(sig.fullPath);
        var.value = "0x" + val.toHexString();
        var.type = std::string(sig.getTypeString()) + "[" +
                   std::to_string(sig.width) + "]";
        var.variablesReference = 0;
        var.evaluateName = sig.fullPath;
        variables.push_back(var);
      }
    }
    return variables;
  }

  // Find the scope for this path
  const debug::Scope *scope = nullptr;
  const debug::Scope *root = state.getRootScope();

  if (root) {
    if (root->getFullPath() == path) {
      scope = root;
    } else {
      // Navigate to the scope
      std::function<const debug::Scope *(const debug::Scope *)> findScope;
      findScope = [&](const debug::Scope *s) -> const debug::Scope * {
        if (s->getFullPath() == path)
          return s;
        for (const auto &child : s->getChildren()) {
          if (auto found = findScope(child.get()))
            return found;
        }
        return nullptr;
      };
      scope = findScope(root);
    }
  }

  if (!scope)
    return variables;

  // Add child scopes
  for (const auto &child : scope->getChildren()) {
    Variable var;
    var.name = child->getName();
    var.value = "[scope]";
    var.type = "scope";
    var.variablesReference = createReference(child->getFullPath());
    var.evaluateName = child->getFullPath();
    variables.push_back(var);
  }

  // Add signals
  for (const auto &sig : scope->getSignals()) {
    Variable var;
    var.name = sig.name;
    auto val = state.getSignalValue(sig.fullPath);
    var.value = "0x" + val.toHexString();
    var.type =
        std::string(sig.getTypeString()) + "[" + std::to_string(sig.width) + "]";
    var.variablesReference = 0;
    var.evaluateName = sig.fullPath;
    variables.push_back(var);
  }

  return variables;
}

//===----------------------------------------------------------------------===//
// DAPServer Implementation
//===----------------------------------------------------------------------===//

DAPServer::DAPServer(DebugSession &session) : session(session) {
  // Set up session callbacks
  session.setStopCallback([this](StopReason reason, DebugSession &) {
    std::string reasonStr;
    std::string description;

    switch (reason) {
    case StopReason::Breakpoint:
      reasonStr = "breakpoint";
      description = "Breakpoint hit";
      break;
    case StopReason::Step:
      reasonStr = "step";
      break;
    case StopReason::Finished:
      sendTerminatedEvent();
      return;
    case StopReason::UserInterrupt:
      reasonStr = "pause";
      break;
    default:
      reasonStr = "exception";
      description = std::string(toString(reason));
      break;
    }

    varRefs.clear(); // Clear variable references at each stop
    sendStoppedEvent(reasonStr, description);
  });

  session.setOutputCallback([this](StringRef msg) {
    sendOutputEvent("console", msg);
  });
}

DAPServer::~DAPServer() { stop(); }

int DAPServer::run() { return run(std::cin, llvm::outs()); }

int DAPServer::run(std::istream &in, llvm::raw_ostream &out) {
  input = &in;
  output = &out;
  running = true;

  DAPTransport transport(in, out);

  while (running) {
    auto msgOpt = transport.readMessage();
    if (!msgOpt) {
      // End of input
      break;
    }

    // Parse as request
    Request req;
    if (auto seq = msgOpt->getInteger("seq"))
      req.seq = *seq;
    if (auto cmd = msgOpt->getString("command"))
      req.command = cmd->str();
    if (auto *args = msgOpt->getObject("arguments"))
      req.arguments = llvm::json::Object(*args);

    // Handle the request
    Response resp = handleRequest(req);
    resp.seq = nextSeq++;

    // Send response
    transport.writeMessage(resp.toJSON());
  }

  return 0;
}

void DAPServer::stop() { running = false; }

Response DAPServer::handleRequest(const Request &req) {
  if (req.command == "initialize")
    return handleInitialize(req);
  if (req.command == "launch")
    return handleLaunch(req);
  if (req.command == "attach")
    return handleAttach(req);
  if (req.command == "disconnect")
    return handleDisconnect(req);
  if (req.command == "configurationDone")
    return handleConfigurationDone(req);
  if (req.command == "setBreakpoints")
    return handleSetBreakpoints(req);
  if (req.command == "setDataBreakpoints")
    return handleSetDataBreakpoints(req);
  if (req.command == "threads")
    return handleThreads(req);
  if (req.command == "stackTrace")
    return handleStackTrace(req);
  if (req.command == "scopes")
    return handleScopes(req);
  if (req.command == "variables")
    return handleVariables(req);
  if (req.command == "setVariable")
    return handleSetVariable(req);
  if (req.command == "evaluate")
    return handleEvaluate(req);
  if (req.command == "continue")
    return handleContinue(req);
  if (req.command == "next")
    return handleNext(req);
  if (req.command == "stepIn")
    return handleStepIn(req);
  if (req.command == "stepOut")
    return handleStepOut(req);
  if (req.command == "pause")
    return handlePause(req);
  if (req.command == "terminate")
    return handleTerminate(req);
  if (req.command == "restart")
    return handleRestart(req);
  if (req.command == "completions")
    return handleCompletions(req);
  if (req.command == "modules")
    return handleModules(req);

  // Unknown command
  Response resp(req, false);
  resp.message = "Unknown command: " + req.command;
  return resp;
}

Response DAPServer::handleInitialize(const Request &req) {
  Response resp(req, true);
  resp.body = capabilities.toJSON();
  initialized = true;

  // Send initialized event after response
  sendInitializedEvent();

  return resp;
}

Response DAPServer::handleLaunch(const Request &req) {
  Response resp(req, true);

  // Start the debug session
  if (!session.start()) {
    resp.success = false;
    resp.message = "Failed to start debug session";
    return resp;
  }

  launched = true;

  // Send a stopped event at the beginning
  sendStoppedEvent("entry", "Stopped at entry");

  return resp;
}

Response DAPServer::handleAttach(const Request &req) {
  // Attach works same as launch for hardware debugging
  return handleLaunch(req);
}

Response DAPServer::handleDisconnect(const Request &req) {
  Response resp(req, true);

  bool terminateDebuggee = false;
  if (auto term = req.arguments.getBoolean("terminateDebuggee"))
    terminateDebuggee = *term;

  if (terminateDebuggee || launched) {
    session.stop();
  }

  running = false;
  return resp;
}

Response DAPServer::handleConfigurationDone(const Request &req) {
  Response resp(req, true);
  return resp;
}

Response DAPServer::handleSetBreakpoints(const Request &req) {
  Response resp(req, true);

  // Get source file
  std::string sourcePath;
  if (auto *source = req.arguments.getObject("source")) {
    if (auto path = source->getString("path"))
      sourcePath = path->str();
  }

  // Clear existing breakpoints for this source
  auto &mgr = session.getBreakpointManager();
  auto it = sourceBreakpoints.find(sourcePath);
  if (it != sourceBreakpoints.end()) {
    for (const auto &bp : it->second) {
      auto circtIt = dapToCirctBreakpoints.find(bp.id);
      if (circtIt != dapToCirctBreakpoints.end()) {
        mgr.removeBreakpoint(circtIt->second);
        dapToCirctBreakpoints.erase(circtIt);
      }
    }
    it->second.clear();
  }

  // Set new breakpoints
  llvm::json::Array breakpointsArray;

  if (auto *breakpoints = req.arguments.getArray("breakpoints")) {
    for (const auto &bpVal : *breakpoints) {
      auto *bpObj = bpVal.getAsObject();
      if (!bpObj)
        continue;

      int line = 0;
      if (auto l = bpObj->getInteger("line"))
        line = *l;

      // Create the breakpoint
      unsigned circtId = mgr.addLineBreakpoint(sourcePath, line);

      // Handle condition
      if (auto cond = bpObj->getString("condition")) {
        // For conditional breakpoints, remove the line breakpoint and add a
        // condition
        mgr.removeBreakpoint(circtId);
        circtId = mgr.addConditionBreakpoint(cond->str());
      }

      // Create DAP breakpoint
      DAPBreakpoint dapBp;
      dapBp.id = nextDAPBreakpointId++;
      dapBp.verified = true;
      dapBp.line = line;
      dapBp.source = sourcePath;

      dapToCirctBreakpoints[dapBp.id] = circtId;
      sourceBreakpoints[sourcePath].push_back(dapBp);

      breakpointsArray.push_back(dapBp.toJSON());
    }
  }

  resp.body["breakpoints"] = std::move(breakpointsArray);
  return resp;
}

Response DAPServer::handleSetDataBreakpoints(const Request &req) {
  Response resp(req, true);

  auto &mgr = session.getBreakpointManager();
  llvm::json::Array breakpointsArray;

  if (auto *breakpoints = req.arguments.getArray("breakpoints")) {
    for (const auto &bpVal : *breakpoints) {
      auto *bpObj = bpVal.getAsObject();
      if (!bpObj)
        continue;

      std::string dataId;
      if (auto id = bpObj->getString("dataId"))
        dataId = id->str();

      // Create signal breakpoint
      unsigned circtId = mgr.addSignalBreakpoint(dataId);

      DAPBreakpoint dapBp;
      dapBp.id = nextDAPBreakpointId++;
      dapBp.verified = true;
      dapBp.message = "Signal: " + dataId;

      dapToCirctBreakpoints[dapBp.id] = circtId;

      breakpointsArray.push_back(dapBp.toJSON());
    }
  }

  resp.body["breakpoints"] = std::move(breakpointsArray);
  return resp;
}

Response DAPServer::handleThreads(const Request &req) {
  Response resp(req, true);

  // Hardware simulation has a single "thread"
  llvm::json::Array threads;
  threads.push_back(llvm::json::Object{
      {"id", 1},
      {"name", "Simulation"},
  });
  resp.body["threads"] = std::move(threads);

  return resp;
}

Response DAPServer::handleStackTrace(const Request &req) {
  Response resp(req, true);

  // Hardware doesn't have a call stack - show simulation state
  llvm::json::Array stackFrames;

  const auto &state = session.getState();
  std::string scopePath = state.getCurrentScope()
                              ? state.getCurrentScope()->getFullPath()
                              : session.getConfig().topModule;

  llvm::json::Object frame{
      {"id", 1},
      {"name", scopePath},
      {"line", 0},
      {"column", 0},
  };

  // Add source if we have location info
  auto loc = state.getCurrentLocation();
  if (loc) {
    frame["source"] = llvm::json::Object{
        {"name", llvm::sys::path::filename(loc->first).str()},
        {"path", loc->first},
    };
    frame["line"] = static_cast<int64_t>(loc->second);
  }

  stackFrames.push_back(std::move(frame));

  resp.body["stackFrames"] = std::move(stackFrames);
  resp.body["totalFrames"] = 1;

  return resp;
}

Response DAPServer::handleScopes(const Request &req) {
  Response resp(req, true);

  int frameId = 0;
  if (auto fid = req.arguments.getInteger("frameId"))
    frameId = *fid;

  auto scopes = varRefs.getScopes(session.getState(), frameId);

  llvm::json::Array scopesArray;
  for (const auto &scope : scopes) {
    llvm::json::Object scopeObj{
        {"name", scope.name},
        {"variablesReference", scope.variablesReference},
        {"expensive", false},
    };
    scopesArray.push_back(std::move(scopeObj));
  }

  resp.body["scopes"] = std::move(scopesArray);
  return resp;
}

Response DAPServer::handleVariables(const Request &req) {
  Response resp(req, true);

  int varRef = 0;
  if (auto ref = req.arguments.getInteger("variablesReference"))
    varRef = *ref;

  auto variables = varRefs.getVariables(session.getState(), varRef);

  llvm::json::Array varsArray;
  for (const auto &var : variables) {
    llvm::json::Object varObj{
        {"name", var.name},
        {"value", var.value},
        {"type", var.type},
        {"variablesReference", var.variablesReference},
    };
    if (!var.evaluateName.empty())
      varObj["evaluateName"] = var.evaluateName;
    varsArray.push_back(std::move(varObj));
  }

  resp.body["variables"] = std::move(varsArray);
  return resp;
}

Response DAPServer::handleSetVariable(const Request &req) {
  Response resp(req, true);

  std::string name;
  std::string value;

  if (auto n = req.arguments.getString("name"))
    name = n->str();
  if (auto v = req.arguments.getString("value"))
    value = v->str();

  auto parsedVal = SignalValue::fromString(value, 32);
  if (!parsedVal) {
    resp.success = false;
    resp.message = "Invalid value: " + value;
    return resp;
  }

  if (!session.setSignal(name, *parsedVal)) {
    resp.success = false;
    resp.message = "Failed to set signal: " + name;
    return resp;
  }

  resp.body["value"] = "0x" + parsedVal->toHexString();
  return resp;
}

Response DAPServer::handleEvaluate(const Request &req) {
  Response resp(req, true);

  std::string expression;
  if (auto expr = req.arguments.getString("expression"))
    expression = expr->str();

  auto result = session.evaluate(expression);
  if (!result.success) {
    resp.success = false;
    resp.message = result.error;
    return resp;
  }

  resp.body["result"] = "0x" + result.value->toHexString();
  resp.body["variablesReference"] = 0;
  return resp;
}

Response DAPServer::handleContinue(const Request &req) {
  Response resp(req, true);
  resp.body["allThreadsContinued"] = true;

  // Continue in background (non-blocking)
  session.continueExec();

  return resp;
}

Response DAPServer::handleNext(const Request &req) {
  Response resp(req, true);

  // Step one clock cycle
  session.step(1);

  return resp;
}

Response DAPServer::handleStepIn(const Request &req) {
  Response resp(req, true);

  // For hardware, step in = step delta
  session.stepDelta();

  return resp;
}

Response DAPServer::handleStepOut(const Request &req) {
  Response resp(req, true);

  // For hardware, step out = step clock
  session.step(1);

  return resp;
}

Response DAPServer::handlePause(const Request &req) {
  Response resp(req, true);

  // Signal interrupt - would need threading support
  sendStoppedEvent("pause", "Paused");

  return resp;
}

Response DAPServer::handleTerminate(const Request &req) {
  Response resp(req, true);

  session.stop();
  sendTerminatedEvent();

  return resp;
}

Response DAPServer::handleRestart(const Request &req) {
  Response resp(req, true);

  session.reset();
  session.start();
  sendStoppedEvent("entry", "Restarted");

  return resp;
}

Response DAPServer::handleCompletions(const Request &req) {
  Response resp(req, true);

  std::string text;
  if (auto t = req.arguments.getString("text"))
    text = t->str();

  llvm::json::Array targets;

  // Complete signal names
  auto signals = session.findSignals(text);
  for (const auto &sig : signals) {
    targets.push_back(llvm::json::Object{
        {"label", sig.fullPath},
        {"text", sig.fullPath},
        {"type", "variable"},
    });
  }

  resp.body["targets"] = std::move(targets);
  return resp;
}

Response DAPServer::handleModules(const Request &req) {
  Response resp(req, true);

  llvm::json::Array modules;

  // List hierarchy as modules
  const auto *root = session.getState().getRootScope();
  if (root) {
    std::function<void(const debug::Scope *)> addModule;
    addModule = [&](const debug::Scope *scope) {
      modules.push_back(llvm::json::Object{
          {"id", scope->getFullPath()},
          {"name", scope->getName()},
          {"path", scope->getFullPath()},
      });
      for (const auto &child : scope->getChildren())
        addModule(child.get());
    };
    addModule(root);
  }

  resp.body["modules"] = std::move(modules);
  resp.body["totalModules"] = static_cast<int64_t>(modules.size());

  return resp;
}

void DAPServer::sendStoppedEvent(StringRef reason, StringRef description,
                                 int threadId) {
  Event event("stopped");
  event.seq = nextSeq++;
  event.body["reason"] = reason.str();
  event.body["threadId"] = threadId;
  event.body["allThreadsStopped"] = true;

  if (!description.empty())
    event.body["description"] = description.str();

  sendEvent(event);
}

void DAPServer::sendTerminatedEvent() {
  Event event("terminated");
  event.seq = nextSeq++;
  sendEvent(event);
}

void DAPServer::sendOutputEvent(StringRef category, StringRef output) {
  Event event("output");
  event.seq = nextSeq++;
  event.body["category"] = category.str();
  event.body["output"] = output.str();
  sendEvent(event);
}

void DAPServer::sendInitializedEvent() {
  Event event("initialized");
  event.seq = nextSeq++;
  sendEvent(event);
}

void DAPServer::sendEvent(const Event &event) {
  if (!output)
    return;

  DAPTransport transport(std::cin, *output);
  transport.writeMessage(event.toJSON());
}

void DAPServer::sendResponse(const Response &response) {
  if (!output)
    return;

  DAPTransport transport(std::cin, *output);
  transport.writeMessage(response.toJSON());
}

llvm::json::Object DAPServer::DAPBreakpoint::toJSON() const {
  llvm::json::Object obj{
      {"id", id},
      {"verified", verified},
      {"line", line},
  };

  if (!message.empty())
    obj["message"] = message;
  if (column)
    obj["column"] = *column;
  if (endLine)
    obj["endLine"] = *endLine;
  if (endColumn)
    obj["endColumn"] = *endColumn;
  if (!source.empty())
    obj["source"] = llvm::json::Object{{"path", source}};

  return obj;
}

DAPServer::DAPBreakpoint DAPServer::toDAPBreakpoint(const Breakpoint &bp) {
  DAPBreakpoint dapBp;
  dapBp.id = nextDAPBreakpointId++;
  dapBp.verified = bp.isEnabled();
  dapBp.message = bp.getDescription();

  if (bp.getType() == Breakpoint::Type::Line) {
    const auto &lineBp = static_cast<const LineBreakpoint &>(bp);
    dapBp.line = lineBp.getLine();
    dapBp.source = lineBp.getFile().str();
  }

  return dapBp;
}

//===----------------------------------------------------------------------===//
// DAPTransport Implementation
//===----------------------------------------------------------------------===//

DAPTransport::DAPTransport(std::istream &in, llvm::raw_ostream &out)
    : in(in), out(out) {}

std::optional<llvm::json::Object> DAPTransport::readMessage() {
  // Read Content-Length header
  auto lengthOpt = readContentLength();
  if (!lengthOpt)
    return std::nullopt;

  size_t length = *lengthOpt;

  // Read the JSON body
  std::string body(length, '\0');
  in.read(&body[0], length);

  if (in.gcount() != static_cast<std::streamsize>(length))
    return std::nullopt;

  // Parse JSON
  auto parsed = llvm::json::parse(body);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return std::nullopt;
  }

  auto *obj = parsed->getAsObject();
  if (!obj)
    return std::nullopt;

  return llvm::json::Object(*obj);
}

void DAPTransport::writeMessage(const llvm::json::Object &msg) {
  std::lock_guard<std::mutex> lock(writeMutex);

  std::string body;
  llvm::raw_string_ostream bodyStream(body);
  llvm::json::Object msgCopy(msg);
  bodyStream << llvm::json::Value(std::move(msgCopy));
  bodyStream.flush();

  out << "Content-Length: " << body.size() << "\r\n\r\n" << body;
  out.flush();
}

std::optional<size_t> DAPTransport::readContentLength() {
  std::string line;

  while (std::getline(in, line)) {
    // Remove trailing \r if present
    if (!line.empty() && line.back() == '\r')
      line.pop_back();

    // Empty line marks end of headers
    if (line.empty())
      break;

    // Look for Content-Length header
    if (line.rfind("Content-Length: ", 0) == 0) {
      std::string lengthStr = line.substr(16);
      size_t length;
      if (llvm::StringRef(lengthStr).getAsInteger(10, length))
        return std::nullopt;
      return length;
    }
  }

  return std::nullopt;
}
