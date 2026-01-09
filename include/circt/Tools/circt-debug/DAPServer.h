//===- DAPServer.h - Debug Adapter Protocol Server --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the Debug Adapter Protocol (DAP) server for
// integrating the CIRCT debugger with IDEs like VS Code.
//
// DAP Protocol: https://microsoft.github.io/debug-adapter-protocol/
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_DEBUG_DAPSERVER_H
#define CIRCT_TOOLS_CIRCT_DEBUG_DAPSERVER_H

#include "circt/Support/JSON.h"
#include "circt/Tools/circt-debug/Debug.h"
#include "circt/Tools/circt-debug/DebugSession.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace circt {
namespace debug {
namespace dap {

//===----------------------------------------------------------------------===//
// DAP Message Types
//===----------------------------------------------------------------------===//

/// Base class for all DAP messages.
struct Message {
  int seq = 0;
  std::string type;

  virtual ~Message() = default;
  virtual llvm::json::Object toJSON() const;
};

/// A DAP request message from the client.
struct Request : public Message {
  std::string command;
  llvm::json::Object arguments;

  Request() { type = "request"; }
  llvm::json::Object toJSON() const override;
};

/// A DAP response message to the client.
struct Response : public Message {
  int requestSeq = 0;
  bool success = true;
  std::string command;
  std::string message;
  llvm::json::Object body;

  Response() { type = "response"; }
  Response(const Request &req, bool success = true);
  llvm::json::Object toJSON() const override;
};

/// A DAP event message to the client.
struct Event : public Message {
  std::string event;
  llvm::json::Object body;

  Event() { type = "event"; }
  Event(StringRef eventName);
  llvm::json::Object toJSON() const override;
};

//===----------------------------------------------------------------------===//
// DAP Capabilities
//===----------------------------------------------------------------------===//

/// Describes the capabilities of this debug adapter.
struct Capabilities {
  bool supportsConfigurationDoneRequest = true;
  bool supportsFunctionBreakpoints = false;
  bool supportsConditionalBreakpoints = true;
  bool supportsHitConditionalBreakpoints = true;
  bool supportsEvaluateForHovers = true;
  bool supportsStepBack = false;
  bool supportsSetVariable = true;
  bool supportsRestartFrame = false;
  bool supportsGotoTargetsRequest = false;
  bool supportsStepInTargetsRequest = false;
  bool supportsCompletionsRequest = true;
  bool supportsModulesRequest = true;
  bool supportsRestartRequest = true;
  bool supportsExceptionOptions = false;
  bool supportsValueFormattingOptions = true;
  bool supportsExceptionInfoRequest = false;
  bool supportTerminateDebuggee = true;
  bool supportsDelayedStackTraceLoading = false;
  bool supportsLoadedSourcesRequest = false;
  bool supportsLogPoints = false;
  bool supportsTerminateThreadsRequest = false;
  bool supportsSetExpression = false;
  bool supportsTerminateRequest = true;
  bool supportsDataBreakpoints = true;
  bool supportsReadMemoryRequest = false;
  bool supportsDisassembleRequest = false;
  bool supportsCancelRequest = false;
  bool supportsBreakpointLocationsRequest = false;
  bool supportsClipboardContext = false;
  bool supportsSteppingGranularity = true;
  bool supportsInstructionBreakpoints = false;
  bool supportsExceptionFilterOptions = false;

  llvm::json::Object toJSON() const;
};

//===----------------------------------------------------------------------===//
// DAP Variable References
//===----------------------------------------------------------------------===//

/// Manages variable references for the DAP protocol.
/// Variables are identified by integer references in DAP.
class VariableReferenceManager {
public:
  /// Variable scope types.
  enum class ScopeType { Local, Global, Hierarchy };

  /// A variable scope.
  struct Scope {
    ScopeType type;
    std::string name;
    std::string path;
    int variablesReference;
  };

  /// A variable entry.
  struct Variable {
    std::string name;
    std::string value;
    std::string type;
    int variablesReference = 0; // 0 = no children
    std::string evaluateName;
  };

  VariableReferenceManager();

  /// Create a new reference for a scope path.
  int createReference(StringRef path);

  /// Get the path for a reference.
  std::optional<std::string> getPath(int reference) const;

  /// Clear all references (call at each stop).
  void clear();

  /// Get scopes for a frame.
  std::vector<Scope> getScopes(const SimState &state, int frameId);

  /// Get variables for a reference.
  std::vector<Variable> getVariables(const SimState &state, int reference);

private:
  int nextReference = 1;
  llvm::DenseMap<int, std::string> referenceToPath;
  llvm::StringMap<int> pathToReference;
};

//===----------------------------------------------------------------------===//
// DAP Server
//===----------------------------------------------------------------------===//

/// Debug Adapter Protocol server.
class DAPServer {
public:
  DAPServer(DebugSession &session);
  ~DAPServer();

  /// Run the server, reading from stdin and writing to stdout.
  int run();

  /// Run the server with custom streams.
  int run(std::istream &in, llvm::raw_ostream &out);

  /// Stop the server.
  void stop();

  /// Check if the server is running.
  bool isRunning() const { return running; }

private:
  //==========================================================================
  // Message I/O
  //==========================================================================

  /// Read a DAP message from input.
  std::optional<Request> readRequest();

  /// Send a response.
  void sendResponse(const Response &response);

  /// Send an event.
  void sendEvent(const Event &event);

  /// Send a raw JSON message.
  void sendMessage(const llvm::json::Object &msg);

  //==========================================================================
  // Request Handlers
  //==========================================================================

  Response handleRequest(const Request &req);

  Response handleInitialize(const Request &req);
  Response handleLaunch(const Request &req);
  Response handleAttach(const Request &req);
  Response handleDisconnect(const Request &req);
  Response handleConfigurationDone(const Request &req);
  Response handleSetBreakpoints(const Request &req);
  Response handleSetDataBreakpoints(const Request &req);
  Response handleThreads(const Request &req);
  Response handleStackTrace(const Request &req);
  Response handleScopes(const Request &req);
  Response handleVariables(const Request &req);
  Response handleSetVariable(const Request &req);
  Response handleEvaluate(const Request &req);
  Response handleContinue(const Request &req);
  Response handleNext(const Request &req);
  Response handleStepIn(const Request &req);
  Response handleStepOut(const Request &req);
  Response handlePause(const Request &req);
  Response handleTerminate(const Request &req);
  Response handleRestart(const Request &req);
  Response handleCompletions(const Request &req);
  Response handleModules(const Request &req);

  //==========================================================================
  // Event Helpers
  //==========================================================================

  /// Send stopped event.
  void sendStoppedEvent(StringRef reason, StringRef description = "",
                        int threadId = 1);

  /// Send terminated event.
  void sendTerminatedEvent();

  /// Send output event.
  void sendOutputEvent(StringRef category, StringRef output);

  /// Send initialized event.
  void sendInitializedEvent();

  //==========================================================================
  // Breakpoint Management
  //==========================================================================

  /// DAP breakpoint info.
  struct DAPBreakpoint {
    int id;
    bool verified;
    std::string message;
    int line;
    std::optional<int> column;
    std::optional<int> endLine;
    std::optional<int> endColumn;
    std::string source;

    llvm::json::Object toJSON() const;
  };

  /// Convert CIRCT breakpoint to DAP breakpoint.
  DAPBreakpoint toDAPBreakpoint(const Breakpoint &bp);

  //==========================================================================
  // State
  //==========================================================================

  DebugSession &session;
  Capabilities capabilities;
  VariableReferenceManager varRefs;

  std::istream *input = nullptr;
  llvm::raw_ostream *output = nullptr;
  std::mutex outputMutex;

  std::atomic<bool> running{false};
  std::atomic<bool> initialized{false};
  std::atomic<bool> launched{false};
  int nextSeq = 1;

  /// Map from DAP breakpoint ID to CIRCT breakpoint ID.
  llvm::DenseMap<int, unsigned> dapToCirctBreakpoints;
  int nextDAPBreakpointId = 1;

  /// Source file to breakpoints mapping.
  llvm::StringMap<std::vector<DAPBreakpoint>> sourceBreakpoints;
};

//===----------------------------------------------------------------------===//
// DAP Transport
//===----------------------------------------------------------------------===//

/// Parses DAP JSON-RPC style messages.
class DAPTransport {
public:
  DAPTransport(std::istream &in, llvm::raw_ostream &out);

  /// Read the next message.
  std::optional<llvm::json::Object> readMessage();

  /// Write a message.
  void writeMessage(const llvm::json::Object &msg);

private:
  /// Read Content-Length header.
  std::optional<size_t> readContentLength();

  std::istream &in;
  llvm::raw_ostream &out;
  std::mutex writeMutex;
};

} // namespace dap
} // namespace debug
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_DEBUG_DAPSERVER_H
