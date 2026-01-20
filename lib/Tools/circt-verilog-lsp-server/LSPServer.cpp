//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "Utils/PendingChanges.h"
#include "VerilogServerImpl/SemanticTokens.h"
#include "VerilogServerImpl/VerilogServer.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LSP/Transport.h"

#include <cstdint>
#include <mutex>
#include <optional>

#define DEBUG_TYPE "circt-verilog-lsp-server"

using namespace llvm;
using namespace llvm::lsp;

//===----------------------------------------------------------------------===//
// Custom LSP Types (not in base LLVM LSP library)
//===----------------------------------------------------------------------===//

namespace {

/// Parameters for textDocument/rename request.
struct RenameParams {
  TextDocumentIdentifier textDocument;
  Position position;
  std::string newName;
};

inline bool fromJSON(const json::Value &value, RenameParams &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("position", result.position) && o.map("newName", result.newName);
}

/// Parameters for textDocument/semanticTokens/full request.
struct SemanticTokensParams {
  TextDocumentIdentifier textDocument;
};

inline bool fromJSON(const json::Value &value, SemanticTokensParams &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument);
}

/// Parameters for workspace/symbol request.
struct WorkspaceSymbolParams {
  std::string query;
};

inline bool fromJSON(const json::Value &value, WorkspaceSymbolParams &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("query", result.query);
}

/// Formatting options from LSP.
struct FormattingOptions {
  unsigned tabSize = 2;
  bool insertSpaces = true;
};

inline bool fromJSON(const json::Value &value, FormattingOptions &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("tabSize", result.tabSize) &&
         o.map("insertSpaces", result.insertSpaces);
}

/// Parameters for textDocument/formatting request.
struct DocumentFormattingParams {
  TextDocumentIdentifier textDocument;
  FormattingOptions options;
};

inline bool fromJSON(const json::Value &value, DocumentFormattingParams &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("options", result.options);
}

/// Parameters for textDocument/rangeFormatting request.
struct DocumentRangeFormattingParams {
  TextDocumentIdentifier textDocument;
  Range range;
  FormattingOptions options;
};

inline bool fromJSON(const json::Value &value, DocumentRangeFormattingParams &result,
                     json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("textDocument", result.textDocument) &&
         o.map("range", result.range) && o.map("options", result.options);
}

} // namespace

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {

struct LSPServer {

  LSPServer(const circt::lsp::LSPServerOptions &options,
            circt::lsp::VerilogServer &server, JSONTransport &transport)
      : server(server), transport(transport),
        debounceOptions(circt::lsp::DebounceOptions::fromLSPOptions(options)) {}

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  void onInitialize(const InitializeParams &params,
                    Callback<json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change
  //===--------------------------------------------------------------------===//

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void onGoToDefinition(const TextDocumentPositionParams &params,
                        Callback<std::vector<Location>> reply);
  void onReference(const ReferenceParams &params,
                   Callback<std::vector<Location>> reply);

  //===--------------------------------------------------------------------===//
  // Document Highlight
  //===--------------------------------------------------------------------===//

  void onDocumentHighlight(const TextDocumentPositionParams &params,
                           Callback<json::Value> reply);

  //===--------------------------------------------------------------------===//
  // Hover
  //===--------------------------------------------------------------------===//

  void onHover(const TextDocumentPositionParams &params,
               Callback<std::optional<Hover>> reply);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  void onDocumentSymbol(const DocumentSymbolParams &params,
                        Callback<std::vector<DocumentSymbol>> reply);

  //===--------------------------------------------------------------------===//
  // Workspace Symbols
  //===--------------------------------------------------------------------===//

  void onWorkspaceSymbol(const WorkspaceSymbolParams &params,
                         Callback<json::Value> reply);

  //===--------------------------------------------------------------------===//
  // Auto-Completion
  //===--------------------------------------------------------------------===//

  void onCompletion(const CompletionParams &params,
                    Callback<CompletionList> reply);

  //===--------------------------------------------------------------------===//
  // Code Actions
  //===--------------------------------------------------------------------===//

  void onCodeAction(const CodeActionParams &params,
                    Callback<std::vector<CodeAction>> reply);

  //===--------------------------------------------------------------------===//
  // Rename Symbol
  //===--------------------------------------------------------------------===//

  void onPrepareRename(const TextDocumentPositionParams &params,
                       Callback<std::optional<Range>> reply);
  void onRename(const RenameParams &params, Callback<WorkspaceEdit> reply);

  //===--------------------------------------------------------------------===//
  // Document Links
  //===--------------------------------------------------------------------===//

  void onDocumentLink(const DocumentLinkParams &params,
                      Callback<std::vector<DocumentLink>> reply);

  //===--------------------------------------------------------------------===//
  // Semantic Tokens
  //===--------------------------------------------------------------------===//

  void onSemanticTokensFull(const SemanticTokensParams &params,
                            Callback<json::Value> reply);

  //===--------------------------------------------------------------------===//
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  void onInlayHints(const InlayHintsParams &params,
                    Callback<std::vector<InlayHint>> reply);

  //===--------------------------------------------------------------------===//
  // Signature Help
  //===--------------------------------------------------------------------===//

  void onSignatureHelp(const TextDocumentPositionParams &params,
                       Callback<SignatureHelp> reply);

  //===--------------------------------------------------------------------===//
  // Document Formatting
  //===--------------------------------------------------------------------===//

  void onDocumentFormatting(const DocumentFormattingParams &params,
                            Callback<std::vector<TextEdit>> reply);

  void onDocumentRangeFormatting(const DocumentRangeFormattingParams &params,
                                 Callback<std::vector<TextEdit>> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  circt::lsp::VerilogServer &server;
  JSONTransport &transport;

  /// A thread-safe version of `publishDiagnostics`
  void sendDiagnostics(const PublishDiagnosticsParams &p) {
    std::scoped_lock<std::mutex> lk(diagnosticsMutex);
    publishDiagnostics(p); // serialize the write
  }

  void
  setPublishDiagnostics(OutgoingNotification<PublishDiagnosticsParams> diag) {
    std::scoped_lock<std::mutex> lk(diagnosticsMutex);
    publishDiagnostics = std::move(diag);
  }

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  bool shutdownRequestReceived = false;

private:
  /// A mutex to serialize access to publishing diagnostics
  std::mutex diagnosticsMutex;
  /// An outgoing notification used to send diagnostics to the client when they
  /// are ready to be processed.
  OutgoingNotification<PublishDiagnosticsParams> publishDiagnostics;

  circt::lsp::PendingChangesMap pendingChanges;
  circt::lsp::DebounceOptions debounceOptions;
};

} // namespace
//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void LSPServer::onInitialize(const InitializeParams &params,
                             Callback<json::Value> reply) {
  // Note: LLVM's base InitializeParams doesn't include workspaceFolders.
  // Workspace folders are typically sent via workspace/didChangeWorkspaceFolders.
  // Pass an empty object for now - workspace will be configured via notifications.
  server.initializeWorkspace(json::Object{});

  // Send a response with the capabilities of this server.
  json::Object serverCaps{
      {
          "textDocumentSync",
          llvm::json::Object{
              {"openClose", true},
              {"change", (int)TextDocumentSyncKind::Incremental},
              {"save", true},

          },

      },
      {"definitionProvider", true},
      {"referencesProvider", true},
      {"documentHighlightProvider", true},
      {"hoverProvider", true},
      {"documentSymbolProvider", true},
      {"workspaceSymbolProvider", true},
      {"completionProvider",
       llvm::json::Object{
           {"triggerCharacters", llvm::json::Array{"."}},
           {"resolveProvider", false},
       }},
      {"signatureHelpProvider",
       llvm::json::Object{
           {"triggerCharacters", llvm::json::Array{"(", ","}},
           {"retriggerCharacters", llvm::json::Array{","}},
       }},
      {"codeActionProvider", true},
      {"renameProvider",
       llvm::json::Object{
           {"prepareProvider", true},
       }},
      {"documentLinkProvider",
       llvm::json::Object{
           {"resolveProvider", false},
       }},
      {"semanticTokensProvider", circt::lsp::getSemanticTokensOptions()},
      {"inlayHintProvider", true},
      {"documentFormattingProvider", true},
      {"documentRangeFormattingProvider", true},
      // Workspace capabilities
      {"workspace",
       llvm::json::Object{
           {"workspaceFolders",
            llvm::json::Object{
                {"supported", true},
                {"changeNotifications", true},
            }},
       }},
  };

  json::Object result{
      {{"serverInfo", json::Object{{"name", "circt-verilog-lsp-server"},
                                   {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::onInitialized(const InitializedParams &) {}
void LSPServer::onShutdown(const NoParams &, Callback<std::nullptr_t> reply) {
  shutdownRequestReceived = true;
  pendingChanges.abort();
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addDocument(params.textDocument.uri, params.textDocument.text,
                     params.textDocument.version, diagParams.diagnostics);

  // Publish any recorded diagnostics.
  sendDiagnostics(diagParams);
}

void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  pendingChanges.erase(params.textDocument.uri);
  std::optional<int64_t> version =
      server.removeDocument(params.textDocument.uri);
  if (!version)
    return;

  // Empty out the diagnostics shown for this document. This will clear out
  // anything currently displayed by the client for this document (e.g. in the
  // "Problems" pane of VSCode).
  sendDiagnostics(PublishDiagnosticsParams(params.textDocument.uri, *version));
}

void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  pendingChanges.debounceAndUpdate(
      params, debounceOptions,
      [this, params](std::unique_ptr<circt::lsp::PendingChanges> result) {
        if (!result)
          return; // obsolete

        PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                            result->version);
        server.updateDocument(params.textDocument.uri, result->changes,
                              result->version, diagParams.diagnostics);

        sendDiagnostics(diagParams);
      });
}

//===----------------------------------------------------------------------===//
// Definitions and References
//===----------------------------------------------------------------------===//

void LSPServer::onGoToDefinition(const TextDocumentPositionParams &params,
                                 Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.getLocationsOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

void LSPServer::onReference(const ReferenceParams &params,
                            Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.findReferencesOf(params.textDocument.uri, params.position,
                          params.context.includeDeclaration, locations);
  reply(std::move(locations));
}

//===----------------------------------------------------------------------===//
// Document Highlight
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentHighlight(const TextDocumentPositionParams &params,
                                    Callback<json::Value> reply) {
  std::vector<circt::lsp::VerilogServer::DocumentHighlight> highlights;
  server.getDocumentHighlights(params.textDocument.uri, params.position,
                               highlights);

  // Convert to JSON manually since base LLVM LSP doesn't have DocumentHighlight
  if (highlights.empty()) {
    reply(json::Value(nullptr));
    return;
  }

  json::Array result;
  for (const auto &hl : highlights) {
    json::Object obj;
    obj["range"] = toJSON(hl.range);
    obj["kind"] = static_cast<int>(hl.kind);
    result.push_back(std::move(obj));
  }
  reply(std::move(result));
}

//===----------------------------------------------------------------------===//
// Hover
//===----------------------------------------------------------------------===//

void LSPServer::onHover(const TextDocumentPositionParams &params,
                        Callback<std::optional<Hover>> reply) {
  reply(server.getHover(params.textDocument.uri, params.position));
}

//===----------------------------------------------------------------------===//
// Document Symbols
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentSymbol(const DocumentSymbolParams &params,
                                 Callback<std::vector<DocumentSymbol>> reply) {
  std::vector<DocumentSymbol> symbols;
  server.getDocumentSymbols(params.textDocument.uri, symbols);
  reply(std::move(symbols));
}

void LSPServer::onWorkspaceSymbol(const WorkspaceSymbolParams &params,
                                  Callback<json::Value> reply) {
  std::vector<circt::lsp::WorkspaceSymbol> symbols;
  server.getWorkspaceSymbols(params.query, symbols);

  json::Array result;
  result.reserve(symbols.size());
  for (const auto &symbol : symbols) {
    json::Object obj;
    obj["name"] = symbol.name;
    obj["kind"] = static_cast<int>(symbol.kind);
    obj["location"] = llvm::lsp::toJSON(symbol.location);
    if (!symbol.containerName.empty())
      obj["containerName"] = symbol.containerName;
    result.push_back(std::move(obj));
  }
  reply(std::move(result));
}

//===----------------------------------------------------------------------===//
// Auto-Completion
//===----------------------------------------------------------------------===//

void LSPServer::onCompletion(const CompletionParams &params,
                             Callback<CompletionList> reply) {
  CompletionList completions;
  server.getCompletions(params.textDocument.uri, params.position, completions);
  reply(std::move(completions));
}

//===----------------------------------------------------------------------===//
// Code Actions
//===----------------------------------------------------------------------===//

void LSPServer::onCodeAction(const CodeActionParams &params,
                             Callback<std::vector<CodeAction>> reply) {
  std::vector<CodeAction> codeActions;
  server.getCodeActions(params.textDocument.uri, params.range,
                        params.context.diagnostics, codeActions);
  reply(std::move(codeActions));
}

//===----------------------------------------------------------------------===//
// Rename Symbol
//===----------------------------------------------------------------------===//

void LSPServer::onPrepareRename(const TextDocumentPositionParams &params,
                                Callback<std::optional<Range>> reply) {
  auto result = server.prepareRename(params.textDocument.uri, params.position);
  if (result)
    reply(result->first);
  else
    reply(std::nullopt);
}

void LSPServer::onRename(const RenameParams &params,
                         Callback<WorkspaceEdit> reply) {
  auto result =
      server.renameSymbol(params.textDocument.uri, params.position, params.newName);
  if (result)
    reply(std::move(*result));
  else
    reply(make_error<LSPError>("cannot rename symbol at this position",
                               ErrorCode::RequestFailed));
}

//===----------------------------------------------------------------------===//
// Document Links
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentLink(const DocumentLinkParams &params,
                               Callback<std::vector<DocumentLink>> reply) {
  std::vector<DocumentLink> links;
  server.getDocumentLinks(params.textDocument.uri, links);
  reply(std::move(links));
}

//===----------------------------------------------------------------------===//
// Semantic Tokens
//===----------------------------------------------------------------------===//

void LSPServer::onSemanticTokensFull(const SemanticTokensParams &params,
                                     Callback<json::Value> reply) {
  std::vector<uint32_t> data;
  server.getSemanticTokens(params.textDocument.uri, data);

  circt::lsp::SemanticTokensResult result;
  result.data = std::move(data);
  reply(circt::lsp::toJSON(result));
}

//===----------------------------------------------------------------------===//
// Inlay Hints
//===----------------------------------------------------------------------===//

void LSPServer::onInlayHints(const InlayHintsParams &params,
                             Callback<std::vector<InlayHint>> reply) {
  std::vector<InlayHint> hints;
  server.getInlayHints(params.textDocument.uri, params.range, hints);
  reply(std::move(hints));
}

//===----------------------------------------------------------------------===//
// Signature Help
//===----------------------------------------------------------------------===//

void LSPServer::onSignatureHelp(const TextDocumentPositionParams &params,
                                Callback<SignatureHelp> reply) {
  reply(server.getSignatureHelp(params.textDocument.uri, params.position));
}

//===----------------------------------------------------------------------===//
// Document Formatting
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentFormatting(const DocumentFormattingParams &params,
                                     Callback<std::vector<TextEdit>> reply) {
  std::vector<TextEdit> edits;
  circt::lsp::VerilogServer::FormattingOptions options;
  options.tabSize = params.options.tabSize;
  options.insertSpaces = params.options.insertSpaces;
  server.formatDocument(params.textDocument.uri, options, edits);
  reply(std::move(edits));
}

void LSPServer::onDocumentRangeFormatting(
    const DocumentRangeFormattingParams &params,
    Callback<std::vector<TextEdit>> reply) {
  std::vector<TextEdit> edits;
  circt::lsp::VerilogServer::FormattingOptions options;
  options.tabSize = params.options.tabSize;
  options.insertSpaces = params.options.insertSpaces;
  server.formatRange(params.textDocument.uri, params.range, options, edits);
  reply(std::move(edits));
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult
circt::lsp::runVerilogLSPServer(const circt::lsp::LSPServerOptions &options,
                                VerilogServer &server,
                                JSONTransport &transport) {
  LSPServer lspServer(options, server, transport);
  MessageHandler messageHandler(transport);

  // Diagnostics
  lspServer.setPublishDiagnostics(
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics"));

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);

  messageHandler.notification("textDocument/didChange", &lspServer,
                              &LSPServer::onDocumentDidChange);
  // Definitions and References
  messageHandler.method("textDocument/definition", &lspServer,
                        &LSPServer::onGoToDefinition);
  messageHandler.method("textDocument/references", &lspServer,
                        &LSPServer::onReference);

  // Document Highlight
  messageHandler.method("textDocument/documentHighlight", &lspServer,
                        &LSPServer::onDocumentHighlight);

  // Hover
  messageHandler.method("textDocument/hover", &lspServer,
                        &LSPServer::onHover);

  // Document Symbols
  messageHandler.method("textDocument/documentSymbol", &lspServer,
                        &LSPServer::onDocumentSymbol);

  // Workspace Symbols
  messageHandler.method("workspace/symbol", &lspServer,
                        &LSPServer::onWorkspaceSymbol);

  // Auto-Completion
  messageHandler.method("textDocument/completion", &lspServer,
                        &LSPServer::onCompletion);

  // Code Actions
  messageHandler.method("textDocument/codeAction", &lspServer,
                        &LSPServer::onCodeAction);

  // Rename Symbol
  messageHandler.method("textDocument/prepareRename", &lspServer,
                        &LSPServer::onPrepareRename);
  messageHandler.method("textDocument/rename", &lspServer,
                        &LSPServer::onRename);

  // Document Links
  messageHandler.method("textDocument/documentLink", &lspServer,
                        &LSPServer::onDocumentLink);

  // Semantic Tokens
  messageHandler.method("textDocument/semanticTokens/full", &lspServer,
                        &LSPServer::onSemanticTokensFull);

  // Inlay Hints
  messageHandler.method("textDocument/inlayHint", &lspServer,
                        &LSPServer::onInlayHints);

  // Signature Help
  messageHandler.method("textDocument/signatureHelp", &lspServer,
                        &LSPServer::onSignatureHelp);

  // Document Formatting
  messageHandler.method("textDocument/formatting", &lspServer,
                        &LSPServer::onDocumentFormatting);
  messageHandler.method("textDocument/rangeFormatting", &lspServer,
                        &LSPServer::onDocumentRangeFormatting);

  // Run the main loop of the transport.
  if (Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    consumeError(std::move(error));
    return failure();
  }

  return success(lspServer.shutdownRequestReceived);
}
