//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogServer.h
//
// This header declares VerilogServer, the top-level coordinator of the CIRCT
// Verilog LSP server. It manages all open files and acts as the entry point for
// language server operations such as “didOpen”, “didChange”, “didClose”,
// “definition”, and “references”.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVER_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGSERVER_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Protocol.h"

#include <memory>
#include <optional>
#include <vector>

namespace mlir {
namespace lsp {
struct Diagnostic;
struct Position;
struct Location;
struct TextDocumentContentChangeEvent;
class URIForFile;
} // namespace lsp
} // namespace mlir

namespace circt {
namespace lsp {
struct VerilogServerOptions;
using TextDocumentContentChangeEvent =
    llvm::lsp::TextDocumentContentChangeEvent;
using URIForFile = llvm::lsp::URIForFile;
using Diagnostic = llvm::lsp::Diagnostic;
struct WorkspaceSymbol {
  std::string name;
  llvm::lsp::SymbolKind kind;
  llvm::lsp::Location location;
  std::string containerName;
};

/// This class implements all of the Verilog related functionality necessary for
/// a language server. This class allows for keeping the Verilog specific logic
/// separate from the logic that involves LSP server/client communication.
class VerilogServer {
public:
  VerilogServer(const circt::lsp::VerilogServerOptions &options);
  ~VerilogServer();

  /// Add the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void addDocument(const URIForFile &uri, llvm::StringRef contents,
                   int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Update the document, with the provided `version`, at the given URI. Any
  /// diagnostics emitted for this document should be added to `diagnostics`.
  void updateDocument(const URIForFile &uri,
                      llvm::ArrayRef<TextDocumentContentChangeEvent> changes,
                      int64_t version, std::vector<Diagnostic> &diagnostics);

  /// Remove the document with the given uri. Returns the version of the removed
  /// document, or std::nullopt if the uri did not have a corresponding document
  /// within the server.
  std::optional<int64_t> removeDocument(const URIForFile &uri);

  /// Return the locations of the object pointed at by the given position.
  void getLocationsOf(const URIForFile &uri, const llvm::lsp::Position &defPos,
                      std::vector<llvm::lsp::Location> &locations);

  /// Find all references of the object pointed at by the given position.
  /// If includeDeclaration is true, also include the symbol's declaration.
  void findReferencesOf(const URIForFile &uri, const llvm::lsp::Position &pos,
                        bool includeDeclaration,
                        std::vector<llvm::lsp::Location> &references);

  /// Document highlight kind - indicates whether a reference is a read, write,
  /// or text reference.
  enum class DocumentHighlightKind { Text = 1, Read = 2, Write = 3 };

  /// A document highlight represents a range in a document which deserves
  /// special attention.
  struct DocumentHighlight {
    llvm::lsp::Range range;
    DocumentHighlightKind kind = DocumentHighlightKind::Text;
  };

  /// Return document highlights for all occurrences of the symbol at the given
  /// position within the same document.
  void getDocumentHighlights(const URIForFile &uri,
                             const llvm::lsp::Position &pos,
                             std::vector<DocumentHighlight> &highlights);

  /// Return hover information for the object at the given position.
  std::optional<llvm::lsp::Hover> getHover(const URIForFile &uri,
                                           const llvm::lsp::Position &pos);

  /// Return the document symbols for the given document.
  void getDocumentSymbols(const URIForFile &uri,
                          std::vector<llvm::lsp::DocumentSymbol> &symbols);

  /// Return workspace symbols matching the query string.
  void getWorkspaceSymbols(llvm::StringRef query,
                           std::vector<WorkspaceSymbol> &symbols);

  /// Return completion items for the given position.
  void getCompletions(const URIForFile &uri, const llvm::lsp::Position &pos,
                      llvm::lsp::CompletionList &completions);

  /// Return code actions for the given range and diagnostics.
  void getCodeActions(const URIForFile &uri, const llvm::lsp::Range &range,
                      const std::vector<llvm::lsp::Diagnostic> &diagnostics,
                      std::vector<llvm::lsp::CodeAction> &codeActions);

  /// Prepare a rename operation at the given position.
  std::optional<std::pair<llvm::lsp::Range, std::string>>
  prepareRename(const URIForFile &uri, const llvm::lsp::Position &pos);

  /// Perform a rename operation.
  std::optional<llvm::lsp::WorkspaceEdit>
  renameSymbol(const URIForFile &uri, const llvm::lsp::Position &pos,
               llvm::StringRef newName);

  /// Return document links for include directives.
  void getDocumentLinks(const URIForFile &uri,
                        std::vector<llvm::lsp::DocumentLink> &links);

  /// Return semantic tokens for the entire document.
  void getSemanticTokens(const URIForFile &uri,
                         std::vector<uint32_t> &data);

  /// Return inlay hints for the given range.
  void getInlayHints(const URIForFile &uri, const llvm::lsp::Range &range,
                     std::vector<llvm::lsp::InlayHint> &hints);

  /// Return signature help for function/task calls at the given position.
  llvm::lsp::SignatureHelp getSignatureHelp(const URIForFile &uri,
                                            const llvm::lsp::Position &pos);

  /// Initialize workspace from LSP initialize parameters.
  void initializeWorkspace(const llvm::json::Value &initParams);

  /// Handle workspace folder changes.
  void workspaceFoldersChanged(llvm::ArrayRef<std::string> added,
                               llvm::ArrayRef<std::string> removed);

  /// Handle file change notifications (for config file reloading).
  void onFileChanged(const URIForFile &uri);

  /// Get workspace configuration as JSON for the client.
  llvm::json::Value getWorkspaceConfiguration() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace lsp
} // namespace circt

#endif
