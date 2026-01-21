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
  int score = 0; // Fuzzy match score for sorting (higher is better)
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

  //===--------------------------------------------------------------------===//
  // Call Hierarchy
  //===--------------------------------------------------------------------===//

  /// Call hierarchy item structure for LSP.
  struct CallHierarchyItem {
    std::string name;
    llvm::lsp::SymbolKind kind;
    std::string detail;
    URIForFile uri;
    llvm::lsp::Range range;
    llvm::lsp::Range selectionRange;
    std::string data;
  };

  /// Incoming call structure for LSP.
  struct CallHierarchyIncomingCall {
    CallHierarchyItem from;
    std::vector<llvm::lsp::Range> fromRanges;
  };

  /// Outgoing call structure for LSP.
  struct CallHierarchyOutgoingCall {
    CallHierarchyItem to;
    std::vector<llvm::lsp::Range> fromRanges;
  };

  /// Prepare call hierarchy at the given position.
  std::optional<CallHierarchyItem>
  prepareCallHierarchy(const URIForFile &uri, const llvm::lsp::Position &pos);

  /// Get incoming calls for a call hierarchy item.
  void getIncomingCalls(const CallHierarchyItem &item,
                        std::vector<CallHierarchyIncomingCall> &calls);

  /// Get outgoing calls from a call hierarchy item.
  void getOutgoingCalls(const CallHierarchyItem &item,
                        std::vector<CallHierarchyOutgoingCall> &calls);

  //===--------------------------------------------------------------------===//
  // Type Hierarchy
  //===--------------------------------------------------------------------===//

  /// Type hierarchy item structure for LSP.
  struct TypeHierarchyItem {
    std::string name;
    llvm::lsp::SymbolKind kind;
    std::string detail;
    URIForFile uri;
    llvm::lsp::Range range;
    llvm::lsp::Range selectionRange;
    std::string data;
  };

  /// Prepare type hierarchy at the given position.
  std::optional<TypeHierarchyItem>
  prepareTypeHierarchy(const URIForFile &uri, const llvm::lsp::Position &pos);

  /// Get supertypes (parent classes) for a type hierarchy item.
  void getSupertypes(const TypeHierarchyItem &item,
                     std::vector<TypeHierarchyItem> &supertypes);

  /// Get subtypes (child classes) for a type hierarchy item.
  void getSubtypes(const TypeHierarchyItem &item,
                   std::vector<TypeHierarchyItem> &subtypes);

  /// Formatting options for document formatting.
  struct FormattingOptions {
    /// Number of spaces for indentation (ignored if insertSpaces is false).
    unsigned tabSize = 2;
    /// Use spaces for indentation instead of tabs.
    bool insertSpaces = true;
  };

  /// Format the entire document.
  void formatDocument(const URIForFile &uri, const FormattingOptions &options,
                      std::vector<llvm::lsp::TextEdit> &edits);

  /// Format a range within the document.
  void formatRange(const URIForFile &uri, const llvm::lsp::Range &range,
                   const FormattingOptions &options,
                   std::vector<llvm::lsp::TextEdit> &edits);

  //===--------------------------------------------------------------------===//
  // Code Lens
  //===--------------------------------------------------------------------===//

  /// Code lens information for LSP.
  struct CodeLensInfo {
    /// The range in which this code lens is valid.
    llvm::lsp::Range range;
    /// The command title (text shown to user).
    std::string title;
    /// The command identifier.
    std::string command;
    /// Arguments for the command.
    std::vector<std::string> commandArguments;
    /// Data for lazy resolution.
    std::string data;
  };

  /// Return code lenses for the given document.
  void getCodeLenses(const URIForFile &uri,
                     std::vector<CodeLensInfo> &lenses);

  /// Resolve a code lens with the given data.
  bool resolveCodeLens(llvm::StringRef data, CodeLensInfo &lens);

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
