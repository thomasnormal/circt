//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogTextFile.h
//
// This header declares VerilogTextFile, a lightweight LSP-facing wrapper
// around VerilogDocument that represents an open, editable Verilog source
// buffer in the CIRCT Verilog LSP server.
//
// VerilogTextFile owns the current text contents and version (as tracked by
// the LSP client) and rebuilds its VerilogDocument whenever the file is
// opened or updated. It also forwards language queries (e.g. “go to
// definition”, “find references”) to the underlying VerilogDocument, which
// performs Slang-based parsing, indexing, and location translation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGTEXTFILE_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGTEXTFILE_H_

#include "llvm/Support/LSP/Protocol.h"

#include "VerilogDocument.h"

namespace circt {
namespace lsp {

struct VerilogServerContext;

/// This class represents a text file containing one or more Verilog
/// documents.
class VerilogTextFile {
public:
  /// Initialize a new VerilogTextFile and its VerilogDocument.
  /// Thread-safe; acquires contentMutex and docMutex.
  VerilogTextFile(VerilogServerContext &globalContext,
                  const llvm::lsp::URIForFile &uri,
                  llvm::StringRef fileContents, int64_t version,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);

  /// Return the current version of this text file.
  /// Thread-safe.
  int64_t getVersion() const { return version; }

  /// Update the file to the new version using the provided set of content
  /// changes. Returns failure if the update was unsuccessful.
  /// Thread-safe; acquires contentMutex and docMutex.
  llvm::LogicalResult
  update(const llvm::lsp::URIForFile &uri, int64_t newVersion,
         llvm::ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
         std::vector<llvm::lsp::Diagnostic> &diagnostics);

  /// Return position of definition of an object pointed to by pos.
  /// Thread-safe; acquires docMutex.
  void getLocationsOf(const llvm::lsp::URIForFile &uri,
                      llvm::lsp::Position defPos,
                      std::vector<llvm::lsp::Location> &locations);

  /// Return all references to an object pointed to by pos.
  /// If includeDeclaration is true, also include the symbol's declaration.
  /// Thread-safe; acquires docMutex.
  void findReferencesOf(const llvm::lsp::URIForFile &uri,
                        llvm::lsp::Position pos, bool includeDeclaration,
                        std::vector<llvm::lsp::Location> &references);

  /// Return document highlights for all occurrences of the symbol at the
  /// given position within the same document.
  /// Thread-safe; acquires docMutex.
  void getDocumentHighlights(
      const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos,
      std::vector<VerilogDocument::DocumentHighlight> &highlights);

  /// Return hover information for the object at the given position.
  /// Thread-safe; acquires docMutex.
  std::optional<llvm::lsp::Hover> getHover(const llvm::lsp::URIForFile &uri,
                                           llvm::lsp::Position pos);

  /// Return the document symbols for this file.
  /// Thread-safe; acquires docMutex.
  void getDocumentSymbols(const llvm::lsp::URIForFile &uri,
                          std::vector<llvm::lsp::DocumentSymbol> &symbols);

  /// Return completion items for the given position.
  /// Thread-safe; acquires docMutex.
  void getCompletions(const llvm::lsp::URIForFile &uri,
                      llvm::lsp::Position pos,
                      llvm::lsp::CompletionList &completions);

  /// Return code actions for the given range and diagnostics.
  /// Thread-safe; acquires docMutex.
  void getCodeActions(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Range &range,
                      const std::vector<llvm::lsp::Diagnostic> &diagnostics,
                      std::vector<llvm::lsp::CodeAction> &codeActions);

  /// Prepare a rename operation at the given position.
  /// Thread-safe; acquires docMutex.
  std::optional<std::pair<llvm::lsp::Range, std::string>>
  prepareRename(const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos);

  /// Perform a rename operation.
  /// Thread-safe; acquires docMutex.
  std::optional<llvm::lsp::WorkspaceEdit>
  renameSymbol(const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos,
               llvm::StringRef newName);

  /// Return document links for include directives.
  /// Thread-safe; acquires docMutex.
  void getDocumentLinks(const llvm::lsp::URIForFile &uri,
                        std::vector<llvm::lsp::DocumentLink> &links);

  /// Return semantic tokens for the entire document.
  /// Thread-safe; acquires docMutex.
  void getSemanticTokens(const llvm::lsp::URIForFile &uri,
                         std::vector<uint32_t> &data);

  /// Return inlay hints for the given range.
  /// Thread-safe; acquires docMutex.
  void getInlayHints(const llvm::lsp::URIForFile &uri,
                     const llvm::lsp::Range &range,
                     std::vector<llvm::lsp::InlayHint> &hints);

  /// Return signature help for function/task calls at the given position.
  /// Thread-safe; acquires docMutex.
  llvm::lsp::SignatureHelp getSignatureHelp(const llvm::lsp::URIForFile &uri,
                                            llvm::lsp::Position pos);

  /// Prepare call hierarchy at the given position.
  /// Thread-safe; acquires docMutex.
  std::optional<VerilogDocument::CallHierarchyItem>
  prepareCallHierarchy(const llvm::lsp::URIForFile &uri,
                       llvm::lsp::Position pos);

  /// Get incoming calls for a call hierarchy item.
  /// Thread-safe; acquires docMutex.
  void getIncomingCalls(
      const VerilogDocument::CallHierarchyItem &item,
      std::vector<VerilogDocument::CallHierarchyIncomingCall> &calls);

  /// Get outgoing calls from a call hierarchy item.
  /// Thread-safe; acquires docMutex.
  void getOutgoingCalls(
      const VerilogDocument::CallHierarchyItem &item,
      std::vector<VerilogDocument::CallHierarchyOutgoingCall> &calls);

  /// Format the entire document.
  /// Thread-safe; acquires docMutex.
  void formatDocument(const llvm::lsp::URIForFile &uri,
                      const VerilogDocument::FormattingOptions &options,
                      std::vector<llvm::lsp::TextEdit> &edits);

  /// Format a range within the document.
  /// Thread-safe; acquires docMutex.
  void formatRange(const llvm::lsp::URIForFile &uri,
                   const llvm::lsp::Range &range,
                   const VerilogDocument::FormattingOptions &options,
                   std::vector<llvm::lsp::TextEdit> &edits);

  /// Return code lenses for the document.
  /// Thread-safe; acquires docMutex.
  void getCodeLenses(const llvm::lsp::URIForFile &uri,
                     std::vector<VerilogDocument::CodeLensInfo> &lenses);

  /// Resolve a code lens with the given data.
  /// Thread-safe; acquires docMutex.
  bool resolveCodeLens(llvm::StringRef data,
                       VerilogDocument::CodeLensInfo &lens);

  /// Return document for read access.
  /// Thread-safe; acquires docMutex.
  std::shared_ptr<VerilogDocument> getDocument();

  /// Override document after update.
  /// Thread-safe; acquires docMutex.
  void setDocument(std::shared_ptr<VerilogDocument> newDoc);

private:
  /// Initialize the text file from the given file contents.
  /// NOT thread-safe. ONLY call with contentMutex acquired!
  /// Acquires docMutex.
  void initialize(const llvm::lsp::URIForFile &uri, int64_t newVersion,
                  std::vector<llvm::lsp::Diagnostic> &diagnostics);

  void initializeProjectDriver();

  VerilogServerContext &context;

  /// The full string contents of the file.
  std::string contents;

  /// The project-scale driver
  std::unique_ptr<slang::driver::Driver> projectDriver;
  std::vector<std::string> projectIncludeDirectories;
  /// A mutex to control updates of contents.
  /// Acquire BEFORE docMutex.
  std::shared_mutex contentMutex;

  /// The version of this file.
  int64_t version = 0;

  /// The chunks of this file. The order of these chunks is the order in which
  /// they appear in the text file.
  std::shared_ptr<circt::lsp::VerilogDocument> document;

  /// A mutex to control updates of document;
  /// Acquire AFTER contentMutex.
  std::shared_mutex docMutex;
};

} // namespace lsp
} // namespace circt

#endif
