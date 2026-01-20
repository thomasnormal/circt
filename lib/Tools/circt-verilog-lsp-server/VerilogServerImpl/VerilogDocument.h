//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogDocument.h
//
// This header declares the VerilogDocument class, which models a single open
// Verilog/SystemVerilog source file within the CIRCT Verilog LSP server. It
// owns a Slang driver and compilation unit for that file, provides diagnostic
// and indexing functionality, and exposes utilities for translating between
// LSP and Slang locations.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGDOCUMENT_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_VERILOGDOCUMENT_H_

#include "slang/ast/Compilation.h"
#include "slang/driver/Driver.h"
#include "slang/text/SourceManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LSP/Protocol.h"

#include "VerilogIndex.h"
#include "SemanticTokens.h"

namespace circt {
namespace lsp {

struct VerilogServerContext;

class VerilogDocument {
public:
  VerilogDocument(
      VerilogServerContext &globalContext, const llvm::lsp::URIForFile &uri,
      llvm::StringRef contents, std::vector<llvm::lsp::Diagnostic> &diagnostics,
      const slang::driver::Driver *projectDriver = nullptr,
      const std::vector<std::string> &projectIncludeDirectories = {});
  VerilogDocument(const VerilogDocument &) = delete;
  VerilogDocument &operator=(const VerilogDocument &) = delete;

  const llvm::lsp::URIForFile &getURI() const { return uri; }

  const slang::SourceManager &getSlangSourceManager() const {
    return driver.sourceManager;
  }

  // Return LSP location from slang location.
  llvm::lsp::Location getLspLocation(slang::SourceLocation loc) const;
  llvm::lsp::Location getLspLocation(slang::SourceRange range) const;
  llvm::lsp::Range getLspRange(slang::SourceRange range) const;

  slang::BufferID getMainBufferID() const { return mainBufferId; }

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void getLocationsOf(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Position &defPos,
                      std::vector<llvm::lsp::Location> &locations);

  void findReferencesOf(const llvm::lsp::URIForFile &uri,
                        const llvm::lsp::Position &pos, bool includeDeclaration,
                        std::vector<llvm::lsp::Location> &references);

  //===--------------------------------------------------------------------===//
  // Document Highlight
  //===--------------------------------------------------------------------===//

  /// Document highlight kind - indicates whether a reference is a read, write,
  /// or text reference.
  enum class DocumentHighlightKind { Text = 1, Read = 2, Write = 3 };

  /// A document highlight represents a range in a document which deserves
  /// special attention, typically used to highlight all occurrences of a symbol.
  struct DocumentHighlight {
    llvm::lsp::Range range;
    DocumentHighlightKind kind = DocumentHighlightKind::Text;
  };

  /// Return document highlights for all occurrences of the symbol at the given
  /// position within the same document.
  void getDocumentHighlights(const llvm::lsp::URIForFile &uri,
                             const llvm::lsp::Position &pos,
                             std::vector<DocumentHighlight> &highlights);

  //===--------------------------------------------------------------------===//
  // Hover Information
  //===--------------------------------------------------------------------===//

  /// Return hover information for the object at the given position.
  std::optional<llvm::lsp::Hover> getHover(const llvm::lsp::URIForFile &uri,
                                           const llvm::lsp::Position &pos);

  //===--------------------------------------------------------------------===//
  // Document Symbols
  //===--------------------------------------------------------------------===//

  /// Return document symbols for this document.
  void getDocumentSymbols(const llvm::lsp::URIForFile &uri,
                          std::vector<llvm::lsp::DocumentSymbol> &symbols);

  //===--------------------------------------------------------------------===//
  // Auto-Completion
  //===--------------------------------------------------------------------===//

  /// Return completion items for the given position.
  void getCompletions(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Position &pos,
                      llvm::lsp::CompletionList &completions);

  //===--------------------------------------------------------------------===//
  // Code Actions
  //===--------------------------------------------------------------------===//

  /// Return code actions for the given range and diagnostics.
  void getCodeActions(const llvm::lsp::URIForFile &uri,
                      const llvm::lsp::Range &range,
                      const std::vector<llvm::lsp::Diagnostic> &diagnostics,
                      std::vector<llvm::lsp::CodeAction> &codeActions);

  //===--------------------------------------------------------------------===//
  // Rename Symbol
  //===--------------------------------------------------------------------===//

  /// Prepare a rename operation at the given position.
  /// Returns the range of the symbol being renamed and its current name.
  std::optional<std::pair<llvm::lsp::Range, std::string>>
  prepareRename(const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos);

  /// Perform a rename operation.
  /// Returns a workspace edit with all the changes needed.
  std::optional<llvm::lsp::WorkspaceEdit>
  renameSymbol(const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos,
               llvm::StringRef newName);

  //===--------------------------------------------------------------------===//
  // Document Links
  //===--------------------------------------------------------------------===//

  /// Return document links for include directives.
  void getDocumentLinks(const llvm::lsp::URIForFile &uri,
                        std::vector<llvm::lsp::DocumentLink> &links);

  //===--------------------------------------------------------------------===//
  // Semantic Tokens
  //===--------------------------------------------------------------------===//

  /// Return semantic tokens for the entire document.
  void getSemanticTokens(const llvm::lsp::URIForFile &uri,
                         std::vector<SemanticToken> &tokens);

  //===--------------------------------------------------------------------===//
  // Inlay Hints
  //===--------------------------------------------------------------------===//

  /// Return inlay hints for the given range.
  void getInlayHints(const llvm::lsp::URIForFile &uri,
                     const llvm::lsp::Range &range,
                     std::vector<llvm::lsp::InlayHint> &hints);

  //===--------------------------------------------------------------------===//
  // Signature Help
  //===--------------------------------------------------------------------===//

  /// Return signature help for function/task calls at the given position.
  llvm::lsp::SignatureHelp getSignatureHelp(const llvm::lsp::URIForFile &uri,
                                            const llvm::lsp::Position &pos);

  //===--------------------------------------------------------------------===//
  // Call Hierarchy
  //===--------------------------------------------------------------------===//

  /// Represents a call hierarchy item (function or task).
  struct CallHierarchyItem {
    std::string name;
    llvm::lsp::SymbolKind kind;
    std::string detail;
    llvm::lsp::URIForFile uri;
    llvm::lsp::Range range;
    llvm::lsp::Range selectionRange;
    std::string data; // Encoded symbol info for later lookup
  };

  /// Represents an incoming call (a caller of a function/task).
  struct CallHierarchyIncomingCall {
    CallHierarchyItem from;
    std::vector<llvm::lsp::Range> fromRanges;
  };

  /// Represents an outgoing call (a callee from a function/task).
  struct CallHierarchyOutgoingCall {
    CallHierarchyItem to;
    std::vector<llvm::lsp::Range> fromRanges;
  };

  /// Prepare call hierarchy at the given position.
  /// Returns the call hierarchy item for the function/task at position if any.
  std::optional<CallHierarchyItem>
  prepareCallHierarchy(const llvm::lsp::URIForFile &uri,
                       const llvm::lsp::Position &pos);

  /// Get incoming calls for a call hierarchy item.
  /// Returns all call sites that call the given function/task.
  void getIncomingCalls(const CallHierarchyItem &item,
                        std::vector<CallHierarchyIncomingCall> &calls);

  /// Get outgoing calls from a call hierarchy item.
  /// Returns all functions/tasks called from the given function/task.
  void getOutgoingCalls(const CallHierarchyItem &item,
                        std::vector<CallHierarchyOutgoingCall> &calls);

  //===--------------------------------------------------------------------===//
  // Document Formatting
  //===--------------------------------------------------------------------===//

  /// Formatting options for document formatting.
  struct FormattingOptions {
    /// Number of spaces for indentation (ignored if insertSpaces is false).
    unsigned tabSize = 2;
    /// Use spaces for indentation instead of tabs.
    bool insertSpaces = true;
  };

  /// Format the entire document.
  /// Returns a list of text edits to apply.
  void formatDocument(const llvm::lsp::URIForFile &uri,
                      const FormattingOptions &options,
                      std::vector<llvm::lsp::TextEdit> &edits);

  /// Format a range within the document.
  /// Returns a list of text edits to apply.
  void formatRange(const llvm::lsp::URIForFile &uri,
                   const llvm::lsp::Range &range,
                   const FormattingOptions &options,
                   std::vector<llvm::lsp::TextEdit> &edits);

  std::optional<uint32_t> lspPositionToOffset(const llvm::lsp::Position &pos);
  const char *getPointerFor(const llvm::lsp::Position &pos);

private:
  std::optional<std::pair<slang::BufferID, llvm::SmallString<128>>>
  getOrOpenFile(llvm::StringRef filePath);

  VerilogServerContext &globalContext;

  slang::BufferID mainBufferId;

  // A map from a file name to the corresponding buffer ID in the LLVM
  // source manager.
  llvm::StringMap<std::pair<slang::BufferID, llvm::SmallString<128>>>
      filePathMap;

  // The compilation result.
  llvm::FailureOr<std::unique_ptr<slang::ast::Compilation>> compilation;

  // The slang driver.
  slang::driver::Driver driver;

  /// The index of the parsed module.
  std::unique_ptr<circt::lsp::VerilogIndex> index;

  /// The precomputed line offsets for faster lookups
  std::vector<uint32_t> lineOffsets;
  void computeLineOffsets(std::string_view text);

  /// Scan and index `include directives.
  void scanIncludeDirectives();

  // The URI of the document.
  llvm::lsp::URIForFile uri;
};

} // namespace lsp
} // namespace circt

#endif
