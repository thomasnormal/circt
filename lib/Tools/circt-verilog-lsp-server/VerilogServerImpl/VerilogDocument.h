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
                        const llvm::lsp::Position &pos,
                        std::vector<llvm::lsp::Location> &references);

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

  // The URI of the document.
  llvm::lsp::URIForFile uri;
};

} // namespace lsp
} // namespace circt

#endif
