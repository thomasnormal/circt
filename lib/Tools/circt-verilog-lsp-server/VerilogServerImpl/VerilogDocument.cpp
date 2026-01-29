//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogDocument.cpp
//
// This file implements the VerilogDocument class, which represents a single
// open Verilog/SystemVerilog source file within the CIRCT Verilog LSP server.
// It acts as the per-buffer bridge between the Language Server Protocol (LSP)
// and the Slang front-end infrastructure.
//
// Responsibilities:
//   * Parse and elaborate a single Verilog source buffer using Slang’s driver.
//   * Integrate with project-wide command files (-C) and include/library search
//     paths supplied by the VerilogServerContext.
//   * Handle main-buffer override semantics: when the buffer is already listed
//     in a command file, it reuses the existing Slang buffer; otherwise it
//     injects an in-memory buffer directly.
//   * Collect and forward diagnostics to the LSP client via
//   LSPDiagnosticClient.
//   * Build and own a VerilogIndex for symbol and location queries.
//   * Provide translation utilities between LSP and Slang coordinates, such as
//     UTF-16 ↔ UTF-8 position mapping and conversion to llvm::lsp::Location.
//
// The class is used by VerilogServerContext to maintain open documents,
// service “go to definition” and “find references” requests, and keep file
// state synchronized with the editor.
//
//===----------------------------------------------------------------------===//

#include "slang/ast/ASTVisitor.h"
#include "slang/ast/EvalContext.h"
#include "slang/ast/symbols/BlockSymbols.h"
#include "slang/ast/symbols/ClassSymbols.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/ParameterSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/SubroutineSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/parsing/LexerFacts.h"
#include "slang/syntax/AllSyntax.h"
#include "slang/syntax/SyntaxTree.h"
#include "slang/syntax/SyntaxVisitor.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/Path.h"

#include "../Utils/LSPUtils.h"
#include "LSPDiagnosticClient.h"
#include "VerilogDocument.h"
#include "VerilogServerContext.h"
// TODO: Re-enable when CIRCTLinting library is available
// #include "circt/Analysis/Linting/LintConfig.h"
// #include "circt/Analysis/Linting/LintRules.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

// Define to enable lint integration when CIRCTLinting library is built
// #define CIRCT_VERILOG_LSP_LINTING_ENABLED

using namespace circt::lsp;
using namespace llvm;
using namespace llvm::lsp;

namespace {

/// Convert slang 1-based line number to LSP 0-based line number.
/// Clamps to 0 if slang returns 0 (which can happen for invalid locations).
inline int slangLineToLsp(size_t slangLine) {
  return slangLine > 0 ? static_cast<int>(slangLine) - 1 : 0;
}

/// Convert slang 1-based column number to LSP 0-based column number.
/// Clamps to 0 if slang returns 0 (which can happen for invalid locations).
inline int slangColumnToLsp(size_t slangColumn) {
  return slangColumn > 0 ? static_cast<int>(slangColumn) - 1 : 0;
}

#ifdef CIRCT_VERILOG_LSP_LINTING_ENABLED
/// Convert a lint severity to LSP DiagnosticSeverity.
llvm::lsp::DiagnosticSeverity
lintSeverityToLSP(circt::lint::LintSeverity severity) {
  switch (severity) {
  case circt::lint::LintSeverity::Error:
    return llvm::lsp::DiagnosticSeverity::Error;
  case circt::lint::LintSeverity::Warning:
    return llvm::lsp::DiagnosticSeverity::Warning;
  case circt::lint::LintSeverity::Hint:
    return llvm::lsp::DiagnosticSeverity::Hint;
  case circt::lint::LintSeverity::Ignore:
    return llvm::lsp::DiagnosticSeverity::Information;
  }
  return llvm::lsp::DiagnosticSeverity::Warning;
}

/// Run lint checks on a compilation and add results to diagnostics.
void runLintChecks(const slang::ast::Compilation &compilation,
                   const VerilogDocument &doc, slang::BufferID mainBufferId,
                   std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  // Create default lint configuration
  auto lintConfig = circt::lint::LintConfig::createDefault();

  // Create the lint runner
  circt::lint::LintRunner runner(*lintConfig);

  // Run all enabled lint rules
  auto results = runner.run(compilation);

  // Convert lint diagnostics to LSP diagnostics
  const auto &sm = doc.getSlangSourceManager();
  for (const auto &lintDiag : results.diagnostics) {
    // Skip if the diagnostic is not in the main buffer
    if (!lintDiag.location.valid() ||
        lintDiag.location.buffer() != mainBufferId)
      continue;

    llvm::lsp::Diagnostic lspDiag;
    lspDiag.message = lintDiag.message;
    lspDiag.severity = lintSeverityToLSP(lintDiag.severity);
    lspDiag.source = "circt-lint";

    // Note: LLVM's base LSP Diagnostic doesn't have a 'code' field
    // If lintDiag.code is available, we could append it to the message
    if (lintDiag.code)
      lspDiag.message = "[" + *lintDiag.code + "] " + lspDiag.message;

    // Convert location
    int line = slangLineToLsp(sm.getLineNumber(lintDiag.location));
    int col = slangColumnToLsp(sm.getColumnNumber(lintDiag.location));

    if (lintDiag.range) {
      int endLine = slangLineToLsp(sm.getLineNumber(lintDiag.range->end()));
      int endCol = slangColumnToLsp(sm.getColumnNumber(lintDiag.range->end()));
      lspDiag.range = llvm::lsp::Range(llvm::lsp::Position(line, col),
                                        llvm::lsp::Position(endLine, endCol));
    } else {
      // Single character range
      lspDiag.range = llvm::lsp::Range(llvm::lsp::Position(line, col),
                                        llvm::lsp::Position(line, col + 1));
    }

    // Add related locations
    for (const auto &[relatedLoc, relatedMsg] : lintDiag.relatedLocations) {
      if (!relatedLoc.valid())
        continue;

      int relLine = slangLineToLsp(sm.getLineNumber(relatedLoc));
      int relCol = slangColumnToLsp(sm.getColumnNumber(relatedLoc));

      llvm::lsp::DiagnosticRelatedInformation relatedInfo;
      relatedInfo.message = relatedMsg;
      relatedInfo.location.range =
          llvm::lsp::Range(llvm::lsp::Position(relLine, relCol),
                           llvm::lsp::Position(relLine, relCol + 1));

      // Try to get the URI for the file
      auto fileName = sm.getRawFileName(relatedLoc.buffer());
      if (!fileName.empty()) {
        llvm::SmallString<256> absPath(fileName);
        llvm::sys::fs::make_absolute(absPath);
        if (auto uri = llvm::lsp::URIForFile::fromFile(absPath))
          relatedInfo.location.uri = *uri;
      }

      if (!lspDiag.relatedInformation)
        lspDiag.relatedInformation.emplace();
      lspDiag.relatedInformation->push_back(std::move(relatedInfo));
    }

    diagnostics.push_back(std::move(lspDiag));
  }
}
#endif // CIRCT_VERILOG_LSP_LINTING_ENABLED

} // namespace

static inline void setTopModules(slang::driver::Driver &driver) {
  // Parse the main buffer
  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse main buffer "));
    return;
  }
  // Extract all the top modules in the file directly from the syntax tree
  std::vector<std::string> topModules;
  for (auto &t : driver.syntaxTrees) {
    if (auto *compUnit =
            t->root().as_if<slang::syntax::CompilationUnitSyntax>()) {
      for (auto *member : compUnit->members) {
        // While it's called "ModuleDeclarationSyntax", it also covers
        // packages
        if (auto *moduleDecl =
                member->as_if<slang::syntax::ModuleDeclarationSyntax>()) {
          topModules.emplace_back(moduleDecl->header->name.valueText());
        }
      }
    }
  }
  driver.options.topModules = std::move(topModules);
}

static inline void
copyBuffers(slang::driver::Driver &driver,
            const slang::driver::Driver *const projectDriver,
            const llvm::SmallString<256> &mainBufferFileName) {
  for (auto bId : projectDriver->sourceManager.getAllBuffers()) {
    std::string_view slangRawPath =
        projectDriver->sourceManager.getRawFileName(bId);

    llvm::SmallString<256> slangCanonPath;
    if (llvm::sys::fs::real_path(slangRawPath, slangCanonPath))
      continue;

    if (slangCanonPath ==
        mainBufferFileName) // skip the file you're already compiling
      continue;

    bool alreadyLoaded = false;
    for (auto id : driver.sourceManager.getAllBuffers()) {
      if (driver.sourceManager.getFullPath(id).string() ==
          slangCanonPath.str()) {
        alreadyLoaded = true;
        break;
      }
    }
    if (alreadyLoaded)
      continue;

    auto buffer = driver.sourceManager.assignText(
        slangCanonPath.str(), projectDriver->sourceManager.getSourceText(bId));
    driver.sourceLoader.addBuffer(buffer);
  }
}

VerilogDocument::VerilogDocument(
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef contents, std::vector<llvm::lsp::Diagnostic> &diagnostics,
    const slang::driver::Driver *const projectDriver,
    const std::vector<std::string> &projectIncludeDirectories)
    : globalContext(context), uri(uri) {

  llvm::SmallString<256> canonPath(uri.file());
  if (std::error_code ec = llvm::sys::fs::real_path(uri.file(), canonPath))
    canonPath = uri.file(); // fall back, but try to keep it absolute

  // Build the set of include directories for this file.
  llvm::SmallString<32> uriDirectory(uri.file());
  llvm::sys::path::remove_filename(uriDirectory);

  std::vector<std::string> libDirs;
  libDirs.push_back(uriDirectory.str().str());
  libDirs.insert(libDirs.end(), context.options.libDirs.begin(),
                 context.options.libDirs.end());

  for (const auto &libDir : libDirs)
    driver.sourceLoader.addSearchDirectories(libDir);

  // Add UVM support if configured.
  if (!context.options.uvmPath.empty()) {
    // Add UVM include directories for `include directives.
    (void)driver.sourceManager.addUserDirectories(context.options.uvmPath);

    // Also add the UVM package source file to the compilation.
    // This makes uvm_pkg available for import.
    llvm::SmallString<256> uvmPkgPath(context.options.uvmPath);
    llvm::sys::path::append(uvmPkgPath, "uvm_pkg.sv");
    if (llvm::sys::fs::exists(uvmPkgPath)) {
      driver.sourceLoader.addFiles(uvmPkgPath.str());
    }
  }

  auto memBuffer = llvm::MemoryBuffer::getMemBufferCopy(contents, uri.file());
  if (!memBuffer) {
    circt::lsp::Logger::error(
        Twine("Failed to create memory buffer for file ") + uri.file());
    return;
  }

  auto topSlangBuffer =
      driver.sourceManager.assignText(uri.file(), memBuffer->getBuffer());
  driver.sourceLoader.addBuffer(topSlangBuffer);
  mainBufferId = topSlangBuffer.id;

  auto diagClient = std::make_shared<LSPDiagnosticClient>(*this, diagnostics);
  driver.diagEngine.addClient(diagClient);

  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::LintMode, false);
  driver.options.compilationFlags.emplace(
      slang::ast::CompilationFlags::DisableInstanceCaching, false);

  for (auto &dir : projectIncludeDirectories)
    (void)driver.sourceManager.addUserDirectories(dir);

  if (!driver.processOptions()) {
    circt::lsp::Logger::error(Twine("Failed to process slang driver options!"));
    return;
  }

  driver.diagEngine.setIgnoreAllWarnings(false);

  // Import dependencies from projectDriver if it exists.
  if (projectDriver) {
    // Copy options from project driver
    driver.options = projectDriver->options;
    // Set top modules according to main buffer
    setTopModules(driver);
    // Copy dependency buffers from project driver
    copyBuffers(driver, projectDriver, canonPath);
  }

  if (!driver.parseAllSources()) {
    circt::lsp::Logger::error(Twine("Failed to parse Verilog file ") +
                              uri.file());
    return;
  }

  compilation = driver.createCompilation();
  if (failed(compilation)) {
    circt::lsp::Logger::error(Twine("Failed to compile Verilog file ") +
                              uri.file());
    return;
  }

  for (auto &diag : (*compilation)->getAllDiagnostics())
    driver.diagEngine.issue(diag);

  // Run lint checks and add any additional diagnostics
#ifdef CIRCT_VERILOG_LSP_LINTING_ENABLED
  runLintChecks(**compilation, *this, mainBufferId, diagnostics);
#endif

  computeLineOffsets(driver.sourceManager.getSourceText(mainBufferId));

  index = std::make_unique<VerilogIndex>(mainBufferId, driver.sourceManager);
  // Populate the index.
  index->initialize(**compilation);

  // Scan for include directives
  scanIncludeDirectives();
}

/// Scan the source text for `include directives and register them with the index.
void VerilogDocument::scanIncludeDirectives() {
  if (!index)
    return;

  auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  // Scan for `include directives
  size_t pos = 0;
  while ((pos = text.find("`include", pos)) != std::string_view::npos) {
    size_t directiveStart = pos;
    pos += 8; // Skip "`include"

    // Skip whitespace
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t'))
      ++pos;

    if (pos >= text.size())
      break;

    // Check for quoted filename
    char quote = text[pos];
    if (quote != '"' && quote != '<') {
      continue;
    }
    char endQuote = (quote == '<') ? '>' : '"';
    ++pos;

    size_t fileStart = pos;
    while (pos < text.size() && text[pos] != endQuote && text[pos] != '\n')
      ++pos;

    if (pos >= text.size() || text[pos] != endQuote)
      continue;

    size_t fileEnd = pos;
    ++pos; // Skip closing quote

    // Extract the filename
    std::string filename(text.substr(fileStart, fileEnd - fileStart));

    // Register the include directive
    index->insertInclude(static_cast<uint32_t>(fileStart),
                         static_cast<uint32_t>(fileEnd), filename);
  }
}

llvm::lsp::Location
VerilogDocument::getLspLocation(slang::SourceLocation loc) const {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    const auto &slangSourceManager = getSlangSourceManager();
    auto line = slangLineToLsp(slangSourceManager.getLineNumber(loc));
    auto column = slangColumnToLsp(slangSourceManager.getColumnNumber(loc));
    auto it = loc.buffer();
    if (it == mainBufferId)
      return llvm::lsp::Location(uri, llvm::lsp::Range(Position(line, column)));

    llvm::StringRef fileName = slangSourceManager.getFileName(loc);
    // Ensure absolute path for LSP:
    llvm::SmallString<256> abs(fileName);
    if (!llvm::sys::path::is_absolute(abs)) {
      // Try realPath first
      if (std::error_code ec = llvm::sys::fs::real_path(fileName, abs)) {
        // Fallback: make it absolute relative to the process CWD
        llvm::sys::fs::current_path(abs); // abs = CWD
        llvm::sys::path::append(abs, fileName);
      }
    }

    if (auto uriOrErr = llvm::lsp::URIForFile::fromFile(abs)) {
      if (auto e = uriOrErr.takeError())
        return llvm::lsp::Location();
      return llvm::lsp::Location(*uriOrErr,
                                 llvm::lsp::Range(Position(line, column)));
    }
    return llvm::lsp::Location();
  }
  return llvm::lsp::Location();
}

llvm::lsp::Location
VerilogDocument::getLspLocation(slang::SourceRange range) const {

  auto start = getLspLocation(range.start());
  auto end = getLspLocation(range.end());

  if (start.uri != end.uri)
    return llvm::lsp::Location();

  return llvm::lsp::Location(
      start.uri, llvm::lsp::Range(start.range.start, end.range.end));
}

std::optional<std::pair<slang::BufferID, SmallString<128>>>
VerilogDocument::getOrOpenFile(StringRef filePath) {

  auto fileInfo = filePathMap.find(filePath);
  if (fileInfo != filePathMap.end())
    return fileInfo->second;

  auto getIfExist = [&](StringRef path)
      -> std::optional<std::pair<slang::BufferID, SmallString<128>>> {
    if (llvm::sys::fs::exists(path)) {
      auto memoryBuffer = llvm::MemoryBuffer::getFile(path);
      if (!memoryBuffer) {
        return std::nullopt;
      }

      auto newSlangBuffer = driver.sourceManager.assignText(
          path.str(), memoryBuffer.get()->getBufferStart());
      driver.sourceLoader.addBuffer(newSlangBuffer);

      fileInfo = filePathMap
                     .insert(std::make_pair(
                         filePath, std::make_pair(newSlangBuffer.id, path)))
                     .first;

      return fileInfo->second;
    }
    return std::nullopt;
  };

  if (llvm::sys::path::is_absolute(filePath))
    return getIfExist(filePath);

  // Search locations.
  for (auto &libRoot : globalContext.options.extraSourceLocationDirs) {
    SmallString<128> lib(libRoot);
    llvm::sys::path::append(lib, filePath);
    if (auto fileInfo = getIfExist(lib))
      return fileInfo;
  }

  return std::nullopt;
}

static llvm::lsp::Range getRange(const mlir::FileLineColRange &fileLoc) {
  return llvm::lsp::Range(
      llvm::lsp::Position(fileLoc.getStartLine(), fileLoc.getStartColumn()),
      llvm::lsp::Position(fileLoc.getEndLine(), fileLoc.getEndColumn()));
}

/// Build a vector of line start offsets (0-based).
void VerilogDocument::computeLineOffsets(std::string_view text) {
  lineOffsets.clear();
  lineOffsets.reserve(1024);
  lineOffsets.push_back(0);
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n') {
      lineOffsets.push_back(static_cast<uint32_t>(i + 1));
    }
  }
}

// LSP (0-based line, UTF-16 character) -> byte offset into UTF-8 buffer.
std::optional<uint32_t>
VerilogDocument::lspPositionToOffset(const llvm::lsp::Position &pos) {

  auto &sm = getSlangSourceManager();

  std::string_view text = sm.getSourceText(mainBufferId);

  // Clamp line index
  if ((unsigned)pos.line >= lineOffsets.size())
    return std::nullopt;

  size_t lineStart = lineOffsets[pos.line];
  size_t lineEnd = ((unsigned)(pos.line + 1) < lineOffsets.size())
                       ? lineOffsets[pos.line + 1] - 1
                       : text.size();

  const llvm::UTF8 *src =
      reinterpret_cast<const llvm::UTF8 *>(text.data() + lineStart);
  const llvm::UTF8 *srcEnd =
      reinterpret_cast<const llvm::UTF8 *>(text.data() + lineEnd);

  // Convert up to 'target' UTF-16 code units; stop early if line ends.
  const uint32_t target = pos.character;
  if (target == 0)
    return static_cast<uint32_t>(
        src - reinterpret_cast<const llvm::UTF8 *>(text.data()));

  std::vector<llvm::UTF16> sink(target);
  llvm::UTF16 *out = sink.data();
  llvm::UTF16 *outEnd = out + sink.size();

  (void)llvm::ConvertUTF8toUTF16(&src, srcEnd, &out, outEnd,
                                 llvm::lenientConversion);

  return static_cast<uint32_t>(reinterpret_cast<const char *>(src) -
                               text.data());
}

const char *VerilogDocument::getPointerFor(const llvm::lsp::Position &pos) {
  auto &sm = getSlangSourceManager();
  auto slangBufferOffset = lspPositionToOffset(pos);

  if (!slangBufferOffset.has_value())
    return nullptr;

  uint32_t offset = slangBufferOffset.value();
  return sm.getSourceText(mainBufferId).data() + offset;
}

void VerilogDocument::getLocationsOf(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &defPos,
    std::vector<llvm::lsp::Location> &locations) {

  const auto &slangBufferPointer = getPointerFor(defPos);

  if (!index)
    return;

  // First, check if the cursor is on an include directive
  auto offsetOpt = lspPositionToOffset(defPos);
  if (offsetOpt) {
    uint32_t offset = *offsetOpt;
    const auto &includeMap = index->getIncludes();
    for (const auto &[range, path] : includeMap) {
      if (offset >= range.first && offset < range.second) {
        // The cursor is on an include directive
        llvm::SmallString<256> absPath(path);
        if (!llvm::sys::path::is_absolute(absPath)) {
          // Try to resolve relative to the document's directory
          llvm::SmallString<256> docDir(uri.file());
          llvm::sys::path::remove_filename(docDir);
          llvm::sys::path::append(docDir, path);
          if (llvm::sys::fs::exists(docDir))
            absPath = docDir;
          else {
            // Try libDirs
            for (const auto &libDir : globalContext.options.libDirs) {
              llvm::SmallString<256> libPath(libDir);
              llvm::sys::path::append(libPath, path);
              if (llvm::sys::fs::exists(libPath)) {
                absPath = libPath;
                break;
              }
            }
          }
        }

        if (llvm::sys::fs::exists(absPath)) {
          auto uriOrErr = llvm::lsp::URIForFile::fromFile(absPath);
          if (uriOrErr) {
            // Point to the beginning of the included file
            locations.emplace_back(*uriOrErr,
                                   llvm::lsp::Range(llvm::lsp::Position(0, 0)));
            return;
          }
        }
      }
    }
  }

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  // Found no element in the given index.
  if (!it.valid() || slangBufferPointer < it.start())
    return;

  auto element = it.value();
  if (auto attr = dyn_cast<Attribute>(element)) {

    // Check if the attribute is a FileLineColRange.
    if (auto fileLoc = dyn_cast<mlir::FileLineColRange>(attr)) {

      // Return URI for the file.
      auto fileInfo = getOrOpenFile(fileLoc.getFilename().getValue());
      if (!fileInfo)
        return;
      const auto &[bufferId, filePath] = *fileInfo;
      auto uri = llvm::lsp::URIForFile::fromFile(filePath);
      if (auto e = uri.takeError()) {
        circt::lsp::Logger::error("failed to open file " + filePath);
        return;
      }
      locations.emplace_back(uri.get(), getRange(fileLoc));
    }

    return;
  }

  // If the element is verilog symbol, return the definition of the symbol.
  const auto *symbol = cast<const slang::ast::Symbol *>(element);

  slang::SourceRange range(symbol->location,
                           symbol->location +
                               (symbol->name.size() ? symbol->name.size() : 1));
  locations.push_back(getLspLocation(range));
}

void VerilogDocument::findReferencesOf(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos,
    bool includeDeclaration, std::vector<llvm::lsp::Location> &references) {

  if (!index)
    return;

  const auto &slangBufferPointer = getPointerFor(pos);
  const auto &intervalMap = index->getIntervalMap();
  auto intervalIt = intervalMap.find(slangBufferPointer);

  if (!intervalIt.valid() || slangBufferPointer < intervalIt.start())
    return;

  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(intervalIt.value());
  if (!symbol)
    return;

  // Include the declaration location if requested
  if (includeDeclaration && symbol->location.valid()) {
    slang::SourceRange declRange(
        symbol->location,
        symbol->location + (symbol->name.size() ? symbol->name.size() : 1));
    references.push_back(getLspLocation(declRange));
  }

  // Add all references to the symbol
  auto it = index->getReferences().find(symbol);
  if (it == index->getReferences().end())
    return;
  for (auto referenceRange : it->second)
    references.push_back(getLspLocation(referenceRange));
}

//===----------------------------------------------------------------------===//
// Document Highlight
//===----------------------------------------------------------------------===//

void VerilogDocument::getDocumentHighlights(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Position &pos,
    std::vector<DocumentHighlight> &highlights) {

  if (!index)
    return;

  const auto &slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return;

  const auto &intervalMap = index->getIntervalMap();
  auto intervalIt = intervalMap.find(slangBufferPointer);

  if (!intervalIt.valid() || slangBufferPointer < intervalIt.start())
    return;

  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(intervalIt.value());
  if (!symbol)
    return;

  const auto &sm = getSlangSourceManager();

  // Helper lambda to convert slang SourceRange to LSP Range
  auto toHighlight = [&](slang::SourceRange range,
                         DocumentHighlightKind kind) -> std::optional<DocumentHighlight> {
    // Only include highlights from the same buffer (document)
    if (!range.start().valid() || range.start().buffer() != mainBufferId)
      return std::nullopt;

    int startLine = slangLineToLsp(sm.getLineNumber(range.start()));
    int startCol = slangColumnToLsp(sm.getColumnNumber(range.start()));
    int endLine = slangLineToLsp(sm.getLineNumber(range.end()));
    int endCol = slangColumnToLsp(sm.getColumnNumber(range.end()));

    DocumentHighlight highlight;
    highlight.range = llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                                       llvm::lsp::Position(endLine, endCol));
    highlight.kind = kind;
    return highlight;
  };

  // Add the declaration location (as "Write" kind for definitions)
  if (symbol->location.valid() && symbol->location.buffer() == mainBufferId) {
    slang::SourceRange declRange(
        symbol->location,
        symbol->location + (symbol->name.size() ? symbol->name.size() : 1));
    if (auto hl = toHighlight(declRange, DocumentHighlightKind::Write))
      highlights.push_back(*hl);
  }

  // Add all references to the symbol (as "Read" kind for uses)
  auto refIt = index->getReferences().find(symbol);
  if (refIt != index->getReferences().end()) {
    for (const auto &referenceRange : refIt->second) {
      if (auto hl = toHighlight(referenceRange, DocumentHighlightKind::Read))
        highlights.push_back(*hl);
    }
  }
}

//===----------------------------------------------------------------------===//
// Hover Information
//===----------------------------------------------------------------------===//

/// Format a type description for hover information.
static std::string formatTypeDescription(const slang::ast::Type &type) {
  std::string result;
  llvm::raw_string_ostream os(result);

  if (type.isIntegral()) {
    if (type.isSigned())
      os << "signed ";
    auto width = type.getBitWidth();
    if (width == 1)
      os << "logic";
    else
      os << "logic [" << (width - 1) << ":0]";
  } else if (type.isFloating()) {
    if (type.getBitWidth() == 32)
      os << "shortreal";
    else
      os << "real";
  } else if (type.isString()) {
    os << "string";
  } else if (type.isVoid()) {
    os << "void";
  } else {
    os << type.toString();
  }

  return result;
}

/// Format symbol information for hover display.
static std::string formatSymbolInfo(const slang::ast::Symbol &symbol) {
  std::string result;
  llvm::raw_string_ostream os(result);

  switch (symbol.kind) {
  case slang::ast::SymbolKind::Variable:
  case slang::ast::SymbolKind::Net:
  case slang::ast::SymbolKind::FormalArgument: {
    const auto &valueSymbol = symbol.as<slang::ast::ValueSymbol>();
    os << "```systemverilog\n";
    os << formatTypeDescription(valueSymbol.getType()) << " " << symbol.name;
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::Port: {
    const auto &port = symbol.as<slang::ast::PortSymbol>();
    os << "```systemverilog\n";
    // Show direction
    switch (port.direction) {
    case slang::ast::ArgumentDirection::In:
      os << "input ";
      break;
    case slang::ast::ArgumentDirection::Out:
      os << "output ";
      break;
    case slang::ast::ArgumentDirection::InOut:
      os << "inout ";
      break;
    case slang::ast::ArgumentDirection::Ref:
      os << "ref ";
      break;
    }
    os << formatTypeDescription(port.getType()) << " " << symbol.name;
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::Parameter:
  case slang::ast::SymbolKind::EnumValue: {
    const auto &valueSymbol = symbol.as<slang::ast::ValueSymbol>();
    os << "```systemverilog\n";
    os << "parameter " << formatTypeDescription(valueSymbol.getType()) << " "
       << symbol.name;
    // Try to get the value
    auto initExpr = valueSymbol.getInitializer();
    if (initExpr) {
      slang::ast::EvalContext evalCtx(symbol);
      auto cv = initExpr->eval(evalCtx);
      if (cv)
        os << " = " << cv.toString();
    }
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::Instance: {
    const auto &inst = symbol.as<slang::ast::InstanceSymbol>();
    os << "```systemverilog\n";
    os << inst.getDefinition().name << " " << symbol.name;
    os << "\n```\n\n";
    // Show ports summary
    os << "**Ports:**\n";
    for (const auto *portSym : inst.body.getPortList()) {
      if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
        os << "- ";
        switch (port->direction) {
        case slang::ast::ArgumentDirection::In:
          os << "`input` ";
          break;
        case slang::ast::ArgumentDirection::Out:
          os << "`output` ";
          break;
        case slang::ast::ArgumentDirection::InOut:
          os << "`inout` ";
          break;
        case slang::ast::ArgumentDirection::Ref:
          os << "`ref` ";
          break;
        }
        os << "`" << port->name << "`";
        auto width = port->getType().getBitWidth();
        if (width > 1)
          os << " [" << (width - 1) << ":0]";
        os << "\n";
      }
    }
    break;
  }
  case slang::ast::SymbolKind::Definition: {
    const auto &def = symbol.as<slang::ast::DefinitionSymbol>();
    os << "```systemverilog\n";
    os << "module " << symbol.name;
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::Subroutine: {
    const auto &sub = symbol.as<slang::ast::SubroutineSymbol>();
    os << "```systemverilog\n";
    if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
      os << "function ";
    else
      os << "task ";
    os << formatTypeDescription(sub.getReturnType()) << " " << symbol.name;
    os << "(";
    bool first = true;
    for (const auto *arg : sub.getArguments()) {
      if (!first)
        os << ", ";
      first = false;
      switch (arg->direction) {
      case slang::ast::ArgumentDirection::In:
        os << "input ";
        break;
      case slang::ast::ArgumentDirection::Out:
        os << "output ";
        break;
      case slang::ast::ArgumentDirection::InOut:
        os << "inout ";
        break;
      case slang::ast::ArgumentDirection::Ref:
        os << "ref ";
        break;
      }
      os << formatTypeDescription(arg->getType()) << " " << arg->name;
    }
    os << ")";
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::Package: {
    os << "```systemverilog\n";
    os << "package " << symbol.name;
    os << "\n```";
    break;
  }
  case slang::ast::SymbolKind::ClassType: {
    const auto &classType = symbol.as<slang::ast::ClassType>();
    os << "```systemverilog\n";
    os << "class " << symbol.name;
    // Show base class if exists
    if (classType.getBaseClass()) {
      os << " extends " << classType.getBaseClass()->name;
    }
    os << "\n```\n\n";
    // Show members summary
    os << "**Members:**\n";
    for (const auto &member : classType.members()) {
      if (member.name.empty())
        continue;
      os << "- ";
      switch (member.kind) {
      case slang::ast::SymbolKind::ClassProperty:
        os << "`property` ";
        break;
      case slang::ast::SymbolKind::Subroutine:
        os << "`method` ";
        break;
      default:
        os << "`" << slang::ast::toString(member.kind) << "` ";
        break;
      }
      os << "`" << member.name << "`\n";
    }
    break;
  }
  case slang::ast::SymbolKind::ClassProperty: {
    const auto &prop = symbol.as<slang::ast::ClassPropertySymbol>();
    os << "```systemverilog\n";
    // Show visibility
    switch (prop.visibility) {
    case slang::ast::Visibility::Public:
      break; // public is implicit
    case slang::ast::Visibility::Protected:
      os << "protected ";
      break;
    case slang::ast::Visibility::Local:
      os << "local ";
      break;
    }
    os << formatTypeDescription(prop.getType()) << " " << symbol.name;
    os << "\n```";
    break;
  }
  default:
    os << "```systemverilog\n";
    os << symbol.name;
    os << "\n```";
    break;
  }

  return result;
}

std::optional<llvm::lsp::Hover>
VerilogDocument::getHover(const llvm::lsp::URIForFile &uri,
                          const llvm::lsp::Position &pos) {
  if (!index)
    return std::nullopt;

  const auto *slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return std::nullopt;

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  // Found no element at the given position.
  if (!it.valid() || slangBufferPointer < it.start())
    return std::nullopt;

  auto element = it.value();

  // If it's an attribute (e.g., source location comment), return early
  if (auto attr = dyn_cast<Attribute>(element))
    return std::nullopt;

  // Get the symbol and format hover information
  const auto *symbol = cast<const slang::ast::Symbol *>(element);
  if (!symbol)
    return std::nullopt;

  // Calculate the range of the symbol for highlighting
  const auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  // Calculate character positions from pointer offsets
  size_t startOffset = it.start() - text.data();
  size_t endOffset = it.stop() - text.data();

  // Convert offsets to LSP positions
  int startLine = 0, startChar = 0;
  int endLine = 0, endChar = 0;

  for (size_t i = 0; i < lineOffsets.size(); ++i) {
    if (lineOffsets[i] <= startOffset) {
      startLine = i;
      startChar = startOffset - lineOffsets[i];
    }
    if (lineOffsets[i] <= endOffset) {
      endLine = i;
      endChar = endOffset - lineOffsets[i];
    }
  }

  llvm::lsp::Range range(llvm::lsp::Position(startLine, startChar),
                         llvm::lsp::Position(endLine, endChar));
  llvm::lsp::Hover hover(range);
  hover.contents.kind = llvm::lsp::MarkupKind::Markdown;
  hover.contents.value = formatSymbolInfo(*symbol);

  return hover;
}

//===----------------------------------------------------------------------===//
// Document Symbols
//===----------------------------------------------------------------------===//

/// Map slang symbol kind to LSP SymbolKind.
static llvm::lsp::SymbolKind
mapSymbolKind(slang::ast::SymbolKind slangKind) {
  using SK = slang::ast::SymbolKind;
  using LK = llvm::lsp::SymbolKind;

  switch (slangKind) {
  case SK::Definition:
  case SK::Instance:
    return LK::Module;
  case SK::Package:
    return LK::Package;
  case SK::Net:
  case SK::Variable:
    return LK::Variable;
  case SK::Parameter:
  case SK::EnumValue:
    return LK::Constant;
  case SK::Port:
    return LK::Property;
  case SK::Subroutine:
    return LK::Function;
  case SK::ClassType:
  case SK::ClassProperty:
    return LK::Class;
  case SK::InterfacePort:
  case SK::Modport:
    return LK::Interface;
  case SK::EnumType:
  case SK::TypeAlias:
    return LK::Enum;
  case SK::PackedStructType:
  case SK::UnpackedStructType:
    return LK::Struct;
  default:
    return LK::Variable;
  }
}

/// Calculate LSP Range from slang source range.
llvm::lsp::Range VerilogDocument::getLspRange(slang::SourceRange range) const {
  const auto &sm = getSlangSourceManager();
  int startLine = slangLineToLsp(sm.getLineNumber(range.start()));
  int startCol = slangColumnToLsp(sm.getColumnNumber(range.start()));
  int endLine = slangLineToLsp(sm.getLineNumber(range.end()));
  int endCol = slangColumnToLsp(sm.getColumnNumber(range.end()));
  return llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                          llvm::lsp::Position(endLine, endCol));
}

/// Visitor to collect document symbols from the AST.
namespace {
class DocumentSymbolVisitor
    : public slang::ast::ASTVisitor<DocumentSymbolVisitor, true, true> {
public:
  DocumentSymbolVisitor(const VerilogDocument &doc, slang::BufferID bufferId,
                        const slang::SourceManager &sm)
      : doc(doc), bufferId(bufferId), sm(sm) {}

  std::vector<llvm::lsp::DocumentSymbol> symbols;

  void visit(const slang::ast::InstanceBodySymbol &body) {
    // Create a module symbol
    if (body.location.buffer() != bufferId)
      return;

    auto *syntax = body.getSyntax();
    if (!syntax)
      return;

    llvm::lsp::Range fullRange = doc.getLspRange(syntax->sourceRange());
    int nameLine = slangLineToLsp(sm.getLineNumber(body.location));
    int nameCol = slangColumnToLsp(sm.getColumnNumber(body.location));
    llvm::lsp::Range nameRange(llvm::lsp::Position(nameLine, nameCol),
                               llvm::lsp::Position(nameLine, nameCol + body.name.size()));

    llvm::lsp::DocumentSymbol moduleSym(body.name, llvm::lsp::SymbolKind::Module,
                                         fullRange, nameRange);
    moduleSym.detail = "module";

    // Collect children (ports, signals, etc.)
    std::vector<llvm::lsp::DocumentSymbol> children;

    // Add ports
    for (const auto *portSym : body.getPortList()) {
      if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
        if (!port->location.valid() || port->location.buffer() != bufferId)
          continue;

        int pLine = slangLineToLsp(sm.getLineNumber(port->location));
        int pCol = slangColumnToLsp(sm.getColumnNumber(port->location));
        llvm::lsp::Range pRange(llvm::lsp::Position(pLine, pCol),
                                llvm::lsp::Position(pLine, pCol + port->name.size()));

        llvm::lsp::DocumentSymbol portSymbol(port->name, llvm::lsp::SymbolKind::Property,
                                              pRange, pRange);
        std::string detail;
        switch (port->direction) {
        case slang::ast::ArgumentDirection::In:
          detail = "input";
          break;
        case slang::ast::ArgumentDirection::Out:
          detail = "output";
          break;
        case slang::ast::ArgumentDirection::InOut:
          detail = "inout";
          break;
        case slang::ast::ArgumentDirection::Ref:
          detail = "ref";
          break;
        }
        auto width = port->getType().getBitWidth();
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        portSymbol.detail = detail;
        children.push_back(std::move(portSymbol));
      }
    }

    // Visit members for variables, nets, parameters, etc.
    for (const auto &member : body.members()) {
      if (member.location.buffer() != bufferId)
        continue;

      llvm::lsp::SymbolKind kind = mapSymbolKind(member.kind);
      std::string detail;

      switch (member.kind) {
      case slang::ast::SymbolKind::Net: {
        const auto &net = member.as<slang::ast::NetSymbol>();
        auto width = net.getType().getBitWidth();
        detail = "wire";
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        break;
      }
      case slang::ast::SymbolKind::Variable: {
        const auto &var = member.as<slang::ast::VariableSymbol>();
        auto width = var.getType().getBitWidth();
        detail = "logic";
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        break;
      }
      case slang::ast::SymbolKind::Parameter: {
        const auto &param = member.as<slang::ast::ParameterSymbol>();
        detail = "parameter";
        auto initExpr = param.getInitializer();
        if (initExpr) {
          slang::ast::EvalContext evalCtx(member);
          auto cv = initExpr->eval(evalCtx);
          if (cv)
            detail += " = " + std::string(cv.toString());
        }
        break;
      }
      case slang::ast::SymbolKind::Instance: {
        const auto &inst = member.as<slang::ast::InstanceSymbol>();
        detail = std::string(inst.getDefinition().name);
        break;
      }
      case slang::ast::SymbolKind::Subroutine: {
        const auto &sub = member.as<slang::ast::SubroutineSymbol>();
        if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
          detail = "function";
        else
          detail = "task";
        break;
      }
      case slang::ast::SymbolKind::ClassType: {
        // Handle class definitions inside modules
        const auto &classType = member.as<slang::ast::ClassType>();
        kind = llvm::lsp::SymbolKind::Class;
        detail = "class";

        // Create a class symbol with children for methods/properties
        auto *classSyntax = classType.getSyntax();
        llvm::lsp::Range classFullRange = classSyntax
            ? doc.getLspRange(classSyntax->sourceRange())
            : llvm::lsp::Range();
        int cLine = slangLineToLsp(sm.getLineNumber(classType.location));
        int cCol = slangColumnToLsp(sm.getColumnNumber(classType.location));
        llvm::lsp::Range classNameRange(llvm::lsp::Position(cLine, cCol),
                                        llvm::lsp::Position(cLine, cCol + classType.name.size()));

        llvm::lsp::DocumentSymbol classSym(classType.name, kind,
                                           classSyntax ? classFullRange : classNameRange,
                                           classNameRange);
        classSym.detail = detail;

        // Add class members (methods, properties)
        std::vector<llvm::lsp::DocumentSymbol> classChildren;
        for (const auto &classMember : classType.members()) {
          if (classMember.location.buffer() != bufferId || classMember.name.empty())
            continue;

          llvm::lsp::SymbolKind memberKind = mapSymbolKind(classMember.kind);
          std::string memberDetail;

          switch (classMember.kind) {
          case slang::ast::SymbolKind::ClassProperty: {
            const auto &prop = classMember.as<slang::ast::ClassPropertySymbol>();
            memberDetail = formatTypeDescription(prop.getType());
            memberKind = llvm::lsp::SymbolKind::Field;
            break;
          }
          case slang::ast::SymbolKind::Subroutine: {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
              memberDetail = "function";
            else
              memberDetail = "task";
            memberKind = llvm::lsp::SymbolKind::Method;
            break;
          }
          default:
            continue;
          }

          int cmLine = slangLineToLsp(sm.getLineNumber(classMember.location));
          int cmCol = slangColumnToLsp(sm.getColumnNumber(classMember.location));
          llvm::lsp::Range cmRange(llvm::lsp::Position(cmLine, cmCol),
                                   llvm::lsp::Position(cmLine, cmCol + classMember.name.size()));

          llvm::lsp::DocumentSymbol classMemberSym(classMember.name, memberKind, cmRange, cmRange);
          classMemberSym.detail = memberDetail;
          classChildren.push_back(std::move(classMemberSym));
        }

        classSym.children = std::move(classChildren);
        children.push_back(std::move(classSym));
        continue; // Already added, skip the common path
      }
      case slang::ast::SymbolKind::ProceduralBlock: {
        // Handle always blocks, initial blocks, final blocks
        const auto &procBlock = member.as<slang::ast::ProceduralBlockSymbol>();
        kind = llvm::lsp::SymbolKind::Event;

        // Generate a name based on the block type
        std::string blockName;
        switch (procBlock.procedureKind) {
        case slang::ast::ProceduralBlockKind::Initial:
          blockName = "initial";
          detail = "initial block";
          break;
        case slang::ast::ProceduralBlockKind::Final:
          blockName = "final";
          detail = "final block";
          break;
        case slang::ast::ProceduralBlockKind::Always:
          blockName = "always";
          detail = "always block";
          break;
        case slang::ast::ProceduralBlockKind::AlwaysComb:
          blockName = "always_comb";
          detail = "combinational block";
          break;
        case slang::ast::ProceduralBlockKind::AlwaysLatch:
          blockName = "always_latch";
          detail = "latch block";
          break;
        case slang::ast::ProceduralBlockKind::AlwaysFF:
          blockName = "always_ff";
          detail = "flip-flop block";
          break;
        }

        int pLine = slangLineToLsp(sm.getLineNumber(procBlock.location));
        int pCol = slangColumnToLsp(sm.getColumnNumber(procBlock.location));
        llvm::lsp::Range pRange(llvm::lsp::Position(pLine, pCol),
                                llvm::lsp::Position(pLine, pCol + blockName.size()));

        llvm::lsp::DocumentSymbol procSym(blockName, kind, pRange, pRange);
        procSym.detail = detail;
        children.push_back(std::move(procSym));
        continue; // Already added, skip the common path
      }
      default:
        continue; // Skip other member types
      }

      if (member.name.empty())
        continue;

      int mLine = slangLineToLsp(sm.getLineNumber(member.location));
      int mCol = slangColumnToLsp(sm.getColumnNumber(member.location));
      llvm::lsp::Range mRange(llvm::lsp::Position(mLine, mCol),
                              llvm::lsp::Position(mLine, mCol + member.name.size()));

      llvm::lsp::DocumentSymbol memberSym(member.name, kind, mRange, mRange);
      memberSym.detail = detail;
      children.push_back(std::move(memberSym));
    }

    moduleSym.children = std::move(children);
    symbols.push_back(std::move(moduleSym));
  }

  void visit(const slang::ast::PackageSymbol &pkg) {
    if (pkg.location.buffer() != bufferId)
      return;

    auto *syntax = pkg.getSyntax();
    if (!syntax)
      return;

    llvm::lsp::Range fullRange = doc.getLspRange(syntax->sourceRange());
    int nameLine = slangLineToLsp(sm.getLineNumber(pkg.location));
    int nameCol = slangColumnToLsp(sm.getColumnNumber(pkg.location));
    llvm::lsp::Range nameRange(llvm::lsp::Position(nameLine, nameCol),
                               llvm::lsp::Position(nameLine, nameCol + pkg.name.size()));

    llvm::lsp::DocumentSymbol pkgSym(pkg.name, llvm::lsp::SymbolKind::Package,
                                      fullRange, nameRange);
    pkgSym.detail = "package";

    // Collect package members
    std::vector<llvm::lsp::DocumentSymbol> children;
    for (const auto &member : pkg.members()) {
      if (member.location.buffer() != bufferId || member.name.empty())
        continue;

      llvm::lsp::SymbolKind kind = mapSymbolKind(member.kind);

      // Handle classes specially to include their methods/properties
      if (member.kind == slang::ast::SymbolKind::ClassType) {
        const auto &classType = member.as<slang::ast::ClassType>();
        kind = llvm::lsp::SymbolKind::Class;

        auto *classSyntax = classType.getSyntax();
        llvm::lsp::Range classFullRange = classSyntax
            ? doc.getLspRange(classSyntax->sourceRange())
            : llvm::lsp::Range();
        int cLine = slangLineToLsp(sm.getLineNumber(classType.location));
        int cCol = slangColumnToLsp(sm.getColumnNumber(classType.location));
        llvm::lsp::Range classNameRange(llvm::lsp::Position(cLine, cCol),
                                        llvm::lsp::Position(cLine, cCol + classType.name.size()));

        llvm::lsp::DocumentSymbol classSym(classType.name, kind,
                                           classSyntax ? classFullRange : classNameRange,
                                           classNameRange);
        classSym.detail = "class";

        // Add class members (methods, properties)
        std::vector<llvm::lsp::DocumentSymbol> classChildren;
        for (const auto &classMember : classType.members()) {
          if (classMember.location.buffer() != bufferId || classMember.name.empty())
            continue;

          llvm::lsp::SymbolKind memberKind = mapSymbolKind(classMember.kind);
          std::string memberDetail;

          switch (classMember.kind) {
          case slang::ast::SymbolKind::ClassProperty: {
            const auto &prop = classMember.as<slang::ast::ClassPropertySymbol>();
            memberDetail = formatTypeDescription(prop.getType());
            memberKind = llvm::lsp::SymbolKind::Field;
            break;
          }
          case slang::ast::SymbolKind::Subroutine: {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
              memberDetail = "function";
            else
              memberDetail = "task";
            memberKind = llvm::lsp::SymbolKind::Method;
            break;
          }
          default:
            continue;
          }

          int cmLine = slangLineToLsp(sm.getLineNumber(classMember.location));
          int cmCol = slangColumnToLsp(sm.getColumnNumber(classMember.location));
          llvm::lsp::Range cmRange(llvm::lsp::Position(cmLine, cmCol),
                                   llvm::lsp::Position(cmLine, cmCol + classMember.name.size()));

          llvm::lsp::DocumentSymbol classMemberSym(classMember.name, memberKind, cmRange, cmRange);
          classMemberSym.detail = memberDetail;
          classChildren.push_back(std::move(classMemberSym));
        }

        classSym.children = std::move(classChildren);
        children.push_back(std::move(classSym));
        continue;
      }

      int mLine = slangLineToLsp(sm.getLineNumber(member.location));
      int mCol = slangColumnToLsp(sm.getColumnNumber(member.location));
      llvm::lsp::Range mRange(llvm::lsp::Position(mLine, mCol),
                              llvm::lsp::Position(mLine, mCol + member.name.size()));

      llvm::lsp::DocumentSymbol memberSym(member.name, kind, mRange, mRange);
      children.push_back(std::move(memberSym));
    }

    pkgSym.children = std::move(children);
    symbols.push_back(std::move(pkgSym));
  }

  /// Visit interface definitions to extract document symbols.
  /// Unlike modules which are instantiated, interfaces may not be instantiated
  /// but still need to appear in the document symbol list.
  void visitInterfaceDefinition(const slang::ast::DefinitionSymbol &def,
                                slang::ast::Compilation &compilation) {
    if (def.definitionKind != slang::ast::DefinitionKind::Interface)
      return;

    if (def.location.buffer() != bufferId)
      return;

    auto *syntax = def.getSyntax();
    if (!syntax)
      return;

    llvm::lsp::Range fullRange = doc.getLspRange(syntax->sourceRange());
    int nameLine = slangLineToLsp(sm.getLineNumber(def.location));
    int nameCol = slangColumnToLsp(sm.getColumnNumber(def.location));
    llvm::lsp::Range nameRange(llvm::lsp::Position(nameLine, nameCol),
                               llvm::lsp::Position(nameLine, nameCol + def.name.size()));

    // Use SymbolKind::Interface (11 in LSP spec)
    llvm::lsp::DocumentSymbol ifSym(def.name, llvm::lsp::SymbolKind::Interface,
                                    fullRange, nameRange);
    ifSym.detail = "interface";

    // Create a temporary instance body to get the interface members.
    // This allows us to extract ports, signals, modports, etc.
    auto &body = slang::ast::InstanceBodySymbol::fromDefinition(
        compilation, def, def.location, slang::ast::InstanceFlags::None,
        nullptr, nullptr, nullptr);

    std::vector<llvm::lsp::DocumentSymbol> children;

    // Add ports
    for (const auto *portSym : body.getPortList()) {
      if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
        if (!port->location.valid() || port->location.buffer() != bufferId)
          continue;

        int pLine = slangLineToLsp(sm.getLineNumber(port->location));
        int pCol = slangColumnToLsp(sm.getColumnNumber(port->location));
        llvm::lsp::Range pRange(llvm::lsp::Position(pLine, pCol),
                                llvm::lsp::Position(pLine, pCol + port->name.size()));

        llvm::lsp::DocumentSymbol portSymbol(port->name, llvm::lsp::SymbolKind::Property,
                                             pRange, pRange);
        std::string detail;
        switch (port->direction) {
        case slang::ast::ArgumentDirection::In:
          detail = "input";
          break;
        case slang::ast::ArgumentDirection::Out:
          detail = "output";
          break;
        case slang::ast::ArgumentDirection::InOut:
          detail = "inout";
          break;
        case slang::ast::ArgumentDirection::Ref:
          detail = "ref";
          break;
        }
        auto width = port->getType().getBitWidth();
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        portSymbol.detail = detail;
        children.push_back(std::move(portSymbol));
      }
    }

    // Visit members for variables, nets, modports, etc.
    for (const auto &member : body.members()) {
      if (member.location.buffer() != bufferId)
        continue;

      llvm::lsp::SymbolKind kind = mapSymbolKind(member.kind);
      std::string detail;

      switch (member.kind) {
      case slang::ast::SymbolKind::Net: {
        const auto &net = member.as<slang::ast::NetSymbol>();
        auto width = net.getType().getBitWidth();
        detail = "wire";
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        break;
      }
      case slang::ast::SymbolKind::Variable: {
        const auto &var = member.as<slang::ast::VariableSymbol>();
        auto width = var.getType().getBitWidth();
        detail = "logic";
        if (width > 1)
          detail += " [" + std::to_string(width - 1) + ":0]";
        break;
      }
      case slang::ast::SymbolKind::Modport: {
        detail = "modport";
        break;
      }
      case slang::ast::SymbolKind::Parameter: {
        const auto &param = member.as<slang::ast::ParameterSymbol>();
        detail = "parameter";
        auto initExpr = param.getInitializer();
        if (initExpr) {
          slang::ast::EvalContext evalCtx(member);
          auto cv = initExpr->eval(evalCtx);
          if (cv)
            detail += " = " + std::string(cv.toString());
        }
        break;
      }
      case slang::ast::SymbolKind::Subroutine: {
        const auto &sub = member.as<slang::ast::SubroutineSymbol>();
        if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
          detail = "function";
        else
          detail = "task";
        break;
      }
      default:
        continue; // Skip other member types
      }

      if (member.name.empty())
        continue;

      int mLine = slangLineToLsp(sm.getLineNumber(member.location));
      int mCol = slangColumnToLsp(sm.getColumnNumber(member.location));
      llvm::lsp::Range mRange(llvm::lsp::Position(mLine, mCol),
                              llvm::lsp::Position(mLine, mCol + member.name.size()));

      llvm::lsp::DocumentSymbol memberSym(member.name, kind, mRange, mRange);
      memberSym.detail = detail;
      children.push_back(std::move(memberSym));
    }

    ifSym.children = std::move(children);
    symbols.push_back(std::move(ifSym));
  }

  template <typename T>
  void visit(const T &) {}

private:
  const VerilogDocument &doc;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
};
} // namespace

void VerilogDocument::getDocumentSymbols(
    const llvm::lsp::URIForFile &uri,
    std::vector<llvm::lsp::DocumentSymbol> &symbols) {
  if (failed(compilation))
    return;

  const auto &root = (*compilation)->getRoot();
  DocumentSymbolVisitor visitor(*this, mainBufferId, getSlangSourceManager());

  // Visit packages
  for (auto *package : (*compilation)->getPackages()) {
    if (package->location.buffer() != mainBufferId)
      continue;
    visitor.visit(*package);
  }

  // Visit top instances (modules)
  for (auto *inst : root.topInstances) {
    if (inst->body.location.buffer() != mainBufferId)
      continue;
    visitor.visit(inst->body);
  }

  // Visit interface definitions
  // Interfaces may not be instantiated as top instances but should still
  // appear in the document symbol list.
  for (const auto *sym : (*compilation)->getDefinitions()) {
    if (sym->kind == slang::ast::SymbolKind::Definition) {
      const auto &def = sym->as<slang::ast::DefinitionSymbol>();
      if (def.definitionKind == slang::ast::DefinitionKind::Interface &&
          def.location.buffer() == mainBufferId) {
        visitor.visitInterfaceDefinition(def, **compilation);
      }
    }
  }

  symbols = std::move(visitor.symbols);
}

//===----------------------------------------------------------------------===//
// Auto-Completion
//===----------------------------------------------------------------------===//

/// Map slang symbol kind to LSP CompletionItemKind.
static llvm::lsp::CompletionItemKind
mapToCompletionKind(slang::ast::SymbolKind slangKind) {
  using SK = slang::ast::SymbolKind;
  using CK = llvm::lsp::CompletionItemKind;

  switch (slangKind) {
  case SK::Definition:
    return CK::Module;
  case SK::Instance:
    return CK::Module;
  case SK::Package:
    return CK::Module;
  case SK::Net:
  case SK::Variable:
    return CK::Variable;
  case SK::Parameter:
    return CK::Constant;
  case SK::Port:
    return CK::Field;
  case SK::Subroutine:
    return CK::Function;
  case SK::ClassType:
  case SK::ClassProperty:
    return CK::Class;
  case SK::InterfacePort:
    return CK::Interface;
  case SK::EnumType:
  case SK::TypeAlias:
    return CK::Enum;
  case SK::EnumValue:
    return CK::EnumMember;
  case SK::PackedStructType:
  case SK::UnpackedStructType:
    return CK::Struct;
  default:
    return CK::Variable;
  }
}

/// Get a human-readable detail string for a symbol.
static std::string getCompletionDetail(const slang::ast::Symbol &symbol) {
  std::string result;
  llvm::raw_string_ostream os(result);

  switch (symbol.kind) {
  case slang::ast::SymbolKind::Variable:
  case slang::ast::SymbolKind::Net: {
    const auto &valueSymbol = symbol.as<slang::ast::ValueSymbol>();
    os << formatTypeDescription(valueSymbol.getType());
    break;
  }
  case slang::ast::SymbolKind::Port: {
    const auto &port = symbol.as<slang::ast::PortSymbol>();
    switch (port.direction) {
    case slang::ast::ArgumentDirection::In:
      os << "input ";
      break;
    case slang::ast::ArgumentDirection::Out:
      os << "output ";
      break;
    case slang::ast::ArgumentDirection::InOut:
      os << "inout ";
      break;
    case slang::ast::ArgumentDirection::Ref:
      os << "ref ";
      break;
    }
    os << formatTypeDescription(port.getType());
    break;
  }
  case slang::ast::SymbolKind::Parameter: {
    const auto &param = symbol.as<slang::ast::ParameterSymbol>();
    os << "parameter " << formatTypeDescription(param.getType());
    auto initExpr = param.getInitializer();
    if (initExpr) {
      slang::ast::EvalContext evalCtx(symbol);
      auto cv = initExpr->eval(evalCtx);
      if (cv)
        os << " = " << cv.toString();
    }
    break;
  }
  case slang::ast::SymbolKind::Definition: {
    os << "module";
    break;
  }
  case slang::ast::SymbolKind::Instance: {
    const auto &inst = symbol.as<slang::ast::InstanceSymbol>();
    os << inst.getDefinition().name << " instance";
    break;
  }
  case slang::ast::SymbolKind::Subroutine: {
    const auto &sub = symbol.as<slang::ast::SubroutineSymbol>();
    if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
      os << "function";
    else
      os << "task";
    break;
  }
  case slang::ast::SymbolKind::Package: {
    os << "package";
    break;
  }
  default:
    break;
  }

  return result;
}

/// Verilog/SystemVerilog keywords for completion.
static const char *const verilogKeywords[] = {
    // Module/interface keywords
    "module",
    "endmodule",
    "interface",
    "endinterface",
    "program",
    "endprogram",
    "package",
    "endpackage",
    // Port/signal keywords
    "input",
    "output",
    "inout",
    "wire",
    "reg",
    "logic",
    "integer",
    "real",
    "time",
    // Parameter keywords
    "parameter",
    "localparam",
    // Type keywords
    "signed",
    "unsigned",
    "bit",
    "byte",
    "shortint",
    "int",
    "longint",
    "shortreal",
    "string",
    "void",
    // Behavioral keywords
    "always",
    "always_comb",
    "always_ff",
    "always_latch",
    "initial",
    "final",
    "assign",
    // Control flow
    "if",
    "else",
    "case",
    "casex",
    "casez",
    "default",
    "endcase",
    "for",
    "while",
    "do",
    "foreach",
    "repeat",
    "forever",
    "begin",
    "end",
    "fork",
    "join",
    "join_any",
    "join_none",
    // Task/function
    "function",
    "endfunction",
    "task",
    "endtask",
    "return",
    // Class/OOP
    "class",
    "endclass",
    "extends",
    "implements",
    "virtual",
    "static",
    "protected",
    "local",
    "new",
    "this",
    "super",
    // Generate
    "generate",
    "endgenerate",
    "genvar",
    // Timing
    "posedge",
    "negedge",
    "edge",
    // Assertions
    "assert",
    "assume",
    "cover",
    "property",
    "sequence",
    // Miscellaneous
    "typedef",
    "enum",
    "struct",
    "union",
    "const",
    "automatic",
    "import",
    "export",
    nullptr};

/// Collect symbols visible at the given scope.
namespace {
class CompletionCollector
    : public slang::ast::ASTVisitor<CompletionCollector, true, true> {
public:
  CompletionCollector(llvm::lsp::CompletionList &completions,
                      llvm::StringRef prefix, slang::BufferID bufferId)
      : completions(completions), prefix(prefix), bufferId(bufferId) {}

  llvm::lsp::CompletionList &completions;
  llvm::StringRef prefix;
  slang::BufferID bufferId;

  void addSymbol(const slang::ast::Symbol &symbol) {
    if (symbol.name.empty())
      return;

    // Filter by prefix if provided
    if (!prefix.empty() &&
        !llvm::StringRef(symbol.name).starts_with_insensitive(prefix))
      return;

    llvm::lsp::CompletionItem item;
    item.label = std::string(symbol.name);
    item.kind = mapToCompletionKind(symbol.kind);
    item.detail = getCompletionDetail(symbol);
    item.insertText = std::string(symbol.name);
    item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;

    completions.items.push_back(std::move(item));
  }

  void visit(const slang::ast::InstanceBodySymbol &body) {
    // Add ports
    for (const auto *portSym : body.getPortList()) {
      addSymbol(*portSym);
    }

    // Add members
    for (const auto &member : body.members()) {
      switch (member.kind) {
      case slang::ast::SymbolKind::Net:
      case slang::ast::SymbolKind::Variable:
      case slang::ast::SymbolKind::Parameter:
      case slang::ast::SymbolKind::Instance:
      case slang::ast::SymbolKind::Subroutine:
        addSymbol(member);
        break;
      default:
        break;
      }
    }

    visitDefault(body);
  }

  void visit(const slang::ast::PackageSymbol &pkg) {
    for (const auto &member : pkg.members()) {
      addSymbol(member);
    }
  }

  template <typename T>
  void visit(const T &node) {
    visitDefault(node);
  }
};

/// Enumeration to distinguish completion contexts.
enum class VerilogCompletionContext {
  /// Normal context (keywords, symbols, etc.)
  Normal,
  /// Member access after '.' on an instance/variable
  MemberAccess,
  /// Port connection in module instantiation (e.g., .port_name())
  PortConnection
};

/// Result of completion context analysis, including the chain of identifiers
/// for chained member access (e.g., "obj.field1.field2" -> ["obj", "field1"]).
struct CompletionContextResult {
  VerilogCompletionContext context;
  /// Chain of identifiers before the final dot (for member access).
  /// E.g., for "obj.field1.field2", this would be ["obj", "field1"].
  llvm::SmallVector<std::string, 4> identifierChain;
};

/// Analyze the completion context and extract the identifier chain if in member
/// access or port connection context. Supports chained member access like
/// "obj.field1.field2.".
static CompletionContextResult analyzeCompletionContext(std::string_view text,
                                                        uint32_t offset) {
  // Find the start of the current identifier (prefix)
  uint32_t prefixStart = offset;
  while (prefixStart > 0 &&
         (std::isalnum(text[prefixStart - 1]) || text[prefixStart - 1] == '_'))
    --prefixStart;

  // Check if there's a '.' before the prefix
  uint32_t dotPos = prefixStart;
  while (dotPos > 0 && std::isspace(text[dotPos - 1]))
    --dotPos;

  if (dotPos > 0 && text[dotPos - 1] == '.') {
    // Parse the entire chain of identifiers separated by dots
    // E.g., "obj.field1.field2" -> ["obj", "field1"]
    llvm::SmallVector<std::string, 4> chain;
    uint32_t parsePos = dotPos - 1; // Start at the first dot

    while (true) {
      // Skip whitespace before the dot
      while (parsePos > 0 && std::isspace(text[parsePos - 1]))
        --parsePos;

      // Find the identifier before this position
      uint32_t idEnd = parsePos;
      uint32_t idStart = parsePos;
      while (idStart > 0 &&
             (std::isalnum(text[idStart - 1]) || text[idStart - 1] == '_'))
        --idStart;

      if (idStart == idEnd) {
        // No identifier found, stop
        break;
      }

      // Add the identifier to the front of the chain
      chain.insert(chain.begin(),
                   std::string(text.substr(idStart, idEnd - idStart)));

      // Check if there's another dot before this identifier
      parsePos = idStart;
      while (parsePos > 0 && std::isspace(text[parsePos - 1]))
        --parsePos;

      if (parsePos == 0 || text[parsePos - 1] != '.') {
        // No more dots, we've found the entire chain
        break;
      }

      // Move past the dot to continue parsing
      parsePos--;
    }

    // Try to detect if we're in a port connection context
    // Look back from the start of the chain for opening paren '('
    uint32_t scanPos = chain.empty() ? dotPos : prefixStart;
    // Find the start of the first identifier in the chain
    if (!chain.empty()) {
      // Re-find the start of the first identifier
      scanPos = dotPos - 1;
      for (size_t i = 0; i < chain.size() - 1; ++i) {
        // Skip back past each identifier and its preceding dot
        while (scanPos > 0 && std::isspace(text[scanPos - 1]))
          --scanPos;
        while (scanPos > 0 &&
               (std::isalnum(text[scanPos - 1]) || text[scanPos - 1] == '_'))
          --scanPos;
        while (scanPos > 0 && std::isspace(text[scanPos - 1]))
          --scanPos;
        if (scanPos > 0 && text[scanPos - 1] == '.')
          --scanPos;
      }
      while (scanPos > 0 && std::isspace(text[scanPos - 1]))
        --scanPos;
      while (scanPos > 0 &&
             (std::isalnum(text[scanPos - 1]) || text[scanPos - 1] == '_'))
        --scanPos;
    }

    int parenDepth = 0;
    bool foundParenContext = false;
    while (scanPos > 0) {
      --scanPos;
      char c = text[scanPos];
      if (c == ')') {
        parenDepth++;
      } else if (c == '(') {
        if (parenDepth == 0) {
          foundParenContext = true;
          break;
        }
        parenDepth--;
      } else if (c == ';' || c == '\n') {
        // Stop at statement boundary or newline (simple heuristic)
        // But continue if we're still in parentheses
        if (parenDepth == 0)
          break;
      }
    }

    // If we found a paren context and the chain is empty, it might be
    // a port connection like ".port_name(value)"
    if (foundParenContext && chain.empty()) {
      return {VerilogCompletionContext::PortConnection, {}};
    }

    return {VerilogCompletionContext::MemberAccess, std::move(chain)};
  }

  return {VerilogCompletionContext::Normal, {}};
}

/// Helper to search a scope for a symbol by name, recursively.
static const slang::ast::Symbol *
findSymbolInScope(const slang::ast::Scope &scope, llvm::StringRef name) {
  // First, try direct lookup
  for (const auto &member : scope.members()) {
    if (name == llvm::StringRef(member.name.data(), member.name.size()))
      return &member;

    // If this is a scope, search inside it too (for procedural blocks, etc.)
    if (auto *innerScope = member.as_if<slang::ast::Scope>()) {
      if (auto *found = findSymbolInScope(*innerScope, name))
        return found;
    }
  }
  return nullptr;
}

/// Find a symbol by name in the compilation (search variables, instances,
/// etc.)
static const slang::ast::Symbol *
findSymbolByName(slang::ast::Compilation &compilation, llvm::StringRef name) {
  // Search in top instances first
  for (auto *inst : compilation.getRoot().topInstances) {
    // Search ports
    for (const auto *port : inst->body.getPortList()) {
      if (name == llvm::StringRef(port->name.data(), port->name.size()))
        return port;
    }

    // Search members recursively
    if (auto *found = findSymbolInScope(inst->body, name))
      return found;
  }

  // Search in packages
  for (auto *pkg : compilation.getPackages()) {
    if (auto *found = findSymbolInScope(*pkg, name))
      return found;
  }

  return nullptr;
}

/// Resolve a chain of member accesses starting from a base symbol.
/// Returns the final type in the chain, or nullptr if resolution fails.
/// E.g., for chain ["obj", "field1"], resolves obj, gets its type,
/// finds field1 in that type, and returns the type of field1.
struct ChainResolutionResult {
  const slang::ast::Type *finalType = nullptr;
  const slang::ast::InstanceBodySymbol *instanceBody = nullptr;
  enum Kind { None, ClassType, InstanceType, InterfaceType } kind = None;
};

static ChainResolutionResult
resolveIdentifierChain(slang::ast::Compilation &compilation,
                       llvm::ArrayRef<std::string> chain) {
  if (chain.empty())
    return {};

  // Start by finding the first symbol in the chain
  const slang::ast::Symbol *currentSym =
      findSymbolByName(compilation, chain[0]);
  if (!currentSym)
    return {};

  ChainResolutionResult result;

  // If chain has only one element, return info about that symbol
  if (chain.size() == 1) {
    if (auto *inst = currentSym->as_if<slang::ast::InstanceSymbol>()) {
      result.kind = ChainResolutionResult::InstanceType;
      result.instanceBody = &inst->body;
      return result;
    }
    if (auto *valSym = currentSym->as_if<slang::ast::ValueSymbol>()) {
      const auto &type = valSym->getType();
      if (auto *classType = type.as_if<slang::ast::ClassType>()) {
        result.kind = ChainResolutionResult::ClassType;
        result.finalType = classType;
        return result;
      }
      if (auto *ifaceType =
              type.getCanonicalType().as_if<slang::ast::VirtualInterfaceType>()) {
        result.kind = ChainResolutionResult::InterfaceType;
        result.instanceBody = &ifaceType->iface.body;
        return result;
      }
    }
    return {};
  }

  // Walk through the rest of the chain
  for (size_t i = 1; i < chain.size(); ++i) {
    llvm::StringRef memberName = chain[i];
    const slang::ast::Symbol *nextSym = nullptr;

    // Check if current is an instance
    if (auto *inst = currentSym->as_if<slang::ast::InstanceSymbol>()) {
      // Search for member in the instance body
      nextSym = findSymbolInScope(inst->body, memberName);
    }
    // Check if current is a value with a type
    else if (auto *valSym = currentSym->as_if<slang::ast::ValueSymbol>()) {
      const auto &type = valSym->getType();
      if (auto *classType = type.as_if<slang::ast::ClassType>()) {
        // Search for member in the class
        for (const auto &member : classType->members()) {
          if (memberName == llvm::StringRef(member.name.data(), member.name.size())) {
            nextSym = &member;
            break;
          }
        }
      } else if (auto *ifaceType =
                     type.getCanonicalType().as_if<slang::ast::VirtualInterfaceType>()) {
        nextSym = findSymbolInScope(ifaceType->iface.body, memberName);
      }
    }

    if (!nextSym)
      return {}; // Failed to resolve this step

    currentSym = nextSym;
  }

  // Return info about the final symbol
  if (auto *inst = currentSym->as_if<slang::ast::InstanceSymbol>()) {
    result.kind = ChainResolutionResult::InstanceType;
    result.instanceBody = &inst->body;
    return result;
  }
  if (auto *valSym = currentSym->as_if<slang::ast::ValueSymbol>()) {
    const auto &type = valSym->getType();
    if (auto *classType = type.as_if<slang::ast::ClassType>()) {
      result.kind = ChainResolutionResult::ClassType;
      result.finalType = classType;
      return result;
    }
    if (auto *ifaceType =
            type.getCanonicalType().as_if<slang::ast::VirtualInterfaceType>()) {
      result.kind = ChainResolutionResult::InterfaceType;
      result.instanceBody = &ifaceType->iface.body;
      return result;
    }
  }

  return {};
}

/// Helper to unwrap a TransparentMemberSymbol if present, returning the
/// underlying symbol. If not a transparent member, returns the original symbol.
static const slang::ast::Symbol &
unwrapTransparentMember(const slang::ast::Symbol &member) {
  if (member.kind == slang::ast::SymbolKind::TransparentMember)
    return member.as<slang::ast::TransparentMemberSymbol>().wrapped;
  return member;
}

/// Helper to find the class name where a member was originally defined.
/// Returns empty string if defined in the class being examined (not inherited).
/// For TransparentMemberSymbol, we unwrap and check the actual symbol's scope.
static std::string getInheritedFromClassName(const slang::ast::Symbol &member,
                                             const slang::ast::ClassType &currentClass) {
  // If this is a TransparentMember, it wraps an inherited symbol
  if (member.kind == slang::ast::SymbolKind::TransparentMember) {
    const auto &transparentMember =
        member.as<slang::ast::TransparentMemberSymbol>();
    const slang::ast::Symbol &wrapped = transparentMember.wrapped;

    // Get the scope where this symbol was originally declared
    const slang::ast::Scope *wrappedScope = wrapped.getParentScope();
    if (wrappedScope) {
      const slang::ast::Symbol *scopeSymbol = &wrappedScope->asSymbol();
      if (scopeSymbol->kind == slang::ast::SymbolKind::ClassType)
        return std::string(scopeSymbol->name);
    }
    return "";
  }

  // Not a transparent member - it's directly defined in the current class
  return "";
}

/// Add completion items for members of a class type, including inherited
/// members from base classes and implemented interfaces.
/// Inherited members are annotated with "(from ClassName)" in the detail.
static void addClassMemberCompletions(const slang::ast::ClassType &classType,
                                      llvm::StringRef prefix,
                                      llvm::lsp::CompletionList &completions) {
  // classType.members() already includes inherited members from the class
  // hierarchy. Inherited members are wrapped in TransparentMemberSymbol.
  for (const auto &member : classType.members()) {
    if (member.name.empty())
      continue;

    // Filter by prefix if provided
    if (!prefix.empty() &&
        !llvm::StringRef(member.name).starts_with_insensitive(prefix))
      continue;

    // Unwrap TransparentMemberSymbol to get the actual symbol for completion
    const slang::ast::Symbol &actualMember = unwrapTransparentMember(member);

    llvm::lsp::CompletionItem item;
    item.label = std::string(member.name);
    item.kind = mapToCompletionKind(actualMember.kind);

    // Build detail string, including inheritance info if applicable
    std::string detail = getCompletionDetail(actualMember);
    std::string inheritedFrom = getInheritedFromClassName(member, classType);
    if (!inheritedFrom.empty()) {
      if (!detail.empty())
        detail += " ";
      detail += "(from " + inheritedFrom + ")";
    }
    item.detail = std::move(detail);

    item.insertText = std::string(member.name);
    item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;

    // For methods, add parentheses
    if (actualMember.kind == slang::ast::SymbolKind::Subroutine) {
      const auto &sub = actualMember.as<slang::ast::SubroutineSymbol>();
      if (sub.getArguments().empty()) {
        item.insertText = std::string(member.name) + "()";
      } else {
        item.insertText = std::string(member.name) + "(${0})";
        item.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
      }
      item.kind = llvm::lsp::CompletionItemKind::Method;
    } else if (actualMember.kind == slang::ast::SymbolKind::ClassProperty) {
      item.kind = llvm::lsp::CompletionItemKind::Property;
    }

    completions.items.push_back(std::move(item));
  }
}

/// Add completion items for ports of a module/interface instance.
static void addPortCompletions(const slang::ast::InstanceBodySymbol &body,
                               llvm::StringRef prefix,
                               llvm::lsp::CompletionList &completions) {
  for (const auto *port : body.getPortList()) {
    if (port->name.empty())
      continue;

    // Filter by prefix if provided
    if (!prefix.empty() &&
        !llvm::StringRef(port->name).starts_with_insensitive(prefix))
      continue;

    llvm::lsp::CompletionItem item;
    item.label = std::string(port->name);
    item.kind = llvm::lsp::CompletionItemKind::Field;
    item.detail = getCompletionDetail(*port);
    item.insertText = std::string(port->name);
    item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;

    completions.items.push_back(std::move(item));
  }
}

/// Add completion items for members of an instance body (nets, variables,
/// etc.)
static void addInstanceMemberCompletions(
    const slang::ast::InstanceBodySymbol &body, llvm::StringRef prefix,
    llvm::lsp::CompletionList &completions) {
  // Add ports
  addPortCompletions(body, prefix, completions);

  // Add internal members
  for (const auto &member : body.members()) {
    if (member.name.empty())
      continue;

    // Filter by prefix if provided
    if (!prefix.empty() &&
        !llvm::StringRef(member.name).starts_with_insensitive(prefix))
      continue;

    switch (member.kind) {
    case slang::ast::SymbolKind::Net:
    case slang::ast::SymbolKind::Variable:
    case slang::ast::SymbolKind::Parameter:
    case slang::ast::SymbolKind::Subroutine: {
      llvm::lsp::CompletionItem item;
      item.label = std::string(member.name);
      item.kind = mapToCompletionKind(member.kind);
      item.detail = getCompletionDetail(member);
      item.insertText = std::string(member.name);
      item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;
      completions.items.push_back(std::move(item));
      break;
    }
    default:
      break;
    }
  }
}

} // namespace

void VerilogDocument::getCompletions(const llvm::lsp::URIForFile &uri,
                                     const llvm::lsp::Position &pos,
                                     llvm::lsp::CompletionList &completions) {
  completions.isIncomplete = false;

  // Get the text at the current position to determine prefix
  auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);
  auto offsetOpt = lspPositionToOffset(pos);
  if (!offsetOpt)
    return;

  uint32_t offset = *offsetOpt;

  // Analyze the completion context
  CompletionContextResult contextResult = analyzeCompletionContext(text, offset);

  // Handle member access completion (after '.')
  // Supports chained access like obj.field1.field2.
  if (contextResult.context == VerilogCompletionContext::MemberAccess &&
      !contextResult.identifierChain.empty() && succeeded(compilation)) {
    // Get the prefix after the '.'
    uint32_t prefixStart = offset;
    while (prefixStart > 0 && (std::isalnum(text[prefixStart - 1]) ||
                               text[prefixStart - 1] == '_'))
      --prefixStart;
    llvm::StringRef memberPrefix(text.data() + prefixStart,
                                 offset - prefixStart);

    // Resolve the entire identifier chain to get the final type
    ChainResolutionResult resolution =
        resolveIdentifierChain(**compilation, contextResult.identifierChain);

    switch (resolution.kind) {
    case ChainResolutionResult::ClassType:
      if (resolution.finalType) {
        addClassMemberCompletions(
            *resolution.finalType->as_if<slang::ast::ClassType>(), memberPrefix,
            completions);
        return;
      }
      break;
    case ChainResolutionResult::InstanceType:
      if (resolution.instanceBody) {
        addInstanceMemberCompletions(*resolution.instanceBody, memberPrefix,
                                     completions);
        return;
      }
      break;
    case ChainResolutionResult::InterfaceType:
      if (resolution.instanceBody) {
        addPortCompletions(*resolution.instanceBody, memberPrefix, completions);
        return;
      }
      break;
    case ChainResolutionResult::None:
      // Fall through to normal completion
      break;
    }
  }

  // Handle port connection completion in module instantiation
  if (contextResult.context == VerilogCompletionContext::PortConnection && succeeded(compilation)) {
    // Try to find the module being instantiated by looking back in the text
    // This is a simplified approach - in a real implementation, you'd use
    // the AST to find the enclosing instantiation
    uint32_t scanPos = offset;
    while (scanPos > 0 && text[scanPos - 1] != '(')
      --scanPos;
    if (scanPos > 0) {
      --scanPos; // Skip the '('
      // Skip whitespace
      while (scanPos > 0 && std::isspace(text[scanPos - 1]))
        --scanPos;
      // Skip the instance name
      while (scanPos > 0 && (std::isalnum(text[scanPos - 1]) ||
                             text[scanPos - 1] == '_'))
        --scanPos;
      // Now find the module name before the instance name
      while (scanPos > 0 && std::isspace(text[scanPos - 1]))
        --scanPos;
      uint32_t moduleEnd = scanPos;
      while (scanPos > 0 && (std::isalnum(text[scanPos - 1]) ||
                             text[scanPos - 1] == '_'))
        --scanPos;
      std::string moduleName(text.substr(scanPos, moduleEnd - scanPos));

      // Find the module definition
      for (const auto *def : (*compilation)->getDefinitions()) {
        if (def->name == moduleName) {
          // Get the prefix after the '.'
          uint32_t prefixStart = offset;
          while (prefixStart > 0 && (std::isalnum(text[prefixStart - 1]) ||
                                     text[prefixStart - 1] == '_'))
            --prefixStart;
          llvm::StringRef portPrefix(text.data() + prefixStart,
                                     offset - prefixStart);

          // Add port completions from the definition's instantiated form
          for (auto *inst : (*compilation)->getRoot().topInstances) {
            if (&inst->getDefinition() == def) {
              addPortCompletions(inst->body, portPrefix, completions);
              return;
            }
          }
          break;
        }
      }
    }
  }

  // Find the start of the current identifier (prefix)
  uint32_t prefixStart = offset;
  while (prefixStart > 0 &&
         (std::isalnum(text[prefixStart - 1]) || text[prefixStart - 1] == '_'))
    --prefixStart;

  llvm::StringRef prefix(text.data() + prefixStart, offset - prefixStart);

  // Add keyword completions
  for (const char *const *kw = verilogKeywords; *kw; ++kw) {
    llvm::StringRef keyword(*kw);
    if (prefix.empty() || keyword.starts_with_insensitive(prefix)) {
      llvm::lsp::CompletionItem item;
      item.label = keyword.str();
      item.kind = llvm::lsp::CompletionItemKind::Keyword;
      item.insertText = keyword.str();
      item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;
      completions.items.push_back(std::move(item));
    }
  }

  // Add snippet completions
  if (prefix.empty() || llvm::StringRef("module").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem moduleSnippet;
    moduleSnippet.label = "module (snippet)";
    moduleSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    moduleSnippet.detail = "Module template";
    moduleSnippet.insertText =
        "module ${1:module_name} (\n"
        "  input ${2:clk},\n"
        "  input ${3:rst},\n"
        "  ${4:// ports}\n"
        ");\n"
        "  ${0:// body}\n"
        "endmodule";
    moduleSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(moduleSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("always_ff").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem alwaysFFSnippet;
    alwaysFFSnippet.label = "always_ff (snippet)";
    alwaysFFSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    alwaysFFSnippet.detail = "Sequential always block";
    alwaysFFSnippet.insertText =
        "always_ff @(posedge ${1:clk} or negedge ${2:rst_n}) begin\n"
        "  if (!${2:rst_n}) begin\n"
        "    ${3:// reset}\n"
        "  end else begin\n"
        "    ${0:// logic}\n"
        "  end\n"
        "end";
    alwaysFFSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(alwaysFFSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("always_comb").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem alwaysCombSnippet;
    alwaysCombSnippet.label = "always_comb (snippet)";
    alwaysCombSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    alwaysCombSnippet.detail = "Combinational always block";
    alwaysCombSnippet.insertText = "always_comb begin\n"
                                   "  ${0:// logic}\n"
                                   "end";
    alwaysCombSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(alwaysCombSnippet));
  }

  // UVM-specific snippet completions
  if (prefix.empty() ||
      llvm::StringRef("uvm_component").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmComponentSnippet;
    uvmComponentSnippet.label = "uvm_component (snippet)";
    uvmComponentSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmComponentSnippet.detail = "UVM component class template";
    uvmComponentSnippet.insertText =
        "class ${1:my_component} extends uvm_component;\n"
        "  `uvm_component_utils(${1:my_component})\n"
        "\n"
        "  function new(string name, uvm_component parent);\n"
        "    super.new(name, parent);\n"
        "  endfunction\n"
        "\n"
        "  function void build_phase(uvm_phase phase);\n"
        "    super.build_phase(phase);\n"
        "    ${2:// build phase}\n"
        "  endfunction\n"
        "\n"
        "  task run_phase(uvm_phase phase);\n"
        "    ${0:// run phase}\n"
        "  endtask\n"
        "endclass";
    uvmComponentSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmComponentSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_object").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmObjectSnippet;
    uvmObjectSnippet.label = "uvm_object (snippet)";
    uvmObjectSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmObjectSnippet.detail = "UVM object class template";
    uvmObjectSnippet.insertText =
        "class ${1:my_object} extends uvm_object;\n"
        "  `uvm_object_utils(${1:my_object})\n"
        "\n"
        "  function new(string name = \"${1:my_object}\");\n"
        "    super.new(name);\n"
        "  endfunction\n"
        "\n"
        "  ${0:// properties and methods}\n"
        "endclass";
    uvmObjectSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmObjectSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_sequence_item").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmSeqItemSnippet;
    uvmSeqItemSnippet.label = "uvm_sequence_item (snippet)";
    uvmSeqItemSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmSeqItemSnippet.detail = "UVM sequence item class template";
    uvmSeqItemSnippet.insertText =
        "class ${1:my_item} extends uvm_sequence_item;\n"
        "  `uvm_object_utils_begin(${1:my_item})\n"
        "    ${2:// field macros}\n"
        "  `uvm_object_utils_end\n"
        "\n"
        "  rand ${3:bit [7:0]} ${4:data};\n"
        "\n"
        "  function new(string name = \"${1:my_item}\");\n"
        "    super.new(name);\n"
        "  endfunction\n"
        "\n"
        "  ${0:// constraints and methods}\n"
        "endclass";
    uvmSeqItemSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmSeqItemSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_sequence").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmSequenceSnippet;
    uvmSequenceSnippet.label = "uvm_sequence (snippet)";
    uvmSequenceSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmSequenceSnippet.detail = "UVM sequence class template";
    uvmSequenceSnippet.insertText =
        "class ${1:my_sequence} extends uvm_sequence #(${2:my_item});\n"
        "  `uvm_object_utils(${1:my_sequence})\n"
        "\n"
        "  function new(string name = \"${1:my_sequence}\");\n"
        "    super.new(name);\n"
        "  endfunction\n"
        "\n"
        "  task body();\n"
        "    ${0:// sequence body}\n"
        "  endtask\n"
        "endclass";
    uvmSequenceSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmSequenceSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_driver").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmDriverSnippet;
    uvmDriverSnippet.label = "uvm_driver (snippet)";
    uvmDriverSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmDriverSnippet.detail = "UVM driver class template";
    uvmDriverSnippet.insertText =
        "class ${1:my_driver} extends uvm_driver #(${2:my_item});\n"
        "  `uvm_component_utils(${1:my_driver})\n"
        "\n"
        "  function new(string name, uvm_component parent);\n"
        "    super.new(name, parent);\n"
        "  endfunction\n"
        "\n"
        "  task run_phase(uvm_phase phase);\n"
        "    forever begin\n"
        "      seq_item_port.get_next_item(req);\n"
        "      ${0:// drive transaction}\n"
        "      seq_item_port.item_done();\n"
        "    end\n"
        "  endtask\n"
        "endclass";
    uvmDriverSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmDriverSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_monitor").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmMonitorSnippet;
    uvmMonitorSnippet.label = "uvm_monitor (snippet)";
    uvmMonitorSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmMonitorSnippet.detail = "UVM monitor class template";
    uvmMonitorSnippet.insertText =
        "class ${1:my_monitor} extends uvm_monitor;\n"
        "  `uvm_component_utils(${1:my_monitor})\n"
        "\n"
        "  uvm_analysis_port #(${2:my_item}) ap;\n"
        "\n"
        "  function new(string name, uvm_component parent);\n"
        "    super.new(name, parent);\n"
        "  endfunction\n"
        "\n"
        "  function void build_phase(uvm_phase phase);\n"
        "    super.build_phase(phase);\n"
        "    ap = new(\"ap\", this);\n"
        "  endfunction\n"
        "\n"
        "  task run_phase(uvm_phase phase);\n"
        "    ${0:// monitoring logic}\n"
        "  endtask\n"
        "endclass";
    uvmMonitorSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmMonitorSnippet));
  }

  if (prefix.empty() ||
      llvm::StringRef("uvm_test").starts_with_insensitive(prefix)) {
    llvm::lsp::CompletionItem uvmTestSnippet;
    uvmTestSnippet.label = "uvm_test (snippet)";
    uvmTestSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
    uvmTestSnippet.detail = "UVM test class template";
    uvmTestSnippet.insertText =
        "class ${1:my_test} extends uvm_test;\n"
        "  `uvm_component_utils(${1:my_test})\n"
        "\n"
        "  ${2:my_env} env;\n"
        "\n"
        "  function new(string name, uvm_component parent);\n"
        "    super.new(name, parent);\n"
        "  endfunction\n"
        "\n"
        "  function void build_phase(uvm_phase phase);\n"
        "    super.build_phase(phase);\n"
        "    env = ${2:my_env}::type_id::create(\"env\", this);\n"
        "  endfunction\n"
        "\n"
        "  task run_phase(uvm_phase phase);\n"
        "    phase.raise_objection(this);\n"
        "    ${0:// test body}\n"
        "    phase.drop_objection(this);\n"
        "  endtask\n"
        "endclass";
    uvmTestSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
    completions.items.push_back(std::move(uvmTestSnippet));
  }

  // Add symbols from compilation
  if (succeeded(compilation)) {
    const auto &root = (*compilation)->getRoot();
    CompletionCollector collector(completions, prefix, mainBufferId);

    // Collect from packages
    for (auto *package : (*compilation)->getPackages()) {
      collector.visit(*package);
    }

    // Collect from modules
    for (auto *inst : root.topInstances) {
      collector.visit(inst->body);
    }

    // Add module names for instantiation
    for (const auto *def : (*compilation)->getDefinitions()) {
      if (prefix.empty() ||
          llvm::StringRef(def->name).starts_with_insensitive(prefix)) {
        llvm::lsp::CompletionItem item;
        item.label = std::string(def->name);
        item.kind = llvm::lsp::CompletionItemKind::Module;
        item.detail = "module definition";
        item.insertText = std::string(def->name);
        item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;
        completions.items.push_back(std::move(item));

        // Also add an instantiation snippet
        llvm::lsp::CompletionItem instSnippet;
        instSnippet.label = std::string(def->name) + " (instantiate)";
        instSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
        instSnippet.detail = "Module instantiation";

        std::string snippet = std::string(def->name) + " ${1:inst_name} (\n";
        snippet += "  ${0:// connections}\n";
        snippet += ");";
        instSnippet.insertText = snippet;
        instSnippet.insertTextFormat = llvm::lsp::InsertTextFormat::Snippet;
        completions.items.push_back(std::move(instSnippet));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Code Actions
//===----------------------------------------------------------------------===//

/// Check if a diagnostic message indicates an unknown identifier.
static bool isUnknownIdentifierDiagnostic(llvm::StringRef message) {
  return message.contains("unknown identifier") ||
         message.contains("undeclared identifier") ||
         message.contains("use of undeclared") ||
         message.contains("undefined") ||
         message.contains("was not found");
}

/// Check if a diagnostic message indicates a width mismatch.
static bool isWidthMismatchDiagnostic(llvm::StringRef message) {
  return message.contains("width") && (message.contains("mismatch") ||
                                       message.contains("incompatible"));
}

/// Check if a diagnostic message indicates an unknown module.
static bool isUnknownModuleDiagnostic(llvm::StringRef message) {
  return message.contains("unknown module") ||
         (message.contains("definition of") && message.contains("not found"));
}

/// Check if a diagnostic message indicates a missing import.
static bool isMissingImportDiagnostic(llvm::StringRef message) {
  return message.contains("unknown type") ||
         message.contains("unknown class") ||
         (message.contains("not found") && message.contains("package"));
}

/// Check if the type name looks like a UVM type.
static bool isUVMTypeName(llvm::StringRef typeName) {
  return typeName.starts_with("uvm_") || typeName.starts_with("UVM_");
}

/// Check if the diagnostic message mentions a UVM type.
static bool mentionsUVMType(llvm::StringRef message) {
  return message.contains("uvm_") || message.contains("UVM_");
}

/// Check if a diagnostic message indicates a missing semicolon.
static bool isMissingSemicolonDiagnostic(llvm::StringRef message) {
  // Common patterns from slang/verilog compilers for missing semicolon
  return (message.contains("expected") && message.contains(";")) ||
         (message.contains("expected") && message.contains("semicolon")) ||
         message.contains("missing ';'") ||
         message.contains("missing semicolon");
}

/// Check if a diagnostic message indicates a begin/end block is needed.
static bool isBeginEndNeededDiagnostic(llvm::StringRef message) {
  // Patterns indicating multiple statements where only one is expected
  return message.contains("expected single statement") ||
         message.contains("begin/end required") ||
         (message.contains("multiple statements") &&
          message.contains("begin")) ||
         (message.contains("statement") && message.contains("after") &&
          message.contains("if")) ||
         (message.contains("statement") && message.contains("after") &&
          message.contains("else"));
}

/// Common typos in Verilog/SystemVerilog and their corrections.
/// Returns the corrected keyword if the input is a known typo, otherwise empty.
static std::optional<std::string> getTypoCorrection(llvm::StringRef typo) {
  // Map of common typos to correct keywords
  static const llvm::StringMap<std::string> typoMap = {
      // Type keywords
      {"rge", "reg"},
      {"regg", "reg"},
      {"reg ", "reg"},
      {"wrie", "wire"},
      {"wirre", "wire"},
      {"wir", "wire"},
      {"lgic", "logic"},
      {"logc", "logic"},
      {"loigc", "logic"},
      {"iteger", "integer"},
      {"integr", "integer"},
      {"intger", "integer"},
      {"inout ", "inout"},
      {"iput", "input"},
      {"intpu", "input"},
      {"inptu", "input"},
      {"otuput", "output"},
      {"outpu", "output"},
      {"ouput", "output"},
      {"otput", "output"},
      // Module keywords
      {"modle", "module"},
      {"modul", "module"},
      {"moduel", "module"},
      {"endmodle", "endmodule"},
      {"endmodul", "endmodule"},
      {"endmoduel", "endmodule"},
      // Block keywords
      {"alawys", "always"},
      {"alwyas", "always"},
      {"alwasy", "always"},
      {"asign", "assign"},
      {"assgin", "assign"},
      {"assgn", "assign"},
      {"intial", "initial"},
      {"inital", "initial"},
      {"iniital", "initial"},
      {"begn", "begin"},
      {"bgin", "begin"},
      {"bigin", "begin"},
      {"ned", "end"},
      {"endd", "end"},
      // Control flow
      {"esle", "else"},
      {"els", "else"},
      {"fi", "if"},
      {"iff", "if"},
      {"fro", "for"},
      {"fo", "for"},
      {"whlie", "while"},
      {"whle", "while"},
      {"wihle", "while"},
      {"csae", "case"},
      {"caes", "case"},
      {"cas", "case"},
      {"endcsae", "endcase"},
      {"endcaes", "endcase"},
      // Task/Function
      {"fucntion", "function"},
      {"funtion", "function"},
      {"funciton", "function"},
      {"endfucntion", "endfunction"},
      {"endfuntion", "endfunction"},
      {"taks", "task"},
      {"tak", "task"},
      {"endtaks", "endtask"},
      {"endtak", "endtask"},
      // Other common typos
      {"parmeter", "parameter"},
      {"paramter", "parameter"},
      {"parmaeter", "parameter"},
      {"localpram", "localparam"},
      {"localpaarm", "localparam"},
      {"genvar ", "genvar"},
      {"gnevar", "genvar"},
      {"geerate", "generate"},
      {"genrate", "generate"},
      {"pacakge", "package"},
      {"packge", "package"},
      {"endpackge", "endpackage"},
      {"endpacakge", "endpackage"},
      {"calss", "class"},
      {"clss", "class"},
      {"endcalss", "endclass"},
      {"endclss", "endclass"},
      {"interfce", "interface"},
      {"interafce", "interface"},
      {"endinterfce", "endinterface"},
      {"posege", "posedge"},
      {"posedeg", "posedge"},
      {"negedeg", "negedge"},
      {"negege", "negedge"},
  };

  auto it = typoMap.find(typo.lower());
  if (it != typoMap.end())
    return it->second;

  return std::nullopt;
}

/// Check if a diagnostic indicates a possible typo (unknown keyword/identifier
/// that looks like a keyword).
static bool isPossibleTypoDiagnostic(llvm::StringRef message) {
  return message.contains("unknown") || message.contains("unexpected") ||
         message.contains("expected") || message.contains("invalid");
}

// Forward declaration for use in getTypoAndCorrection.
static std::optional<std::string>
extractIdentifierFromMessage(llvm::StringRef message);

/// Extract a potential typo from a diagnostic message and source text.
static std::optional<std::pair<std::string, std::string>>
getTypoAndCorrection(llvm::StringRef message, llvm::StringRef sourceAtRange) {
  // First try to get the identifier from the message
  auto extracted = extractIdentifierFromMessage(message);

  // Check if extracted or source text is a known typo
  if (extracted) {
    auto correction = getTypoCorrection(*extracted);
    if (correction)
      return std::make_pair(*extracted, *correction);
  }

  // Try the source text at range
  if (!sourceAtRange.empty()) {
    auto correction = getTypoCorrection(sourceAtRange);
    if (correction)
      return std::make_pair(sourceAtRange.str(), *correction);
  }

  return std::nullopt;
}

/// Find the position for inserting an import statement.
/// This looks for existing import statements or the start of the module/class.
static llvm::lsp::Position findImportInsertionPoint(
    std::string_view sourceText, const std::vector<uint32_t> &lineOffsets) {
  if (lineOffsets.empty())
    return llvm::lsp::Position(0, 0);

  // Search for patterns to find a good insertion point
  for (size_t line = 0; line < lineOffsets.size(); ++line) {
    size_t lineStart = lineOffsets[line];
    size_t lineEnd = (line + 1 < lineOffsets.size())
                         ? lineOffsets[line + 1] - 1
                         : sourceText.size();

    std::string_view lineText = sourceText.substr(lineStart, lineEnd - lineStart);

    // If there's an existing import, insert after it
    if (lineText.find("import") != std::string_view::npos &&
        lineText.find("::") != std::string_view::npos) {
      // Find the end of this import statement (line with semicolon)
      for (size_t i = line; i < lineOffsets.size(); ++i) {
        size_t iLineStart = lineOffsets[i];
        size_t iLineEnd = (i + 1 < lineOffsets.size())
                             ? lineOffsets[i + 1] - 1
                             : sourceText.size();
        std::string_view iLineText = sourceText.substr(iLineStart, iLineEnd - iLineStart);
        if (iLineText.find(';') != std::string_view::npos) {
          return llvm::lsp::Position(static_cast<int>(i + 1), 0);
        }
      }
    }

    // If we find module/class/package, insert right after the declaration line
    if (lineText.find("module ") != std::string_view::npos ||
        lineText.find("class ") != std::string_view::npos ||
        lineText.find("package ") != std::string_view::npos) {
      // Find the end of the declaration (line with semicolon or closing paren)
      for (size_t i = line; i < lineOffsets.size(); ++i) {
        size_t iLineStart = lineOffsets[i];
        size_t iLineEnd = (i + 1 < lineOffsets.size())
                             ? lineOffsets[i + 1] - 1
                             : sourceText.size();
        std::string_view iLineText = sourceText.substr(iLineStart, iLineEnd - iLineStart);
        if (iLineText.find(");") != std::string_view::npos ||
            (iLineText.find(';') != std::string_view::npos &&
             iLineText.find('(') == std::string_view::npos)) {
          return llvm::lsp::Position(static_cast<int>(i + 1), 0);
        }
      }
    }
  }

  // Fallback: insert at the beginning of the file
  return llvm::lsp::Position(0, 0);
}

/// Extract identifier name from a diagnostic message.
/// Common patterns: "use of undeclared identifier 'foo'"
///                  "unknown identifier 'bar'"
static std::optional<std::string> extractIdentifierFromMessage(llvm::StringRef message) {
  // Look for pattern: 'identifier_name'
  size_t quoteStart = message.find('\'');
  if (quoteStart == llvm::StringRef::npos)
    return std::nullopt;

  size_t quoteEnd = message.find('\'', quoteStart + 1);
  if (quoteEnd == llvm::StringRef::npos)
    return std::nullopt;

  llvm::StringRef name = message.slice(quoteStart + 1, quoteEnd);
  if (name.empty())
    return std::nullopt;

  return name.str();
}

/// Extract the text at a given range from source.
static std::string extractTextAtRange(std::string_view sourceText,
                                       const std::vector<uint32_t> &lineOffsets,
                                       const llvm::lsp::Range &range) {
  if (lineOffsets.empty())
    return "";

  // Validate line numbers
  if (static_cast<size_t>(range.start.line) >= lineOffsets.size() ||
      static_cast<size_t>(range.end.line) >= lineOffsets.size())
    return "";

  size_t startOffset = lineOffsets[range.start.line] + range.start.character;
  size_t endOffset = lineOffsets[range.end.line] + range.end.character;

  if (startOffset >= sourceText.size() || endOffset > sourceText.size() ||
      startOffset >= endOffset)
    return "";

  return std::string(sourceText.substr(startOffset, endOffset - startOffset));
}

/// Find a suitable insertion point for a signal declaration in a module.
/// Returns the position just after the port list closing parenthesis and semicolon.
static llvm::lsp::Position findDeclarationInsertionPoint(
    std::string_view sourceText, const std::vector<uint32_t> &lineOffsets,
    const llvm::lsp::Range &diagnosticRange) {

  // Start from the diagnostic position and search backward for );
  // which marks the end of the port list
  if (lineOffsets.empty())
    return llvm::lsp::Position(0, 0);

  // Get the line where the diagnostic is
  int diagLine = diagnosticRange.start.line;

  // Search backwards from the diagnostic line to find ");", which ends the port list
  for (int line = diagLine; line >= 0; --line) {
    size_t lineStart = lineOffsets[line];
    size_t lineEnd = (static_cast<size_t>(line + 1) < lineOffsets.size())
                         ? lineOffsets[line + 1] - 1
                         : sourceText.size();

    std::string_view lineText = sourceText.substr(lineStart, lineEnd - lineStart);

    // Look for ); which typically ends the port list
    size_t pos = lineText.find(");");
    if (pos != std::string_view::npos) {
      // Return position at the start of the next line for clean insertion
      if (static_cast<size_t>(line + 1) < lineOffsets.size()) {
        return llvm::lsp::Position(line + 1, 0);
      }
      // Otherwise insert after the );
      return llvm::lsp::Position(line, static_cast<int>(pos + 2));
    }
  }

  // Fallback: insert at the beginning of the file
  return llvm::lsp::Position(0, 0);
}

void VerilogDocument::getCodeActions(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Range &range,
    const std::vector<llvm::lsp::Diagnostic> &diagnostics,
    std::vector<llvm::lsp::CodeAction> &codeActions) {

  // Get source text for extracting identifiers and finding insertion points
  auto &sm = getSlangSourceManager();
  std::string_view sourceText = sm.getSourceText(mainBufferId);

  // Process diagnostics to generate quick fixes
  for (const auto &diag : diagnostics) {
    // Check for unknown identifier - suggest adding wire/logic declaration
    if (isUnknownIdentifierDiagnostic(diag.message)) {
      // Try to extract identifier from diagnostic message or from source range
      std::optional<std::string> identifierName =
          extractIdentifierFromMessage(diag.message);

      // If we couldn't get it from the message, try extracting from the range
      if (!identifierName) {
        std::string extracted =
            extractTextAtRange(sourceText, lineOffsets, diag.range);
        if (!extracted.empty())
          identifierName = extracted;
      }

      if (identifierName && !identifierName->empty()) {
        // Find insertion point for the declaration
        llvm::lsp::Position insertPos =
            findDeclarationInsertionPoint(sourceText, lineOffsets, diag.range);

        // Create "Declare as wire" action with proper edit
        {
          llvm::lsp::CodeAction wireAction;
          wireAction.title = "Declare '" + *identifierName + "' as wire";
          wireAction.kind = llvm::lsp::CodeAction::kQuickFix;
          wireAction.diagnostics = {diag};
          wireAction.isPreferred = false;

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(insertPos, insertPos);
          textEdit.newText = "  wire " + *identifierName + ";\n";
          edit.changes[uri.uri().str()] = {textEdit};
          wireAction.edit = edit;

          codeActions.push_back(std::move(wireAction));
        }

        // Create "Declare as logic" action with proper edit
        {
          llvm::lsp::CodeAction logicAction;
          logicAction.title = "Declare '" + *identifierName + "' as logic";
          logicAction.kind = llvm::lsp::CodeAction::kQuickFix;
          logicAction.diagnostics = {diag};
          logicAction.isPreferred = true; // Prefer logic in SystemVerilog

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(insertPos, insertPos);
          textEdit.newText = "  logic " + *identifierName + ";\n";
          edit.changes[uri.uri().str()] = {textEdit};
          logicAction.edit = edit;

          codeActions.push_back(std::move(logicAction));
        }

        // Create "Declare as reg" action for Verilog compatibility
        {
          llvm::lsp::CodeAction regAction;
          regAction.title = "Declare '" + *identifierName + "' as reg";
          regAction.kind = llvm::lsp::CodeAction::kQuickFix;
          regAction.diagnostics = {diag};

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(insertPos, insertPos);
          textEdit.newText = "  reg " + *identifierName + ";\n";
          edit.changes[uri.uri().str()] = {textEdit};
          regAction.edit = edit;

          codeActions.push_back(std::move(regAction));
        }
      } else {
        // Fallback: provide generic actions without workspace edits
        llvm::lsp::CodeAction wireAction;
        wireAction.title = "Declare as wire";
        wireAction.kind = llvm::lsp::CodeAction::kQuickFix;
        wireAction.diagnostics = {diag};
        codeActions.push_back(std::move(wireAction));

        llvm::lsp::CodeAction logicAction;
        logicAction.title = "Declare as logic";
        logicAction.kind = llvm::lsp::CodeAction::kQuickFix;
        logicAction.diagnostics = {diag};
        codeActions.push_back(std::move(logicAction));
      }
    }

    // Check for width mismatch - suggest explicit cast
    if (isWidthMismatchDiagnostic(diag.message)) {
      // Extract the expression at the diagnostic range
      std::string expr = extractTextAtRange(sourceText, lineOffsets, diag.range);

      if (!expr.empty()) {
        // Add truncation action
        {
          llvm::lsp::CodeAction truncateAction;
          truncateAction.title = "Add explicit truncation";
          truncateAction.kind = llvm::lsp::CodeAction::kQuickFix;
          truncateAction.diagnostics = {diag};

          // Wrap expression with explicit size cast: expr[N-1:0]
          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = diag.range;
          // Use $bits() function for the cast placeholder
          textEdit.newText = expr + "[/* width-1 */:0]";
          edit.changes[uri.uri().str()] = {textEdit};
          truncateAction.edit = edit;

          codeActions.push_back(std::move(truncateAction));
        }

        // Add zero-extension action
        {
          llvm::lsp::CodeAction extendAction;
          extendAction.title = "Add explicit zero-extension";
          extendAction.kind = llvm::lsp::CodeAction::kQuickFix;
          extendAction.diagnostics = {diag};

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = diag.range;
          textEdit.newText = "{/* padding */," + expr + "}";
          edit.changes[uri.uri().str()] = {textEdit};
          extendAction.edit = edit;

          codeActions.push_back(std::move(extendAction));
        }
      }
    }

    // Check for unknown module - suggest creating module stub
    if (isUnknownModuleDiagnostic(diag.message)) {
      std::optional<std::string> moduleName =
          extractIdentifierFromMessage(diag.message);

      if (moduleName && !moduleName->empty()) {
        llvm::lsp::CodeAction createModuleAction;
        createModuleAction.title = "Create module stub for '" + *moduleName + "'";
        createModuleAction.kind = llvm::lsp::CodeAction::kQuickFix;
        createModuleAction.diagnostics = {diag};

        // Insert module stub at the end of the file
        llvm::lsp::Position endPos(static_cast<int>(lineOffsets.size()), 0);
        std::string moduleStub = "\n// TODO: Implement module\n"
                                 "module " + *moduleName + " (\n"
                                 "  // Add ports here\n"
                                 ");\n"
                                 "  // Add implementation here\n"
                                 "endmodule\n";

        llvm::lsp::WorkspaceEdit edit;
        llvm::lsp::TextEdit textEdit;
        textEdit.range = llvm::lsp::Range(endPos, endPos);
        textEdit.newText = moduleStub;
        edit.changes[uri.uri().str()] = {textEdit};
        createModuleAction.edit = edit;

        codeActions.push_back(std::move(createModuleAction));
      }
    }

    // Check for missing import or UVM type errors
    if (isMissingImportDiagnostic(diag.message) || mentionsUVMType(diag.message)) {
      std::optional<std::string> typeName =
          extractIdentifierFromMessage(diag.message);

      // Find the best position for import insertion
      llvm::lsp::Position importPos = findImportInsertionPoint(sourceText, lineOffsets);

      // Check if this is a UVM-related error
      bool isUVMError = mentionsUVMType(diag.message) ||
                        (typeName && isUVMTypeName(*typeName));

      if (isUVMError) {
        // Add UVM-specific import as the preferred action
        {
          llvm::lsp::CodeAction uvmImportAction;
          uvmImportAction.title = "Add 'import uvm_pkg::*;'";
          uvmImportAction.kind = llvm::lsp::CodeAction::kQuickFix;
          uvmImportAction.diagnostics = {diag};
          uvmImportAction.isPreferred = true; // Mark as preferred for UVM errors

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(importPos, importPos);
          textEdit.newText = "import uvm_pkg::*;\n";
          edit.changes[uri.uri().str()] = {textEdit};
          uvmImportAction.edit = edit;

          codeActions.push_back(std::move(uvmImportAction));
        }

        // Also suggest adding UVM macros include
        {
          llvm::lsp::CodeAction includeAction;
          includeAction.title = "Add '`include \"uvm_macros.svh\"'";
          includeAction.kind = llvm::lsp::CodeAction::kQuickFix;
          includeAction.diagnostics = {diag};

          // Insert include at beginning of file
          llvm::lsp::Position includePos(0, 0);

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(includePos, includePos);
          textEdit.newText = "`include \"uvm_macros.svh\"\n";
          edit.changes[uri.uri().str()] = {textEdit};
          includeAction.edit = edit;

          codeActions.push_back(std::move(includeAction));
        }

        // Suggest combined UVM setup (include + import)
        {
          llvm::lsp::CodeAction combinedAction;
          combinedAction.title = "Add UVM boilerplate (include + import)";
          combinedAction.kind = llvm::lsp::CodeAction::kQuickFix;
          combinedAction.diagnostics = {diag};

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(llvm::lsp::Position(0, 0),
                                            llvm::lsp::Position(0, 0));
          textEdit.newText = "`include \"uvm_macros.svh\"\nimport uvm_pkg::*;\n\n";
          edit.changes[uri.uri().str()] = {textEdit};
          combinedAction.edit = edit;

          codeActions.push_back(std::move(combinedAction));
        }
      } else if (typeName && !typeName->empty()) {
        // For non-UVM errors, suggest common packages
        std::vector<std::string> commonPackages = {"std"};

        for (const auto &pkg : commonPackages) {
          llvm::lsp::CodeAction importAction;
          importAction.title = "Add 'import " + pkg + "::*;'";
          importAction.kind = llvm::lsp::CodeAction::kQuickFix;
          importAction.diagnostics = {diag};

          llvm::lsp::WorkspaceEdit edit;
          llvm::lsp::TextEdit textEdit;
          textEdit.range = llvm::lsp::Range(importPos, importPos);
          textEdit.newText = "import " + pkg + "::*;\n";
          edit.changes[uri.uri().str()] = {textEdit};
          importAction.edit = edit;

          codeActions.push_back(std::move(importAction));
        }
      }
    }

    // Check for missing semicolon - suggest inserting semicolon
    if (isMissingSemicolonDiagnostic(diag.message)) {
      llvm::lsp::CodeAction semicolonAction;
      semicolonAction.title = "Insert missing semicolon";
      semicolonAction.kind = llvm::lsp::CodeAction::kQuickFix;
      semicolonAction.diagnostics = {diag};
      semicolonAction.isPreferred = true; // This is usually the correct fix

      llvm::lsp::WorkspaceEdit edit;
      llvm::lsp::TextEdit textEdit;
      // Insert semicolon at the end of the diagnostic range
      textEdit.range = llvm::lsp::Range(diag.range.end, diag.range.end);
      textEdit.newText = ";";
      edit.changes[uri.uri().str()] = {textEdit};
      semicolonAction.edit = edit;

      codeActions.push_back(std::move(semicolonAction));
    }

    // Check for possible typos and suggest corrections
    if (isPossibleTypoDiagnostic(diag.message)) {
      std::string sourceAtRange =
          extractTextAtRange(sourceText, lineOffsets, diag.range);
      auto typoCorrection =
          getTypoAndCorrection(diag.message, sourceAtRange);

      if (typoCorrection) {
        llvm::lsp::CodeAction typoAction;
        typoAction.title =
            "Fix typo: '" + typoCorrection->first + "' -> '" +
            typoCorrection->second + "'";
        typoAction.kind = llvm::lsp::CodeAction::kQuickFix;
        typoAction.diagnostics = {diag};
        typoAction.isPreferred = true; // Typo fixes are usually correct

        llvm::lsp::WorkspaceEdit edit;
        llvm::lsp::TextEdit textEdit;
        textEdit.range = diag.range;
        textEdit.newText = typoCorrection->second;
        edit.changes[uri.uri().str()] = {textEdit};
        typoAction.edit = edit;

        codeActions.push_back(std::move(typoAction));
      }
    }

    // Check for begin/end block needed - suggest wrapping in begin/end
    if (isBeginEndNeededDiagnostic(diag.message)) {
      // Get the statement text that needs to be wrapped
      std::string stmtText =
          extractTextAtRange(sourceText, lineOffsets, diag.range);

      if (!stmtText.empty()) {
        llvm::lsp::CodeAction beginEndAction;
        beginEndAction.title = "Wrap in begin/end block";
        beginEndAction.kind = llvm::lsp::CodeAction::kQuickFix;
        beginEndAction.diagnostics = {diag};

        llvm::lsp::WorkspaceEdit edit;
        llvm::lsp::TextEdit textEdit;
        textEdit.range = diag.range;
        // Preserve indentation by adding proper formatting
        textEdit.newText = "begin\n    " + stmtText + "\n  end";
        edit.changes[uri.uri().str()] = {textEdit};
        beginEndAction.edit = edit;

        codeActions.push_back(std::move(beginEndAction));
      }
    }
  }

  // Add refactoring actions that are always available when there's a selection
  if (range.start != range.end) {
    std::string selectedText = extractTextAtRange(sourceText, lineOffsets, range);

    if (!selectedText.empty() && selectedText.find('\n') == std::string::npos) {
      // Extract selection as new signal
      llvm::lsp::CodeAction extractSignalAction;
      extractSignalAction.title = "Extract to signal";
      extractSignalAction.kind = llvm::lsp::CodeAction::kRefactor;

      // Find insertion point
      llvm::lsp::Position insertPos =
          findDeclarationInsertionPoint(sourceText, lineOffsets, range);

      // Create the edit: declare the signal and replace selection with signal name
      llvm::lsp::WorkspaceEdit edit;

      // Add declaration
      llvm::lsp::TextEdit declEdit;
      declEdit.range = llvm::lsp::Range(insertPos, insertPos);
      declEdit.newText = "  logic extracted_signal; // = " + selectedText + "\n";

      // Replace selection with signal name
      llvm::lsp::TextEdit replaceEdit;
      replaceEdit.range = range;
      replaceEdit.newText = "extracted_signal";

      edit.changes[uri.uri().str()] = {declEdit, replaceEdit};
      extractSignalAction.edit = edit;

      codeActions.push_back(std::move(extractSignalAction));
    }
  }

  // Add module instantiation template action (only if compilation succeeded)
  if (succeeded(compilation)) {
    for (const auto *def : (*compilation)->getDefinitions()) {
      llvm::lsp::CodeAction instantiateAction;
      instantiateAction.title = "Insert " + std::string(def->name) + " instance";
      instantiateAction.kind = llvm::lsp::CodeAction::kRefactor;

      // Build the instantiation template
      std::string instTemplate = std::string(def->name) + " inst_" +
                                 std::string(def->name) + " (\n";

      // Get the ports from the definition
      bool first = true;
      // Try to get ports from the first instance if available
      const auto &root = (*compilation)->getRoot();
      for (auto *inst : root.topInstances) {
        if (&inst->getDefinition() == def) {
          for (const auto *portSym : inst->body.getPortList()) {
            if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
              if (!first)
                instTemplate += ",\n";
              first = false;
              instTemplate += "  ." + std::string(port->name) + "()";
            }
          }
          break;
        }
      }

      instTemplate += "\n);";

      llvm::lsp::WorkspaceEdit edit;
      llvm::lsp::TextEdit textEdit;
      textEdit.range = range;
      textEdit.newText = instTemplate;
      edit.changes[uri.uri().str()] = {textEdit};
      instantiateAction.edit = edit;

      codeActions.push_back(std::move(instantiateAction));
    }
  }
}

//===----------------------------------------------------------------------===//
// Rename Symbol
//===----------------------------------------------------------------------===//

std::optional<std::pair<llvm::lsp::Range, std::string>>
VerilogDocument::prepareRename(const llvm::lsp::URIForFile &uri,
                               const llvm::lsp::Position &pos) {
  if (!index)
    return std::nullopt;

  const auto *slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return std::nullopt;

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  // Found no element at the given position.
  if (!it.valid() || slangBufferPointer < it.start())
    return std::nullopt;

  auto element = it.value();

  // We can only rename Verilog symbols, not attributes
  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(element);
  if (!symbol || symbol->name.empty())
    return std::nullopt;

  // Check if this is a renameable symbol type
  switch (symbol->kind) {
  case slang::ast::SymbolKind::Variable:
  case slang::ast::SymbolKind::Net:
  case slang::ast::SymbolKind::Port:
  case slang::ast::SymbolKind::Parameter:
  case slang::ast::SymbolKind::Definition:
  case slang::ast::SymbolKind::Instance:
  case slang::ast::SymbolKind::Subroutine:
  case slang::ast::SymbolKind::ClassType:
  case slang::ast::SymbolKind::ClassProperty:
  case slang::ast::SymbolKind::InterfacePort:
  case slang::ast::SymbolKind::Modport:
  case slang::ast::SymbolKind::FormalArgument:
  case slang::ast::SymbolKind::TypeAlias:
    break;
  default:
    return std::nullopt; // Not a renameable symbol
  }

  // Calculate the range of the symbol at the cursor
  const auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  size_t startOffset = it.start() - text.data();
  size_t endOffset = it.stop() - text.data();

  // Convert offsets to LSP positions
  int startLine = 0, startChar = 0;
  int endLine = 0, endChar = 0;

  for (size_t i = 0; i < lineOffsets.size(); ++i) {
    if (lineOffsets[i] <= startOffset) {
      startLine = i;
      startChar = startOffset - lineOffsets[i];
    }
    if (lineOffsets[i] <= endOffset) {
      endLine = i;
      endChar = endOffset - lineOffsets[i];
    }
  }

  llvm::lsp::Range range(llvm::lsp::Position(startLine, startChar),
                         llvm::lsp::Position(endLine, endChar));

  return std::make_pair(range, std::string(symbol->name));
}

std::optional<llvm::lsp::WorkspaceEdit>
VerilogDocument::renameSymbol(const llvm::lsp::URIForFile &uri,
                              const llvm::lsp::Position &pos,
                              llvm::StringRef newName) {
  if (!index)
    return std::nullopt;

  const auto *slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return std::nullopt;

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  // Found no element at the given position.
  if (!it.valid() || slangBufferPointer < it.start())
    return std::nullopt;

  auto element = it.value();

  // We can only rename Verilog symbols, not attributes
  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(element);
  if (!symbol || symbol->name.empty())
    return std::nullopt;

  // Validate the new name
  if (newName.empty())
    return std::nullopt;

  // Check if new name is a valid identifier
  if (!std::isalpha(newName[0]) && newName[0] != '_')
    return std::nullopt;
  for (char c : newName) {
    if (!std::isalnum(c) && c != '_' && c != '$')
      return std::nullopt;
  }

  llvm::lsp::WorkspaceEdit edit;
  std::vector<llvm::lsp::TextEdit> textEdits;

  // Get the definition location
  slang::SourceRange defRange(symbol->location,
                              symbol->location + symbol->name.size());

  // Add edit for the definition
  if (defRange.start().buffer() == mainBufferId) {
    llvm::lsp::TextEdit defEdit;
    defEdit.range = getLspRange(defRange);
    defEdit.newText = newName.str();
    textEdits.push_back(std::move(defEdit));
  }

  // Add edits for all references
  auto refIt = index->getReferences().find(symbol);
  if (refIt != index->getReferences().end()) {
    for (const auto &refRange : refIt->second) {
      if (refRange.start().buffer() == mainBufferId) {
        llvm::lsp::TextEdit refEdit;
        refEdit.range = getLspRange(refRange);
        refEdit.newText = newName.str();
        textEdits.push_back(std::move(refEdit));
      }
    }
  }

  if (textEdits.empty())
    return std::nullopt;

  // Sort edits by position (descending) to avoid offset issues when applying
  std::sort(textEdits.begin(), textEdits.end(),
            [](const llvm::lsp::TextEdit &a, const llvm::lsp::TextEdit &b) {
              // Compare positions: a > b if a comes after b
              if (a.range.start.line != b.range.start.line)
                return a.range.start.line > b.range.start.line;
              return a.range.start.character > b.range.start.character;
            });

  // Remove duplicates
  textEdits.erase(std::unique(textEdits.begin(), textEdits.end()),
                  textEdits.end());

  edit.changes[uri.uri().str()] = std::move(textEdits);
  return edit;
}

//===----------------------------------------------------------------------===//
// Document Links
//===----------------------------------------------------------------------===//

void VerilogDocument::getDocumentLinks(
    const llvm::lsp::URIForFile &uri,
    std::vector<llvm::lsp::DocumentLink> &links) {
  if (!index)
    return;

  const auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  const auto &includeMap = index->getIncludes();
  for (const auto &[offsetRange, path] : includeMap) {
    // Convert offsets to LSP range
    int startLine = 0, startChar = 0;
    int endLine = 0, endChar = 0;

    for (size_t i = 0; i < lineOffsets.size(); ++i) {
      if (lineOffsets[i] <= offsetRange.first) {
        startLine = i;
        startChar = offsetRange.first - lineOffsets[i];
      }
      if (lineOffsets[i] <= offsetRange.second) {
        endLine = i;
        endChar = offsetRange.second - lineOffsets[i];
      }
    }

    llvm::lsp::Range range(llvm::lsp::Position(startLine, startChar),
                           llvm::lsp::Position(endLine, endChar));

    // Resolve the file path
    llvm::SmallString<256> absPath(path);
    if (!llvm::sys::path::is_absolute(absPath)) {
      // Try to resolve relative to the document's directory
      llvm::SmallString<256> docDir(uri.file());
      llvm::sys::path::remove_filename(docDir);
      llvm::sys::path::append(docDir, path);
      if (llvm::sys::fs::exists(docDir))
        absPath = docDir;
      else {
        // Try libDirs
        for (const auto &libDir : globalContext.options.libDirs) {
          llvm::SmallString<256> libPath(libDir);
          llvm::sys::path::append(libPath, path);
          if (llvm::sys::fs::exists(libPath)) {
            absPath = libPath;
            break;
          }
        }
      }
    }

    if (llvm::sys::fs::exists(absPath)) {
      auto uriOrErr = llvm::lsp::URIForFile::fromFile(absPath);
      if (uriOrErr) {
        links.emplace_back(range, std::move(*uriOrErr));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Semantic Tokens
//===----------------------------------------------------------------------===//

namespace {

/// Map slang symbol kind to semantic token type.
SemanticTokenType mapToSemanticTokenType(slang::ast::SymbolKind kind) {
  using SK = slang::ast::SymbolKind;
  switch (kind) {
  case SK::Package:
    return SemanticTokenType::Namespace;
  case SK::Definition:
    return SemanticTokenType::Module;
  case SK::Instance:
    return SemanticTokenType::Instance;
  case SK::Net:
    return SemanticTokenType::Net;
  case SK::Variable:
    return SemanticTokenType::Variable;
  case SK::Parameter:
    return SemanticTokenType::Parameter;
  case SK::Port:
    return SemanticTokenType::Port;
  case SK::Subroutine:
    return SemanticTokenType::Function;
  case SK::ClassType:
    return SemanticTokenType::Class;
  case SK::InterfacePort:
    return SemanticTokenType::Interface;
  case SK::EnumType:
  case SK::TypeAlias:
    return SemanticTokenType::Enum;
  case SK::EnumValue:
    return SemanticTokenType::EnumMember;
  case SK::PackedStructType:
  case SK::UnpackedStructType:
    return SemanticTokenType::Struct;
  default:
    return SemanticTokenType::Variable;
  }
}

/// Check if a token kind is an operator.
bool isOperatorToken(slang::parsing::TokenKind kind) {
  using TK = slang::parsing::TokenKind;
  switch (kind) {
  case TK::Plus:
  case TK::DoublePlus:
  case TK::Minus:
  case TK::DoubleMinus:
  case TK::Star:
  case TK::DoubleStar:
  case TK::Slash:
  case TK::Percent:
  case TK::Equals:
  case TK::DoubleEquals:
  case TK::TripleEquals:
  case TK::ExclamationEquals:
  case TK::ExclamationDoubleEquals:
  case TK::LessThan:
  case TK::LessThanEquals:
  case TK::GreaterThan:
  case TK::GreaterThanEquals:
  case TK::And:
  case TK::DoubleAnd:
  case TK::Or:
  case TK::DoubleOr:
  case TK::Xor:
  case TK::Tilde:
  case TK::TildeAnd:
  case TK::TildeOr:
  case TK::TildeXor:
  case TK::LeftShift:
  case TK::RightShift:
  case TK::TripleLeftShift:
  case TK::TripleRightShift:
  case TK::Question:
  case TK::Exclamation:
  case TK::PlusEqual:
  case TK::MinusEqual:
  case TK::StarEqual:
  case TK::SlashEqual:
  case TK::PercentEqual:
  case TK::AndEqual:
  case TK::OrEqual:
  case TK::XorEqual:
  case TK::LeftShiftEqual:
  case TK::RightShiftEqual:
  case TK::MinusArrow:
  case TK::EqualsArrow:
    return true;
  default:
    return false;
  }
}

/// Visitor to collect semantic tokens from the AST.
class SemanticTokenVisitor
    : public slang::ast::ASTVisitor<SemanticTokenVisitor, true, true> {
public:
  SemanticTokenVisitor(std::vector<SemanticToken> &tokens,
                       slang::BufferID bufferId,
                       const slang::SourceManager &sm,
                       const std::vector<uint32_t> &lineOffsets)
      : tokens(tokens), bufferId(bufferId), sm(sm), lineOffsets(lineOffsets) {}

  std::vector<SemanticToken> &tokens;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
  const std::vector<uint32_t> &lineOffsets;

  void addToken(slang::SourceLocation loc, size_t length,
                SemanticTokenType type, uint32_t modifiers = 0) {
    if (!loc.valid() || loc.buffer() != bufferId || length == 0)
      return;

    uint32_t line = slangLineToLsp(sm.getLineNumber(loc));
    uint32_t col = slangColumnToLsp(sm.getColumnNumber(loc));

    tokens.emplace_back(line, col, static_cast<uint32_t>(length),
                        type, modifiers);
  }

  void handle(const slang::ast::InstanceBodySymbol &body) {
    // Add token for the module/interface name
    if (body.location.valid() && body.location.buffer() == bufferId &&
        !body.name.empty()) {
      addToken(body.location, body.name.size(), SemanticTokenType::Module,
               static_cast<uint32_t>(SemanticTokenModifier::Definition));
    }

    // Process ports
    for (const auto *portSym : body.getPortList()) {
      if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
        if (port->location.valid() && port->location.buffer() == bufferId &&
            !port->name.empty()) {
          addToken(port->location, port->name.size(), SemanticTokenType::Port,
                   static_cast<uint32_t>(SemanticTokenModifier::Declaration));
        }
      }
    }

    // Process members
    for (const auto &member : body.members()) {
      if (!member.location.valid() || member.location.buffer() != bufferId ||
          member.name.empty())
        continue;

      SemanticTokenType type = mapToSemanticTokenType(member.kind);
      uint32_t modifiers = static_cast<uint32_t>(SemanticTokenModifier::Declaration);

      // Add readonly modifier for parameters
      if (member.kind == slang::ast::SymbolKind::Parameter)
        modifiers |= static_cast<uint32_t>(SemanticTokenModifier::Readonly);

      addToken(member.location, member.name.size(), type, modifiers);
    }

    visitDefault(body);
  }

  void handle(const slang::ast::PackageSymbol &pkg) {
    if (pkg.location.valid() && pkg.location.buffer() == bufferId &&
        !pkg.name.empty()) {
      addToken(pkg.location, pkg.name.size(), SemanticTokenType::Namespace,
               static_cast<uint32_t>(SemanticTokenModifier::Definition));
    }

    // Process package members
    for (const auto &member : pkg.members()) {
      if (!member.location.valid() || member.location.buffer() != bufferId ||
          member.name.empty())
        continue;

      SemanticTokenType type = mapToSemanticTokenType(member.kind);
      uint32_t modifiers = static_cast<uint32_t>(SemanticTokenModifier::Declaration);

      addToken(member.location, member.name.size(), type, modifiers);
    }
  }

  void handle(const slang::ast::NamedValueExpression &expr) {
    // Token for symbol references
    if (expr.sourceRange.start().buffer() == bufferId) {
      SemanticTokenType type = mapToSemanticTokenType(expr.symbol.kind);
      uint32_t length = std::max(1, static_cast<int>(expr.symbol.name.size()));
      addToken(expr.sourceRange.start(), length, type, 0);
    }
    visitDefault(expr);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

/// Visitor to collect lexer-level tokens (keywords, literals, comments) from
/// syntax tree.
class SyntaxTokenCollector
    : public slang::syntax::SyntaxVisitor<SyntaxTokenCollector> {
public:
  SyntaxTokenCollector(std::vector<SemanticToken> &tokens,
                       slang::BufferID bufferId,
                       const slang::SourceManager &sm)
      : tokens(tokens), bufferId(bufferId), sm(sm) {}

  std::vector<SemanticToken> &tokens;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;

  void addToken(slang::SourceLocation loc, size_t length,
                SemanticTokenType type, uint32_t modifiers = 0) {
    if (!loc.valid() || loc.buffer() != bufferId || length == 0)
      return;

    uint32_t line = slangLineToLsp(sm.getLineNumber(loc));
    uint32_t col = slangColumnToLsp(sm.getColumnNumber(loc));

    tokens.emplace_back(line, col, static_cast<uint32_t>(length),
                        type, modifiers);
  }

  void visitToken(slang::parsing::Token token) {
    if (!token.valid())
      return;

    // Process trivia (comments)
    for (const auto &trivia : token.trivia()) {
      processTrivia(trivia, token.location());
    }

    // Process the token itself
    processLexerToken(token);
  }

  void processTrivia(const slang::parsing::Trivia &trivia,
                     slang::SourceLocation tokenLoc) {
    using TK = slang::parsing::TriviaKind;

    switch (trivia.kind) {
    case TK::LineComment:
    case TK::BlockComment: {
      auto explicitLoc = trivia.getExplicitLocation();
      std::string_view text = trivia.getRawText();
      if (!text.empty()) {
        // For trivia without explicit location, we'd need to compute it
        // from the token location. For now, skip if no explicit location.
        if (explicitLoc) {
          addToken(*explicitLoc, text.size(), SemanticTokenType::Comment);
        }
      }
      break;
    }
    default:
      break;
    }
  }

  void processLexerToken(slang::parsing::Token token) {
    using TK = slang::parsing::TokenKind;

    slang::SourceLocation loc = token.location();
    std::string_view rawText = token.rawText();

    if (!loc.valid() || loc.buffer() != bufferId || rawText.empty())
      return;

    TK kind = token.kind;

    // Keywords
    if (slang::parsing::LexerFacts::isKeyword(kind)) {
      addToken(loc, rawText.size(), SemanticTokenType::Keyword);
      return;
    }

    // Literals
    switch (kind) {
    case TK::StringLiteral:
      addToken(loc, rawText.size(), SemanticTokenType::String);
      return;
    case TK::IntegerLiteral:
    case TK::IntegerBase:
    case TK::UnbasedUnsizedLiteral:
    case TK::RealLiteral:
    case TK::TimeLiteral:
      addToken(loc, rawText.size(), SemanticTokenType::Number);
      return;
    default:
      break;
    }

    // Operators
    if (isOperatorToken(kind)) {
      addToken(loc, rawText.size(), SemanticTokenType::Operator);
      return;
    }
  }

  void handle(const slang::syntax::SyntaxNode &node) {
    // Visit all children
    for (size_t i = 0; i < node.getChildCount(); i++) {
      auto child = node.childNode(i);
      if (child) {
        visit(*child);
      } else {
        // It's a token
        auto token = node.childToken(i);
        if (token.valid()) {
          visitToken(token);
        }
      }
    }
  }
};

} // namespace

void VerilogDocument::getSemanticTokens(
    const llvm::lsp::URIForFile &uri,
    std::vector<SemanticToken> &tokens) {
  if (failed(compilation))
    return;

  // First, collect lexer-level tokens (keywords, literals, operators)
  // from the syntax trees
  SyntaxTokenCollector syntaxCollector(tokens, mainBufferId,
                                       getSlangSourceManager());
  for (const auto &tree : driver.syntaxTrees) {
    syntaxCollector.visit(tree->root());
  }

  // Then, collect AST-level tokens (symbol declarations and references)
  // These will provide more accurate semantic information for identifiers
  SemanticTokenVisitor visitor(tokens, mainBufferId, getSlangSourceManager(),
                               lineOffsets);

  // Visit packages
  for (auto *package : (*compilation)->getPackages()) {
    if (package->location.buffer() == mainBufferId)
      visitor.handle(*package);
  }

  // Visit top instances
  for (auto *inst : (*compilation)->getRoot().topInstances) {
    if (inst->body.location.buffer() == mainBufferId)
      inst->body.visit(visitor);
  }
}

//===----------------------------------------------------------------------===//
// Inlay Hints
//===----------------------------------------------------------------------===//

namespace {

/// Visitor to collect inlay hints from the AST.
class InlayHintVisitor
    : public slang::ast::ASTVisitor<InlayHintVisitor, true, true> {
public:
  InlayHintVisitor(std::vector<llvm::lsp::InlayHint> &hints,
                   slang::BufferID bufferId,
                   const slang::SourceManager &sm,
                   const llvm::lsp::Range &range)
      : hints(hints), bufferId(bufferId), sm(sm), range(range) {}

  std::vector<llvm::lsp::InlayHint> &hints;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
  const llvm::lsp::Range &range;

  bool isInRange(slang::SourceLocation loc) const {
    if (!loc.valid() || loc.buffer() != bufferId)
      return false;

    int line = slangLineToLsp(sm.getLineNumber(loc));
    int col = slangColumnToLsp(sm.getColumnNumber(loc));

    if (line < range.start.line || line > range.end.line)
      return false;
    if (line == range.start.line && col < range.start.character)
      return false;
    if (line == range.end.line && col > range.end.character)
      return false;

    return true;
  }

  llvm::lsp::Position getPosition(slang::SourceLocation loc) const {
    return llvm::lsp::Position(slangLineToLsp(sm.getLineNumber(loc)),
                                slangColumnToLsp(sm.getColumnNumber(loc)));
  }

  void handle(const slang::ast::InstanceBodySymbol &body) {
    // Add width hints for variables/nets
    for (const auto &member : body.members()) {
      if (!isInRange(member.location))
        continue;

      if (auto *var = member.as_if<slang::ast::VariableSymbol>()) {
        if (var->getType().isIntegral()) {
          auto width = var->getType().getBitWidth();
          if (width > 1) {
            // Add hint showing the width after the variable name
            auto pos = getPosition(member.location);
            pos.character += member.name.size();

            llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Type, pos);
            hint.label = ": " + std::to_string(width) + "-bit";
            hint.paddingLeft = true;
            hints.push_back(std::move(hint));
          }
        }
      } else if (auto *net = member.as_if<slang::ast::NetSymbol>()) {
        if (net->getType().isIntegral()) {
          auto width = net->getType().getBitWidth();
          if (width > 1) {
            auto pos = getPosition(member.location);
            pos.character += member.name.size();

            llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Type, pos);
            hint.label = ": " + std::to_string(width) + "-bit";
            hint.paddingLeft = true;
            hints.push_back(std::move(hint));
          }
        }
      } else if (auto *param = member.as_if<slang::ast::ParameterSymbol>()) {
        // Show parameter values
        auto initExpr = param->getInitializer();
        if (initExpr) {
          slang::ast::EvalContext evalCtx(member);
          auto cv = initExpr->eval(evalCtx);
          if (cv) {
            auto pos = getPosition(member.location);
            pos.character += member.name.size();

            llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Type, pos);
            hint.label = " = " + std::string(cv.toString());
            hint.paddingLeft = true;
            hints.push_back(std::move(hint));
          }
        }
      }
    }

    visitDefault(body);
  }

  void handle(const slang::ast::InstanceSymbol &inst) {
    // Add hints for port connections in module instantiation
    if (!isInRange(inst.location))
      return;

    // Show the module type being instantiated
    {
      auto pos = getPosition(inst.location);
      pos.character += inst.name.size();

      llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Type, pos);
      hint.label = " /* " + std::string(inst.getDefinition().name) + " */";
      hint.paddingLeft = true;
      hints.push_back(std::move(hint));
    }

    // Add port name hints for positional port connections
    // In Verilog, an instantiation uses either ALL ordered or ALL named connections
    // We determine the style by checking the first connection that has valid syntax
    auto connections = inst.getPortConnections();

    // First, determine if this instantiation uses ordered connections
    bool usesOrderedConnections = false;
    for (const auto *conn : connections) {
      if (!conn)
        continue;
      const auto *expr = conn->getExpression();
      if (!expr || !expr->syntax)
        continue;

      // Walk up to find what kind of port connection this is
      const slang::syntax::SyntaxNode *syntaxNode = expr->syntax;
      int depth = 0;
      while (syntaxNode && depth < 20) {
        if (syntaxNode->kind == slang::syntax::SyntaxKind::OrderedPortConnection) {
          usesOrderedConnections = true;
          break;
        }
        if (syntaxNode->kind == slang::syntax::SyntaxKind::NamedPortConnection) {
          // Named connection found, not ordered
          break;
        }
        if (syntaxNode->kind == slang::syntax::SyntaxKind::HierarchyInstantiation) {
          break;
        }
        syntaxNode = syntaxNode->parent;
        ++depth;
      }
      // Once we've determined the style from the first valid connection, we're done
      break;
    }

    // Only add hints if the instantiation uses ordered connections
    if (usesOrderedConnections) {
      for (const auto *conn : connections) {
        if (!conn)
          continue;

        const auto *expr = conn->getExpression();
        if (!expr)
          continue;

        // Check if the expression location is in range
        auto exprLoc = expr->sourceRange.start();
        if (!isInRange(exprLoc))
          continue;

        // Get port name from the connection
        std::string_view portName = conn->port.name;
        if (portName.empty())
          continue;

        auto pos = getPosition(exprLoc);
        llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Parameter, pos);
        hint.label = std::string(portName) + ":";
        hint.paddingRight = true;
        hints.push_back(std::move(hint));
      }
    }

    visitDefault(inst);
  }

  void handle(const slang::ast::CallExpression &call) {
    // Add parameter name hints for function/task calls
    if (call.isSystemCall()) {
      visitDefault(call);
      return;
    }

    const auto *sub =
        std::get_if<const slang::ast::SubroutineSymbol *>(&call.subroutine);
    if (!sub || !*sub) {
      visitDefault(call);
      return;
    }

    const auto &subroutine = **sub;
    auto formalArgs = subroutine.getArguments();
    auto actualArgs = call.arguments();

    // Match actual arguments to formal parameters
    size_t argIndex = 0;
    for (const auto *argExpr : actualArgs) {
      if (!argExpr || argIndex >= formalArgs.size())
        break;

      auto exprLoc = argExpr->sourceRange.start();
      if (!isInRange(exprLoc)) {
        ++argIndex;
        continue;
      }

      const auto *formalArg = formalArgs[argIndex];
      if (!formalArg || formalArg->name.empty()) {
        ++argIndex;
        continue;
      }

      // Check if this is a named argument by looking at the syntax
      bool isNamedArg = false;
      if (call.syntax) {
        // Check if this call has InvocationExpressionSyntax
        if (auto *invocation =
                call.syntax->as_if<slang::syntax::InvocationExpressionSyntax>()) {
          if (invocation->arguments) {
            // Check if the argument at this index is named
            size_t syntaxIdx = 0;
            for (auto *argSyntax : invocation->arguments->parameters) {
              if (syntaxIdx == argIndex) {
                if (argSyntax->kind == slang::syntax::SyntaxKind::NamedArgument) {
                  isNamedArg = true;
                }
                break;
              }
              ++syntaxIdx;
            }
          }
        }
      }

      // Only add hint for positional arguments
      if (!isNamedArg) {
        auto pos = getPosition(exprLoc);
        llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Parameter, pos);
        hint.label = std::string(formalArg->name) + ":";
        hint.paddingRight = true;
        hints.push_back(std::move(hint));
      }

      ++argIndex;
    }

    visitDefault(call);
  }

  void handle(const slang::ast::SubroutineSymbol &sub) {
    // Add return type hints for functions without explicit return type annotation
    if (!isInRange(sub.location)) {
      visitDefault(sub);
      return;
    }

    // Only add hint for functions (not tasks)
    if (sub.subroutineKind != slang::ast::SubroutineKind::Function) {
      visitDefault(sub);
      return;
    }

    // Check if the function has a non-void return type
    const auto &returnType = sub.getReturnType();
    if (returnType.isVoid()) {
      visitDefault(sub);
      return;
    }

    // Get the position after the function name (before parameters)
    auto pos = getPosition(sub.location);
    pos.character += sub.name.size();

    llvm::lsp::InlayHint hint(llvm::lsp::InlayHintKind::Type, pos);
    hint.label = " -> " + formatTypeDescription(returnType);
    hint.paddingLeft = true;
    hints.push_back(std::move(hint));

    visitDefault(sub);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

} // namespace

void VerilogDocument::getInlayHints(
    const llvm::lsp::URIForFile &uri,
    const llvm::lsp::Range &range,
    std::vector<llvm::lsp::InlayHint> &hints) {
  if (failed(compilation))
    return;

  InlayHintVisitor visitor(hints, mainBufferId, getSlangSourceManager(), range);

  // Visit top instances
  for (auto *inst : (*compilation)->getRoot().topInstances) {
    if (inst->body.location.buffer() == mainBufferId)
      inst->body.visit(visitor);
  }
}

//===----------------------------------------------------------------------===//
// Signature Help
//===----------------------------------------------------------------------===//

/// Find the function/task call context at the given position.
/// Returns the subroutine symbol and the active parameter index if found.
static std::pair<const slang::ast::SubroutineSymbol *, int>
findCallContext(std::string_view text, uint32_t offset,
                slang::ast::Compilation &compilation,
                slang::BufferID mainBufferId) {
  // Find the opening parenthesis before the cursor
  int parenDepth = 0;
  int activeParam = 0;
  int callStart = -1;

  // Scan backwards from cursor to find the function call start
  for (int i = offset - 1; i >= 0; --i) {
    char c = text[i];
    if (c == ')') {
      ++parenDepth;
    } else if (c == '(') {
      if (parenDepth == 0) {
        callStart = i;
        break;
      }
      --parenDepth;
    } else if (c == ',' && parenDepth == 0) {
      ++activeParam;
    }
  }

  if (callStart < 0)
    return {nullptr, 0};

  // Find the function name before the opening parenthesis
  int nameEnd = callStart;
  while (nameEnd > 0 && std::isspace(text[nameEnd - 1]))
    --nameEnd;

  int nameStart = nameEnd;
  while (nameStart > 0 &&
         (std::isalnum(text[nameStart - 1]) || text[nameStart - 1] == '_' ||
          text[nameStart - 1] == '$'))
    --nameStart;

  if (nameStart >= nameEnd)
    return {nullptr, 0};

  std::string_view funcName = text.substr(nameStart, nameEnd - nameStart);

  // Search for the function/task in the compilation
  const slang::ast::SubroutineSymbol *foundSub = nullptr;

  // Search in packages
  for (auto *package : compilation.getPackages()) {
    for (const auto &member : package->members()) {
      if (member.kind == slang::ast::SymbolKind::Subroutine &&
          member.name == funcName) {
        foundSub = &member.as<slang::ast::SubroutineSymbol>();
        break;
      }
    }
    if (foundSub)
      break;
  }

  // Search in top instances
  if (!foundSub) {
    for (auto *inst : compilation.getRoot().topInstances) {
      for (const auto &member : inst->body.members()) {
        if (member.kind == slang::ast::SymbolKind::Subroutine &&
            member.name == funcName) {
          foundSub = &member.as<slang::ast::SubroutineSymbol>();
          break;
        }
      }
      if (foundSub)
        break;

      // Also search in nested scopes (classes)
      for (const auto &member : inst->body.members()) {
        if (member.kind == slang::ast::SymbolKind::ClassType) {
          const auto &classType = member.as<slang::ast::ClassType>();
          for (const auto &classMember : classType.members()) {
            if (classMember.kind == slang::ast::SymbolKind::Subroutine &&
                classMember.name == funcName) {
              foundSub = &classMember.as<slang::ast::SubroutineSymbol>();
              break;
            }
          }
        }
        if (foundSub)
          break;
      }
      if (foundSub)
        break;
    }
  }

  return {foundSub, activeParam};
}

/// Format a subroutine signature for display.
static std::string formatSubroutineSignature(
    const slang::ast::SubroutineSymbol &sub) {
  std::string result;
  llvm::raw_string_ostream os(result);

  if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
    os << "function ";
  else
    os << "task ";

  os << formatTypeDescription(sub.getReturnType()) << " " << sub.name << "(";

  bool first = true;
  for (const auto *arg : sub.getArguments()) {
    if (!first)
      os << ", ";
    first = false;
    switch (arg->direction) {
    case slang::ast::ArgumentDirection::In:
      os << "input ";
      break;
    case slang::ast::ArgumentDirection::Out:
      os << "output ";
      break;
    case slang::ast::ArgumentDirection::InOut:
      os << "inout ";
      break;
    case slang::ast::ArgumentDirection::Ref:
      os << "ref ";
      break;
    }
    os << formatTypeDescription(arg->getType()) << " " << arg->name;
  }
  os << ")";

  return result;
}

llvm::lsp::SignatureHelp
VerilogDocument::getSignatureHelp(const llvm::lsp::URIForFile &uri,
                                  const llvm::lsp::Position &pos) {
  llvm::lsp::SignatureHelp result;

  if (failed(compilation))
    return result;

  auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);
  auto offsetOpt = lspPositionToOffset(pos);
  if (!offsetOpt)
    return result;

  uint32_t offset = *offsetOpt;

  // Find the function/task call context
  auto [sub, activeParam] =
      findCallContext(text, offset, **compilation, mainBufferId);

  if (!sub)
    return result;

  // Build the signature information
  llvm::lsp::SignatureInformation sigInfo;
  sigInfo.label = formatSubroutineSignature(*sub);

  // Build the documentation
  std::string doc;
  llvm::raw_string_ostream docOs(doc);
  if (sub->subroutineKind == slang::ast::SubroutineKind::Function)
    docOs << "**Function** `" << sub->name << "`\n\n";
  else
    docOs << "**Task** `" << sub->name << "`\n\n";

  if (!sub->getArguments().empty()) {
    docOs << "**Parameters:**\n";
    for (const auto *arg : sub->getArguments()) {
      docOs << "- `" << arg->name << "`: "
            << formatTypeDescription(arg->getType()) << "\n";
    }
  }
  sigInfo.documentation = doc;

  // Build parameter information
  // Calculate label offsets for each parameter
  std::string label = sigInfo.label;
  size_t parenPos = label.find('(');
  size_t currentPos = parenPos + 1;

  for (const auto *arg : sub->getArguments()) {
    llvm::lsp::ParameterInformation paramInfo;

    // Find this parameter in the label
    std::string paramLabel;
    llvm::raw_string_ostream paramOs(paramLabel);
    switch (arg->direction) {
    case slang::ast::ArgumentDirection::In:
      paramOs << "input ";
      break;
    case slang::ast::ArgumentDirection::Out:
      paramOs << "output ";
      break;
    case slang::ast::ArgumentDirection::InOut:
      paramOs << "inout ";
      break;
    case slang::ast::ArgumentDirection::Ref:
      paramOs << "ref ";
      break;
    }
    paramOs << formatTypeDescription(arg->getType()) << " " << arg->name;

    paramInfo.labelString = paramLabel;

    // Find the offset in the label
    size_t paramStart = label.find(paramLabel, currentPos);
    if (paramStart != std::string::npos) {
      paramInfo.labelOffsets = {static_cast<unsigned>(paramStart),
                                static_cast<unsigned>(paramStart + paramLabel.length())};
      currentPos = paramStart + paramLabel.length();
    }

    paramInfo.documentation =
        formatTypeDescription(arg->getType()) + " " + std::string(arg->name);

    sigInfo.parameters.push_back(std::move(paramInfo));
  }

  result.signatures.push_back(std::move(sigInfo));
  result.activeSignature = 0;
  result.activeParameter = std::min(activeParam,
                                    static_cast<int>(sub->getArguments().size()) - 1);
  if (result.activeParameter < 0)
    result.activeParameter = 0;

  return result;
}

//===----------------------------------------------------------------------===//
// Document Formatting
//===----------------------------------------------------------------------===//

namespace {

/// Check if a character is a Verilog keyword that should have space after it.
/// Note: This function is currently unused but kept for potential future use
/// in more sophisticated formatting.
[[maybe_unused]]
static bool isKeywordRequiringSpace(llvm::StringRef word) {
  // Use a simple string comparison approach for keywords
  return word == "if" || word == "else" || word == "for" || word == "while" ||
         word == "do" || word == "case" || word == "casex" ||
         word == "casez" || word == "begin" || word == "end" ||
         word == "module" || word == "endmodule" || word == "function" ||
         word == "endfunction" || word == "task" || word == "endtask" ||
         word == "class" || word == "endclass" || word == "package" ||
         word == "endpackage" || word == "always" || word == "always_ff" ||
         word == "always_comb" || word == "always_latch" ||
         word == "initial" || word == "final" || word == "assign" ||
         word == "return" || word == "repeat" || word == "foreach" ||
         word == "forever" || word == "wait" || word == "fork" ||
         word == "join" || word == "join_any" || word == "join_none" ||
         word == "disable" || word == "assert" || word == "assume" ||
         word == "cover" || word == "expect" || word == "import" ||
         word == "export" || word == "typedef" || word == "input" ||
         word == "output" || word == "inout" || word == "ref" ||
         word == "const" || word == "static" || word == "automatic" ||
         word == "virtual" || word == "extends" || word == "implements" ||
         word == "generate" || word == "endgenerate" || word == "interface" ||
         word == "endinterface" || word == "property" ||
         word == "endproperty" || word == "sequence" || word == "endsequence";
}

/// Check if a line is a preprocessor directive.
static bool isPreprocessorLine(llvm::StringRef line) {
  llvm::StringRef trimmed = line.ltrim();
  return trimmed.starts_with("`");
}

/// Check if a line is a comment line.
static bool isCommentLine(llvm::StringRef line) {
  llvm::StringRef trimmed = line.ltrim();
  return trimmed.starts_with("//") || trimmed.starts_with("/*");
}

/// Get the indentation level change for a line.
/// Returns positive for lines that increase indent, negative for decrease.
static int getIndentDelta(llvm::StringRef line) {
  llvm::StringRef trimmed = line.ltrim();

  // Skip empty lines, comments, and preprocessor directives
  if (trimmed.empty() || isCommentLine(line) || isPreprocessorLine(line))
    return 0;

  int delta = 0;

  // Keywords that increase indentation
  if (trimmed.starts_with("begin") || trimmed.starts_with("module ") ||
      trimmed.starts_with("function ") || trimmed.starts_with("task ") ||
      trimmed.starts_with("class ") || trimmed.starts_with("package ") ||
      trimmed.starts_with("interface ") || trimmed.starts_with("generate") ||
      trimmed.starts_with("always") || trimmed.starts_with("initial") ||
      trimmed.starts_with("final") || trimmed.starts_with("if ") ||
      trimmed.starts_with("if(") || trimmed.starts_with("else ") ||
      trimmed.starts_with("else\n") || trimmed.starts_with("for ") ||
      trimmed.starts_with("for(") || trimmed.starts_with("while ") ||
      trimmed.starts_with("while(") || trimmed.starts_with("case ") ||
      trimmed.starts_with("case(") || trimmed.starts_with("casex ") ||
      trimmed.starts_with("casex(") || trimmed.starts_with("casez ") ||
      trimmed.starts_with("casez(") || trimmed.starts_with("fork") ||
      trimmed.starts_with("property ") || trimmed.starts_with("sequence ") ||
      trimmed.starts_with("covergroup ")) {
    ++delta;
  }

  // Keywords that decrease indentation
  if (trimmed.starts_with("end") || trimmed.starts_with("endmodule") ||
      trimmed.starts_with("endfunction") || trimmed.starts_with("endtask") ||
      trimmed.starts_with("endclass") || trimmed.starts_with("endpackage") ||
      trimmed.starts_with("endinterface") || trimmed.starts_with("endgenerate") ||
      trimmed.starts_with("endcase") || trimmed.starts_with("join") ||
      trimmed.starts_with("endproperty") || trimmed.starts_with("endsequence") ||
      trimmed.starts_with("endgroup")) {
    --delta;
  }

  return delta;
}

/// Check if the line ends with a statement that should not increase indent.
static bool endsWithSemicolon(llvm::StringRef line) {
  llvm::StringRef trimmed = line.rtrim();
  return !trimmed.empty() && trimmed.back() == ';';
}

/// Format a single line of Verilog code.
static std::string formatLine(llvm::StringRef line, unsigned indentLevel,
                              const VerilogDocument::FormattingOptions &options) {
  // Preserve empty lines
  if (line.trim().empty())
    return "";

  // Don't modify preprocessor directives
  if (isPreprocessorLine(line))
    return line.str();

  std::string result;
  llvm::raw_string_ostream os(result);

  // Generate indentation
  std::string indent;
  if (options.insertSpaces) {
    indent = std::string(indentLevel * options.tabSize, ' ');
  } else {
    indent = std::string(indentLevel, '\t');
  }

  os << indent;

  // Process the trimmed content
  llvm::StringRef content = line.ltrim();

  // Add spaces after keywords and around operators
  bool inString = false;
  bool inComment = false;
  bool prevWasSpace = false;

  for (size_t i = 0; i < content.size(); ++i) {
    char c = content[i];
    char next = (i + 1 < content.size()) ? content[i + 1] : '\0';
    char prev = (i > 0) ? content[i - 1] : '\0';

    // Track string literals
    if (c == '"' && prev != '\\')
      inString = !inString;

    // Track comments
    if (!inString && c == '/' && next == '/')
      inComment = true;

    if (inString || inComment) {
      os << c;
      prevWasSpace = (c == ' ');
      continue;
    }

    // Add space after comma if not present
    if (c == ',' && next != ' ' && next != '\n' && next != '\0') {
      os << ", ";
      prevWasSpace = true;
      continue;
    }

    // Add space around binary operators (=, ==, !=, etc.)
    // But not for <= (less-than-or-equal or non-blocking assignment)
    if ((c == '=' && prev != '!' && prev != '=' && prev != '<' &&
         prev != '>' && prev != '+' && prev != '-' && prev != '*' &&
         prev != '/' && prev != '&' && prev != '|' && prev != '^') &&
        next != '=') {
      if (!prevWasSpace && i > 0)
        os << ' ';
      os << c;
      if (next != ' ' && next != '\0' && next != '\n')
        os << ' ';
      prevWasSpace = true;
      continue;
    }

    os << c;
    prevWasSpace = (c == ' ');
  }

  return result;
}

/// Calculate the expected indentation level for a line.
static unsigned calculateIndentLevel(
    const std::vector<std::string> &lines, size_t lineIndex) {
  unsigned level = 0;

  for (size_t i = 0; i < lineIndex; ++i) {
    llvm::StringRef line(lines[i]);
    llvm::StringRef trimmed = line.ltrim();

    // Skip empty lines and comments
    if (trimmed.empty() || isCommentLine(line) || isPreprocessorLine(line))
      continue;

    int delta = getIndentDelta(trimmed);

    // If this line has a closing keyword, it decreases indent before itself
    if (delta < 0) {
      // The negative delta applies to this line and subsequent
    } else if (delta > 0) {
      // Check if the line ends with semicolon (single-line construct)
      if (!endsWithSemicolon(trimmed)) {
        level += delta;
      }
    }
  }

  // Check if current line starts with a closing keyword
  if (lineIndex < lines.size()) {
    llvm::StringRef currentLine(lines[lineIndex]);
    llvm::StringRef trimmed = currentLine.ltrim();
    if (trimmed.starts_with("end") || trimmed.starts_with("join") ||
        trimmed.starts_with("else")) {
      if (level > 0)
        --level;
    }
  }

  return level;
}

} // namespace

void VerilogDocument::formatDocument(const llvm::lsp::URIForFile &uri,
                                     const FormattingOptions &options,
                                     std::vector<llvm::lsp::TextEdit> &edits) {
  auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  if (text.empty())
    return;

  // Remove trailing null characters from the text
  while (!text.empty() && text.back() == '\0')
    text = text.substr(0, text.size() - 1);

  if (text.empty())
    return;

  // Split into lines
  std::vector<std::string> lines;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n') {
      lines.push_back(std::string(text.substr(start, i - start)));
      start = i + 1;
    }
  }
  if (start < text.size())
    lines.push_back(std::string(text.substr(start)));

  // Format each line and track changes
  std::string formattedText;
  llvm::raw_string_ostream os(formattedText);

  unsigned currentIndent = 0;
  bool inMultiLineComment = false;

  for (size_t i = 0; i < lines.size(); ++i) {
    llvm::StringRef line(lines[i]);
    llvm::StringRef trimmed = line.ltrim();

    // Track multi-line comments
    if (trimmed.contains("/*") && !trimmed.contains("*/"))
      inMultiLineComment = true;
    if (inMultiLineComment) {
      if (trimmed.contains("*/"))
        inMultiLineComment = false;
      os << line << "\n";
      continue;
    }

    // Skip empty lines - just add newline
    if (trimmed.empty()) {
      os << "\n";
      continue;
    }

    // Calculate expected indent for this line
    // Decrease indent for closing keywords before processing
    if (trimmed.starts_with("end") || trimmed.starts_with("join") ||
        trimmed.starts_with("else")) {
      if (currentIndent > 0)
        --currentIndent;
    }

    // Format the line
    std::string formatted = formatLine(line, currentIndent, options);
    os << formatted << "\n";

    // Update indent for next line
    int delta = getIndentDelta(trimmed);
    if (delta > 0 && !endsWithSemicolon(trimmed)) {
      currentIndent += delta;
    }
  }

  // Create a single edit replacing the entire document
  llvm::lsp::TextEdit edit;
  edit.range = llvm::lsp::Range(
      llvm::lsp::Position(0, 0),
      llvm::lsp::Position(static_cast<int>(lines.size()), 0));
  edit.newText = formattedText;

  // Only add the edit if there are actual changes
  // Compare without potential trailing null
  std::string originalText(text);
  // Remove trailing null if present
  while (!originalText.empty() && originalText.back() == '\0')
    originalText.pop_back();
  while (!edit.newText.empty() && edit.newText.back() == '\0')
    edit.newText.pop_back();
  if (edit.newText != originalText)
    edits.push_back(std::move(edit));
}

void VerilogDocument::formatRange(const llvm::lsp::URIForFile &uri,
                                  const llvm::lsp::Range &range,
                                  const FormattingOptions &options,
                                  std::vector<llvm::lsp::TextEdit> &edits) {
  auto &sm = getSlangSourceManager();
  std::string_view text = sm.getSourceText(mainBufferId);

  if (text.empty())
    return;

  // Remove trailing null characters from the text
  while (!text.empty() && text.back() == '\0')
    text = text.substr(0, text.size() - 1);

  if (text.empty())
    return;

  // Split into lines
  std::vector<std::string> lines;
  size_t start = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '\n') {
      lines.push_back(std::string(text.substr(start, i - start)));
      start = i + 1;
    }
  }
  if (start < text.size())
    lines.push_back(std::string(text.substr(start)));

  // Determine the range of lines to format
  int startLine = std::max(0, range.start.line);
  int endLine = std::min(static_cast<int>(lines.size()) - 1, range.end.line);

  if (startLine > endLine || startLine >= static_cast<int>(lines.size()))
    return;

  // Calculate the base indentation level at the start of the range
  unsigned baseIndent = calculateIndentLevel(lines, startLine);

  // Format the lines in the range
  std::string formattedText;
  llvm::raw_string_ostream os(formattedText);

  unsigned currentIndent = baseIndent;
  bool inMultiLineComment = false;

  for (int i = startLine; i <= endLine; ++i) {
    llvm::StringRef line(lines[i]);
    llvm::StringRef trimmed = line.ltrim();

    // Track multi-line comments
    if (trimmed.contains("/*") && !trimmed.contains("*/"))
      inMultiLineComment = true;
    if (inMultiLineComment) {
      if (trimmed.contains("*/"))
        inMultiLineComment = false;
      os << line;
      if (i < endLine)
        os << "\n";
      continue;
    }

    // Skip empty lines - just add newline
    if (trimmed.empty()) {
      if (i < endLine)
        os << "\n";
      continue;
    }

    // Decrease indent for closing keywords before processing
    if (trimmed.starts_with("end") || trimmed.starts_with("join") ||
        trimmed.starts_with("else")) {
      if (currentIndent > 0)
        --currentIndent;
    }

    // Format the line
    std::string formatted = formatLine(line, currentIndent, options);
    os << formatted;
    if (i < endLine)
      os << "\n";

    // Update indent for next line
    int delta = getIndentDelta(trimmed);
    if (delta > 0 && !endsWithSemicolon(trimmed)) {
      currentIndent += delta;
    }
  }

  // Create edit for the range
  llvm::lsp::TextEdit edit;
  edit.range = llvm::lsp::Range(
      llvm::lsp::Position(startLine, 0),
      llvm::lsp::Position(endLine, static_cast<int>(lines[endLine].size())));
  edit.newText = formattedText;

  // Only add the edit if there are actual changes
  std::string originalRange;
  for (int i = startLine; i <= endLine; ++i) {
    originalRange += lines[i];
    if (i < endLine)
      originalRange += "\n";
  }

  if (edit.newText != originalRange)
    edits.push_back(std::move(edit));
}

//===----------------------------------------------------------------------===//
// Call Hierarchy
//===----------------------------------------------------------------------===//

namespace {

/// Helper to encode subroutine location for data field.
static std::string encodeSubroutineData(const llvm::lsp::URIForFile &uri,
                                        int line, int col) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << uri.file() << ":" << line << ":" << col;
  return result;
}

/// Helper to decode subroutine location from data field.
static bool decodeSubroutineData(llvm::StringRef data, std::string &file,
                                 int &line, int &col) {
  auto lastColon = data.rfind(':');
  if (lastColon == llvm::StringRef::npos)
    return false;

  auto secondLastColon = data.rfind(':', lastColon - 1);
  if (secondLastColon == llvm::StringRef::npos)
    return false;

  file = data.substr(0, secondLastColon).str();
  if (data.substr(secondLastColon + 1, lastColon - secondLastColon - 1)
          .getAsInteger(10, line))
    return false;
  if (data.substr(lastColon + 1).getAsInteger(10, col))
    return false;
  return true;
}

/// Visitor to collect call expressions within a subroutine.
class CallExpressionCollector
    : public slang::ast::ASTVisitor<CallExpressionCollector, true, true> {
public:
  CallExpressionCollector(slang::BufferID bufferId,
                          const slang::SourceManager &sm)
      : bufferId(bufferId), sm(sm) {}

  struct CallInfo {
    const slang::ast::SubroutineSymbol *callee;
    slang::SourceRange callRange;
  };

  std::vector<CallInfo> calls;

  void handle(const slang::ast::CallExpression &expr) {
    if (expr.isSystemCall())
      return;

    const auto *sub =
        std::get_if<const slang::ast::SubroutineSymbol *>(&expr.subroutine);
    if (!sub || !*sub)
      return;

    // Get the call site location
    slang::SourceRange callRange;
    if (expr.syntax) {
      if (auto *invocation =
              expr.syntax->as_if<slang::syntax::InvocationExpressionSyntax>()) {
        callRange = invocation->left->sourceRange();
      }
    }
    if (!callRange.start().valid()) {
      callRange = expr.sourceRange;
    }

    // Only include calls from this buffer
    if (callRange.start().buffer() == bufferId) {
      calls.push_back({*sub, callRange});
    }

    visitDefault(expr);
  }

  template <typename T>
  void visit(const T &node) {
    if constexpr (std::is_same_v<T, slang::ast::CallExpression>) {
      handle(node);
    } else {
      visitDefault(node);
    }
  }

private:
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
};

/// Visitor to find all call sites of a specific subroutine.
class IncomingCallsCollector
    : public slang::ast::ASTVisitor<IncomingCallsCollector, true, true> {
public:
  IncomingCallsCollector(const slang::ast::SubroutineSymbol *target,
                         slang::BufferID bufferId,
                         const slang::SourceManager &sm)
      : target(target), bufferId(bufferId), sm(sm) {}

  struct CallerInfo {
    const slang::ast::SubroutineSymbol *caller;
    slang::SourceRange callRange;
  };

  std::vector<CallerInfo> callers;
  const slang::ast::SubroutineSymbol *currentSubroutine = nullptr;

  void handle(const slang::ast::SubroutineSymbol &sub) {
    auto *prevSub = currentSubroutine;
    currentSubroutine = &sub;
    visitDefault(sub);
    currentSubroutine = prevSub;
  }

  void handle(const slang::ast::CallExpression &expr) {
    if (expr.isSystemCall()) {
      visitDefault(expr);
      return;
    }

    const auto *sub =
        std::get_if<const slang::ast::SubroutineSymbol *>(&expr.subroutine);
    if (!sub || !*sub) {
      visitDefault(expr);
      return;
    }

    // Check if this call is to our target subroutine
    if (*sub == target && currentSubroutine) {
      slang::SourceRange callRange;
      if (expr.syntax) {
        if (auto *invocation =
                expr.syntax->as_if<slang::syntax::InvocationExpressionSyntax>()) {
          callRange = invocation->left->sourceRange();
        }
      }
      if (!callRange.start().valid()) {
        callRange = expr.sourceRange;
      }

      if (callRange.start().buffer() == bufferId) {
        callers.push_back({currentSubroutine, callRange});
      }
    }

    visitDefault(expr);
  }

  template <typename T>
  void visit(const T &node) {
    if constexpr (std::is_same_v<T, slang::ast::SubroutineSymbol>) {
      handle(node);
    } else if constexpr (std::is_same_v<T, slang::ast::CallExpression>) {
      handle(node);
    } else {
      visitDefault(node);
    }
  }

private:
  const slang::ast::SubroutineSymbol *target;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
};

} // namespace

std::optional<VerilogDocument::CallHierarchyItem>
VerilogDocument::prepareCallHierarchy(const llvm::lsp::URIForFile &uri,
                                      const llvm::lsp::Position &pos) {
  if (!index || failed(compilation))
    return std::nullopt;

  const auto *slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return std::nullopt;

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  if (!it.valid() || slangBufferPointer < it.start())
    return std::nullopt;

  auto element = it.value();

  // Check if it's a symbol
  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(element);
  if (!symbol)
    return std::nullopt;

  // Only handle subroutines (functions and tasks)
  if (symbol->kind != slang::ast::SymbolKind::Subroutine)
    return std::nullopt;

  const auto &sub = symbol->as<slang::ast::SubroutineSymbol>();
  const auto &sm = getSlangSourceManager();

  // Build the call hierarchy item
  CallHierarchyItem item;
  item.name = sub.name;
  item.kind = llvm::lsp::SymbolKind::Function;

  // Build detail string (signature)
  std::string detail;
  llvm::raw_string_ostream detailOs(detail);
  if (sub.subroutineKind == slang::ast::SubroutineKind::Function)
    detailOs << "function ";
  else
    detailOs << "task ";
  detailOs << formatTypeDescription(sub.getReturnType()) << " " << sub.name
           << "(";
  bool first = true;
  for (const auto *arg : sub.getArguments()) {
    if (!first)
      detailOs << ", ";
    first = false;
    detailOs << formatTypeDescription(arg->getType()) << " " << arg->name;
  }
  detailOs << ")";
  item.detail = detail;

  item.uri = uri;

  // Get the range of the entire subroutine
  if (auto *syntax = sub.getSyntax()) {
    auto syntaxRange = syntax->sourceRange();
    int startLine = slangLineToLsp(sm.getLineNumber(syntaxRange.start()));
    int startCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.start()));
    int endLine = slangLineToLsp(sm.getLineNumber(syntaxRange.end()));
    int endCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.end()));
    item.range =
        llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                         llvm::lsp::Position(endLine, endCol));
  } else {
    // Fallback to just the name location
    int line = slangLineToLsp(sm.getLineNumber(sub.location));
    int col = slangColumnToLsp(sm.getColumnNumber(sub.location));
    item.range = llvm::lsp::Range(llvm::lsp::Position(line, col),
                                  llvm::lsp::Position(line, col + sub.name.size()));
  }

  // Selection range is just the name
  int nameLine = slangLineToLsp(sm.getLineNumber(sub.location));
  int nameCol = slangColumnToLsp(sm.getColumnNumber(sub.location));
  item.selectionRange =
      llvm::lsp::Range(llvm::lsp::Position(nameLine, nameCol),
                       llvm::lsp::Position(nameLine, nameCol + sub.name.size()));

  // Encode location for later lookup
  item.data = encodeSubroutineData(uri, nameLine, nameCol);

  return item;
}

void VerilogDocument::getIncomingCalls(
    const CallHierarchyItem &item,
    std::vector<CallHierarchyIncomingCall> &calls) {
  if (failed(compilation))
    return;

  // Decode the target subroutine location from item.data
  std::string file;
  int targetLine, targetCol;
  if (!decodeSubroutineData(item.data, file, targetLine, targetCol))
    return;

  const auto &sm = getSlangSourceManager();

  // Find the target subroutine by matching name and checking if it matches
  // the expected location
  const slang::ast::SubroutineSymbol *targetSub = nullptr;

  auto findSubroutine = [&](const slang::ast::Scope &scope) {
    for (const auto &member : scope.members()) {
      if (member.kind == slang::ast::SymbolKind::Subroutine &&
          member.name == item.name) {
        const auto &sub = member.as<slang::ast::SubroutineSymbol>();
        int line = slangLineToLsp(sm.getLineNumber(sub.location));
        int col = slangColumnToLsp(sm.getColumnNumber(sub.location));
        if (line == targetLine && col == targetCol) {
          targetSub = &sub;
          return true;
        }
      }
      // Check nested scopes (classes)
      if (member.kind == slang::ast::SymbolKind::ClassType) {
        const auto &classType = member.as<slang::ast::ClassType>();
        for (const auto &classMember : classType.members()) {
          if (classMember.kind == slang::ast::SymbolKind::Subroutine &&
              classMember.name == item.name) {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            int line = slangLineToLsp(sm.getLineNumber(sub.location));
            int col = slangColumnToLsp(sm.getColumnNumber(sub.location));
            if (line == targetLine && col == targetCol) {
              targetSub = &sub;
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  // Search in packages
  for (auto *package : (*compilation)->getPackages()) {
    if (findSubroutine(*package))
      break;
  }

  // Search in top instances
  if (!targetSub) {
    for (auto *inst : (*compilation)->getRoot().topInstances) {
      if (findSubroutine(inst->body))
        break;
    }
  }

  if (!targetSub)
    return;

  // Now find all calls to this subroutine
  IncomingCallsCollector collector(targetSub, mainBufferId, sm);

  // Visit all scopes
  for (auto *package : (*compilation)->getPackages()) {
    if (package->location.buffer() == mainBufferId)
      package->visit(collector);
  }
  for (auto *inst : (*compilation)->getRoot().topInstances) {
    if (inst->body.location.buffer() == mainBufferId)
      inst->body.visit(collector);
  }

  // Group calls by caller
  llvm::DenseMap<const slang::ast::SubroutineSymbol *,
                 std::vector<slang::SourceRange>>
      callerMap;
  for (const auto &callerInfo : collector.callers) {
    callerMap[callerInfo.caller].push_back(callerInfo.callRange);
  }

  // Build the incoming calls list
  for (const auto &[caller, ranges] : callerMap) {
    CallHierarchyIncomingCall incoming;

    // Build the caller item
    incoming.from.name = caller->name;
    incoming.from.kind = llvm::lsp::SymbolKind::Function;

    // Build detail
    std::string detail;
    llvm::raw_string_ostream detailOs(detail);
    if (caller->subroutineKind == slang::ast::SubroutineKind::Function)
      detailOs << "function ";
    else
      detailOs << "task ";
    detailOs << caller->name;
    incoming.from.detail = detail;

    incoming.from.uri = uri;

    // Range of the caller
    if (auto *syntax = caller->getSyntax()) {
      auto syntaxRange = syntax->sourceRange();
      int startLine = slangLineToLsp(sm.getLineNumber(syntaxRange.start()));
      int startCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.start()));
      int endLine = slangLineToLsp(sm.getLineNumber(syntaxRange.end()));
      int endCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.end()));
      incoming.from.range =
          llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                           llvm::lsp::Position(endLine, endCol));
    } else {
      int line = slangLineToLsp(sm.getLineNumber(caller->location));
      int col = slangColumnToLsp(sm.getColumnNumber(caller->location));
      incoming.from.range =
          llvm::lsp::Range(llvm::lsp::Position(line, col),
                           llvm::lsp::Position(line, col + caller->name.size()));
    }

    // Selection range
    int callerLine = slangLineToLsp(sm.getLineNumber(caller->location));
    int callerCol = slangColumnToLsp(sm.getColumnNumber(caller->location));
    incoming.from.selectionRange = llvm::lsp::Range(
        llvm::lsp::Position(callerLine, callerCol),
        llvm::lsp::Position(callerLine, callerCol + caller->name.size()));

    incoming.from.data = encodeSubroutineData(uri, callerLine, callerCol);

    // Convert call ranges
    for (const auto &range : ranges) {
      int startLine = slangLineToLsp(sm.getLineNumber(range.start()));
      int startCol = slangColumnToLsp(sm.getColumnNumber(range.start()));
      int endLine = slangLineToLsp(sm.getLineNumber(range.end()));
      int endCol = slangColumnToLsp(sm.getColumnNumber(range.end()));
      incoming.fromRanges.push_back(
          llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                           llvm::lsp::Position(endLine, endCol)));
    }

    calls.push_back(std::move(incoming));
  }
}

void VerilogDocument::getOutgoingCalls(
    const CallHierarchyItem &item,
    std::vector<CallHierarchyOutgoingCall> &calls) {
  if (failed(compilation))
    return;

  // Decode the source subroutine location from item.data
  std::string file;
  int sourceLine, sourceCol;
  if (!decodeSubroutineData(item.data, file, sourceLine, sourceCol))
    return;

  const auto &sm = getSlangSourceManager();

  // Find the source subroutine
  const slang::ast::SubroutineSymbol *sourceSub = nullptr;

  auto findSubroutine = [&](const slang::ast::Scope &scope) {
    for (const auto &member : scope.members()) {
      if (member.kind == slang::ast::SymbolKind::Subroutine &&
          member.name == item.name) {
        const auto &sub = member.as<slang::ast::SubroutineSymbol>();
        int line = slangLineToLsp(sm.getLineNumber(sub.location));
        int col = slangColumnToLsp(sm.getColumnNumber(sub.location));
        if (line == sourceLine && col == sourceCol) {
          sourceSub = &sub;
          return true;
        }
      }
      // Check nested scopes (classes)
      if (member.kind == slang::ast::SymbolKind::ClassType) {
        const auto &classType = member.as<slang::ast::ClassType>();
        for (const auto &classMember : classType.members()) {
          if (classMember.kind == slang::ast::SymbolKind::Subroutine &&
              classMember.name == item.name) {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            int line = slangLineToLsp(sm.getLineNumber(sub.location));
            int col = slangColumnToLsp(sm.getColumnNumber(sub.location));
            if (line == sourceLine && col == sourceCol) {
              sourceSub = &sub;
              return true;
            }
          }
        }
      }
    }
    return false;
  };

  // Search in packages
  for (auto *package : (*compilation)->getPackages()) {
    if (findSubroutine(*package))
      break;
  }

  // Search in top instances
  if (!sourceSub) {
    for (auto *inst : (*compilation)->getRoot().topInstances) {
      if (findSubroutine(inst->body))
        break;
    }
  }

  if (!sourceSub)
    return;

  // Collect all calls from this subroutine
  CallExpressionCollector collector(mainBufferId, sm);
  sourceSub->visit(collector);

  // Group calls by callee
  llvm::DenseMap<const slang::ast::SubroutineSymbol *,
                 std::vector<slang::SourceRange>>
      calleeMap;
  for (const auto &callInfo : collector.calls) {
    calleeMap[callInfo.callee].push_back(callInfo.callRange);
  }

  // Build the outgoing calls list
  for (const auto &[callee, ranges] : calleeMap) {
    CallHierarchyOutgoingCall outgoing;

    // Build the callee item
    outgoing.to.name = callee->name;
    outgoing.to.kind = llvm::lsp::SymbolKind::Function;

    // Build detail
    std::string detail;
    llvm::raw_string_ostream detailOs(detail);
    if (callee->subroutineKind == slang::ast::SubroutineKind::Function)
      detailOs << "function ";
    else
      detailOs << "task ";
    detailOs << callee->name;
    outgoing.to.detail = detail;

    // Determine URI - callee might be in a different file
    llvm::lsp::URIForFile calleeUri = uri;
    if (callee->location.valid() && callee->location.buffer() != mainBufferId) {
      auto fileName = sm.getFileName(callee->location);
      if (!fileName.empty()) {
        llvm::SmallString<256> absPath(fileName);
        if (!llvm::sys::path::is_absolute(absPath)) {
          llvm::sys::fs::make_absolute(absPath);
        }
        if (auto uriOrErr = llvm::lsp::URIForFile::fromFile(absPath)) {
          calleeUri = *uriOrErr;
        }
      }
    }
    outgoing.to.uri = calleeUri;

    // Range of the callee
    if (auto *syntax = callee->getSyntax()) {
      auto syntaxRange = syntax->sourceRange();
      if (syntaxRange.start().valid()) {
        int startLine = slangLineToLsp(sm.getLineNumber(syntaxRange.start()));
        int startCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.start()));
        int endLine = slangLineToLsp(sm.getLineNumber(syntaxRange.end()));
        int endCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.end()));
        outgoing.to.range =
            llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                             llvm::lsp::Position(endLine, endCol));
      } else {
        int line = slangLineToLsp(sm.getLineNumber(callee->location));
        int col = slangColumnToLsp(sm.getColumnNumber(callee->location));
        outgoing.to.range = llvm::lsp::Range(
            llvm::lsp::Position(line, col),
            llvm::lsp::Position(line, col + callee->name.size()));
      }
    } else {
      int line = slangLineToLsp(sm.getLineNumber(callee->location));
      int col = slangColumnToLsp(sm.getColumnNumber(callee->location));
      outgoing.to.range = llvm::lsp::Range(
          llvm::lsp::Position(line, col),
          llvm::lsp::Position(line, col + callee->name.size()));
    }

    // Selection range
    int calleeLine = slangLineToLsp(sm.getLineNumber(callee->location));
    int calleeCol = slangColumnToLsp(sm.getColumnNumber(callee->location));
    outgoing.to.selectionRange = llvm::lsp::Range(
        llvm::lsp::Position(calleeLine, calleeCol),
        llvm::lsp::Position(calleeLine, calleeCol + callee->name.size()));

    outgoing.to.data = encodeSubroutineData(calleeUri, calleeLine, calleeCol);

    // Convert call ranges (these are the locations in the source subroutine
    // where the call is made)
    for (const auto &range : ranges) {
      int startLine = slangLineToLsp(sm.getLineNumber(range.start()));
      int startCol = slangColumnToLsp(sm.getColumnNumber(range.start()));
      int endLine = slangLineToLsp(sm.getLineNumber(range.end()));
      int endCol = slangColumnToLsp(sm.getColumnNumber(range.end()));
      outgoing.fromRanges.push_back(
          llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                           llvm::lsp::Position(endLine, endCol)));
    }

    calls.push_back(std::move(outgoing));
  }
}

//===----------------------------------------------------------------------===//
// Code Lens
//===----------------------------------------------------------------------===//

namespace {

/// Visitor to collect code lens information from the AST.
class CodeLensVisitor
    : public slang::ast::ASTVisitor<CodeLensVisitor, true, true> {
public:
  CodeLensVisitor(const VerilogDocument &doc, slang::BufferID bufferId,
                  const slang::SourceManager &sm, const VerilogIndex *index)
      : doc(doc), bufferId(bufferId), sm(sm), index(index) {}

  std::vector<VerilogDocument::CodeLensInfo> lenses;

  /// Count references to a symbol.
  size_t countReferences(const slang::ast::Symbol *symbol) const {
    if (!index)
      return 0;
    auto it = index->getReferences().find(symbol);
    if (it == index->getReferences().end())
      return 0;
    return it->second.size();
  }

  /// Create a reference count code lens.
  void addReferenceLens(const slang::ast::Symbol &symbol, llvm::StringRef kind) {
    if (symbol.location.buffer() != bufferId)
      return;
    if (symbol.name.empty())
      return;

    size_t refCount = countReferences(&symbol);

    int line = slangLineToLsp(sm.getLineNumber(symbol.location));
    int col = slangColumnToLsp(sm.getColumnNumber(symbol.location));

    VerilogDocument::CodeLensInfo lens;
    lens.range = llvm::lsp::Range(
        llvm::lsp::Position(line, col),
        llvm::lsp::Position(line, col + static_cast<int>(symbol.name.size())));

    // Format the title
    std::string title;
    llvm::raw_string_ostream os(title);
    if (refCount == 0)
      os << "no references";
    else if (refCount == 1)
      os << "1 reference";
    else
      os << refCount << " references";
    lens.title = title;

    // Command to find references
    lens.command = "editor.action.findReferences";
    lens.commandArguments.push_back(doc.getURI().uri().str());
    lens.commandArguments.push_back(std::to_string(line));
    lens.commandArguments.push_back(std::to_string(col));

    // Data for lazy resolution (format: uri:line:col:type)
    std::string data;
    llvm::raw_string_ostream dataOs(data);
    dataOs << doc.getURI().file() << ":" << line << ":" << col << ":" << kind.str();
    lens.data = data;

    lenses.push_back(std::move(lens));
  }

  /// Handle module/interface instance body.
  void visit(const slang::ast::InstanceBodySymbol &body) {
    // Add reference lens for the module/interface definition
    if (body.location.buffer() == bufferId) {
      addReferenceLens(body, "module");
    }

    // Visit members for functions, tasks, etc.
    for (const auto &member : body.members()) {
      // Handle subroutines (functions/tasks)
      if (member.kind == slang::ast::SymbolKind::Subroutine) {
        addReferenceLens(member, "function");
      }
      // Handle classes
      else if (member.kind == slang::ast::SymbolKind::ClassType) {
        const auto &classType = member.as<slang::ast::ClassType>();
        addReferenceLens(classType, "class");

        // Add lenses for class methods
        for (const auto &classMember : classType.members()) {
          if (classMember.kind == slang::ast::SymbolKind::Subroutine) {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            addReferenceLens(sub, "function");

            // Add "Go to implementation" lens for virtual methods
            if (sub.flags.has(slang::ast::MethodFlags::Virtual)) {
              addVirtualMethodLens(sub);
            }
          }
        }
      }
    }
  }

  /// Handle packages.
  void visit(const slang::ast::PackageSymbol &pkg) {
    if (pkg.location.buffer() != bufferId)
      return;

    addReferenceLens(pkg, "package");

    // Visit package members
    for (const auto &member : pkg.members()) {
      if (member.kind == slang::ast::SymbolKind::Subroutine) {
        addReferenceLens(member, "function");
      } else if (member.kind == slang::ast::SymbolKind::ClassType) {
        const auto &classType = member.as<slang::ast::ClassType>();
        addReferenceLens(classType, "class");

        // Add lenses for class methods
        for (const auto &classMember : classType.members()) {
          if (classMember.kind == slang::ast::SymbolKind::Subroutine) {
            const auto &sub = classMember.as<slang::ast::SubroutineSymbol>();
            addReferenceLens(sub, "function");

            // Add "Go to implementation" lens for virtual methods
            if (sub.flags.has(slang::ast::MethodFlags::Virtual)) {
              addVirtualMethodLens(sub);
            }
          }
        }
      }
    }
  }

  /// Handle interface definitions.
  void visitInterfaceDefinition(const slang::ast::DefinitionSymbol &def) {
    if (def.definitionKind != slang::ast::DefinitionKind::Interface)
      return;
    if (def.location.buffer() != bufferId)
      return;

    addReferenceLens(def, "interface");
  }

  /// Add "Go to implementation" lens for virtual methods.
  void addVirtualMethodLens(const slang::ast::SubroutineSymbol &method) {
    if (method.location.buffer() != bufferId)
      return;

    int line = slangLineToLsp(sm.getLineNumber(method.location));
    int col = slangColumnToLsp(sm.getColumnNumber(method.location));

    VerilogDocument::CodeLensInfo lens;
    lens.range = llvm::lsp::Range(
        llvm::lsp::Position(line, col),
        llvm::lsp::Position(line, col + static_cast<int>(method.name.size())));

    lens.title = "Go to implementations";
    lens.command = "editor.action.goToImplementation";
    lens.commandArguments.push_back(doc.getURI().uri().str());
    lens.commandArguments.push_back(std::to_string(line));
    lens.commandArguments.push_back(std::to_string(col));

    // Data for lazy resolution
    std::string data;
    llvm::raw_string_ostream dataOs(data);
    dataOs << doc.getURI().file() << ":" << line << ":" << col << ":virtual";
    lens.data = data;

    lenses.push_back(std::move(lens));
  }

private:
  const VerilogDocument &doc;
  slang::BufferID bufferId;
  const slang::SourceManager &sm;
  const VerilogIndex *index;
};

} // namespace

void VerilogDocument::getCodeLenses(const llvm::lsp::URIForFile &uri,
                                     std::vector<CodeLensInfo> &lenses) {
  if (failed(compilation))
    return;

  const auto &sm = getSlangSourceManager();
  CodeLensVisitor visitor(*this, mainBufferId, sm, index.get());

  // Visit all definitions to find interfaces
  for (const auto *sym : (*compilation)->getDefinitions()) {
    if (sym->kind == slang::ast::SymbolKind::Definition) {
      const auto &def = sym->as<slang::ast::DefinitionSymbol>();
      if (def.definitionKind == slang::ast::DefinitionKind::Interface) {
        visitor.visitInterfaceDefinition(def);
      }
    }
  }

  // Visit all packages
  for (auto *package : (*compilation)->getPackages()) {
    if (package->location.buffer() == mainBufferId)
      visitor.visit(*package);
  }

  // Visit all top instances
  for (auto *inst : (*compilation)->getRoot().topInstances) {
    if (inst->body.location.buffer() == mainBufferId)
      visitor.visit(inst->body);
  }

  lenses = std::move(visitor.lenses);
}

bool VerilogDocument::resolveCodeLens(llvm::StringRef data, CodeLensInfo &lens) {
  // The data format is "uri:line:col:type"
  // Parse the data to re-compute reference count (for lazy resolution)
  auto lastColon = data.rfind(':');
  if (lastColon == llvm::StringRef::npos)
    return false;

  auto typeStr = data.substr(lastColon + 1);

  // For now, just return true if we can parse the data
  // The lens was already fully resolved in getCodeLenses
  return !typeStr.empty();
}

//===----------------------------------------------------------------------===//
// Type Hierarchy
//===----------------------------------------------------------------------===//

namespace {

/// Encode class location data for type hierarchy items.
static std::string encodeClassData(const llvm::lsp::URIForFile &uri,
                                   int line, int col) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << uri.file() << ":" << line << ":" << col;
  return result;
}

/// Decode class location data from type hierarchy items.
static bool decodeClassData(llvm::StringRef data, std::string &file,
                            int &line, int &col) {
  auto lastColon = data.rfind(':');
  if (lastColon == llvm::StringRef::npos)
    return false;

  auto colStr = data.substr(lastColon + 1);
  if (colStr.getAsInteger(10, col))
    return false;

  auto secondLastColon = data.rfind(':', lastColon - 1);
  if (secondLastColon == llvm::StringRef::npos)
    return false;

  auto lineStr = data.substr(secondLastColon + 1, lastColon - secondLastColon - 1);
  if (lineStr.getAsInteger(10, line))
    return false;

  file = data.substr(0, secondLastColon).str();
  return true;
}

/// Build a TypeHierarchyItem from a ClassType symbol.
static VerilogDocument::TypeHierarchyItem
buildTypeHierarchyItem(const slang::ast::ClassType &classType,
                       const slang::SourceManager &sm,
                       const llvm::lsp::URIForFile &uri) {
  VerilogDocument::TypeHierarchyItem item;
  item.name = classType.name;
  item.kind = llvm::lsp::SymbolKind::Class;

  // Build detail string
  std::string detail;
  llvm::raw_string_ostream detailOs(detail);
  detailOs << "class " << classType.name;
  if (classType.getBaseClass()) {
    detailOs << " extends " << classType.getBaseClass()->name;
  }
  item.detail = detail;

  item.uri = uri;

  // Get the range of the entire class
  if (auto *syntax = classType.getSyntax()) {
    auto syntaxRange = syntax->sourceRange();
    int startLine = slangLineToLsp(sm.getLineNumber(syntaxRange.start()));
    int startCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.start()));
    int endLine = slangLineToLsp(sm.getLineNumber(syntaxRange.end()));
    int endCol = slangColumnToLsp(sm.getColumnNumber(syntaxRange.end()));
    item.range =
        llvm::lsp::Range(llvm::lsp::Position(startLine, startCol),
                         llvm::lsp::Position(endLine, endCol));
  } else {
    // Fallback to just the name location
    int line = slangLineToLsp(sm.getLineNumber(classType.location));
    int col = slangColumnToLsp(sm.getColumnNumber(classType.location));
    item.range = llvm::lsp::Range(
        llvm::lsp::Position(line, col),
        llvm::lsp::Position(line, col + static_cast<int>(classType.name.size())));
  }

  // Selection range is just the name
  int nameLine = slangLineToLsp(sm.getLineNumber(classType.location));
  int nameCol = slangColumnToLsp(sm.getColumnNumber(classType.location));
  item.selectionRange = llvm::lsp::Range(
      llvm::lsp::Position(nameLine, nameCol),
      llvm::lsp::Position(nameLine, nameCol + static_cast<int>(classType.name.size())));

  // Encode location for later lookup
  item.data = encodeClassData(uri, nameLine, nameCol);

  return item;
}

/// Visitor to collect all classes and build inheritance graph.
class ClassInheritanceCollector
    : public slang::ast::ASTVisitor<ClassInheritanceCollector, true, true> {
public:
  /// Map from class symbol to its children (classes that extend it).
  llvm::DenseMap<const slang::ast::ClassType *,
                 std::vector<const slang::ast::ClassType *>> childrenMap;
  /// All classes found.
  std::vector<const slang::ast::ClassType *> allClasses;

  void visit(const slang::ast::ClassType &classType) {
    allClasses.push_back(&classType);

    // Record parent-child relationship
    if (const auto *baseClass = classType.getBaseClass()) {
      if (baseClass->isClass()) {
        const auto *baseClassType = &baseClass->getCanonicalType().as<slang::ast::ClassType>();
        childrenMap[baseClassType].push_back(&classType);
      }
    }

    // Continue visiting nested classes
    visitDefault(classType);
  }

  template <typename T>
  void visit(const T &node) {
    visitDefault(node);
  }
};

} // namespace

std::optional<VerilogDocument::TypeHierarchyItem>
VerilogDocument::prepareTypeHierarchy(const llvm::lsp::URIForFile &uri,
                                      const llvm::lsp::Position &pos) {
  if (!index || failed(compilation))
    return std::nullopt;

  const auto *slangBufferPointer = getPointerFor(pos);
  if (!slangBufferPointer)
    return std::nullopt;

  const auto &intervalMap = index->getIntervalMap();
  auto it = intervalMap.find(slangBufferPointer);

  if (!it.valid() || slangBufferPointer < it.start())
    return std::nullopt;

  auto element = it.value();

  // Check if it's a symbol
  const auto *symbol = dyn_cast<const slang::ast::Symbol *>(element);
  if (!symbol)
    return std::nullopt;

  // Handle ClassType symbols directly
  if (symbol->kind == slang::ast::SymbolKind::ClassType) {
    const auto &classType = symbol->as<slang::ast::ClassType>();
    const auto &sm = getSlangSourceManager();
    return buildTypeHierarchyItem(classType, sm, uri);
  }

  // Also handle references to classes (e.g., in extends clause or variable type)
  // Check if this is a type that refers to a class
  if (symbol->kind == slang::ast::SymbolKind::Variable ||
      symbol->kind == slang::ast::SymbolKind::ClassProperty) {
    const auto &type = symbol->kind == slang::ast::SymbolKind::Variable
                           ? symbol->as<slang::ast::VariableSymbol>().getType()
                           : symbol->as<slang::ast::ClassPropertySymbol>().getType();
    if (type.isClass()) {
      const auto &classType = type.getCanonicalType().as<slang::ast::ClassType>();
      const auto &sm = getSlangSourceManager();
      return buildTypeHierarchyItem(classType, sm, uri);
    }
  }

  return std::nullopt;
}

void VerilogDocument::getSupertypes(
    const TypeHierarchyItem &item,
    std::vector<TypeHierarchyItem> &supertypes) {
  if (failed(compilation))
    return;

  // Decode the target class location from item.data
  std::string file;
  int targetLine, targetCol;
  if (!decodeClassData(item.data, file, targetLine, targetCol))
    return;

  const auto &sm = getSlangSourceManager();

  // Find the target class by matching name and checking if it matches
  // the expected location
  const slang::ast::ClassType *targetClass = nullptr;

  auto findClass = [&](const slang::ast::Scope &scope) {
    for (const auto &member : scope.members()) {
      if (member.kind == slang::ast::SymbolKind::ClassType &&
          member.name == item.name) {
        const auto &classType = member.as<slang::ast::ClassType>();
        int line = slangLineToLsp(sm.getLineNumber(classType.location));
        int col = slangColumnToLsp(sm.getColumnNumber(classType.location));
        if (line == targetLine && col == targetCol) {
          targetClass = &classType;
          return true;
        }
      }
    }
    return false;
  };

  // Search in packages
  for (auto *package : (*compilation)->getPackages()) {
    if (findClass(*package))
      break;
  }

  // Search in top instances
  if (!targetClass) {
    for (auto *inst : (*compilation)->getRoot().topInstances) {
      if (findClass(inst->body))
        break;
    }
  }

  // Search in compilation units
  if (!targetClass) {
    for (const auto *unit : (*compilation)->getCompilationUnits()) {
      if (findClass(*unit))
        break;
    }
  }

  if (!targetClass)
    return;

  // Get the base class if it exists
  if (const auto *baseClass = targetClass->getBaseClass()) {
    if (baseClass->isClass()) {
      const auto &baseClassType = baseClass->getCanonicalType().as<slang::ast::ClassType>();
      supertypes.push_back(buildTypeHierarchyItem(baseClassType, sm, uri));
    }
  }
}

void VerilogDocument::getSubtypes(
    const TypeHierarchyItem &item,
    std::vector<TypeHierarchyItem> &subtypes) {
  if (failed(compilation))
    return;

  // Decode the target class location from item.data
  std::string file;
  int targetLine, targetCol;
  if (!decodeClassData(item.data, file, targetLine, targetCol))
    return;

  const auto &sm = getSlangSourceManager();

  // First, collect all classes and build inheritance graph
  ClassInheritanceCollector collector;

  // Visit all packages
  for (auto *package : (*compilation)->getPackages()) {
    package->visit(collector);
  }

  // Visit all top instances
  for (auto *inst : (*compilation)->getRoot().topInstances) {
    inst->body.visit(collector);
  }

  // Visit all compilation units
  for (const auto *unit : (*compilation)->getCompilationUnits()) {
    for (const auto &member : unit->members()) {
      member.visit(collector);
    }
  }

  // Find the target class by matching name and location
  const slang::ast::ClassType *targetClass = nullptr;
  for (const auto *classType : collector.allClasses) {
    if (classType->name == item.name) {
      int line = slangLineToLsp(sm.getLineNumber(classType->location));
      int col = slangColumnToLsp(sm.getColumnNumber(classType->location));
      if (line == targetLine && col == targetCol) {
        targetClass = classType;
        break;
      }
    }
  }

  if (!targetClass)
    return;

  // Get all classes that directly extend the target class
  auto it = collector.childrenMap.find(targetClass);
  if (it != collector.childrenMap.end()) {
    for (const auto *childClass : it->second) {
      subtypes.push_back(buildTypeHierarchyItem(*childClass, sm, uri));
    }
  }
}
