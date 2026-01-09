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
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/ParameterSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/SubroutineSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/syntax/AllSyntax.h"
#include "slang/syntax/SyntaxTree.h"

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
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

using namespace circt::lsp;
using namespace llvm;
using namespace llvm::lsp;

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

  computeLineOffsets(driver.sourceManager.getSourceText(mainBufferId));

  index = std::make_unique<VerilogIndex>(mainBufferId, driver.sourceManager);
  // Populate the index.
  index->initialize(**compilation);
}

llvm::lsp::Location
VerilogDocument::getLspLocation(slang::SourceLocation loc) const {
  if (loc && loc.buffer() != slang::SourceLocation::NoLocation.buffer()) {
    const auto &slangSourceManager = getSlangSourceManager();
    auto line = slangSourceManager.getLineNumber(loc) - 1;
    auto column = slangSourceManager.getColumnNumber(loc) - 1;
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
    std::vector<llvm::lsp::Location> &references) {

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

  auto it = index->getReferences().find(symbol);
  if (it == index->getReferences().end())
    return;
  for (auto referenceRange : it->second)
    references.push_back(getLspLocation(referenceRange));
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
      auto cv = initExpr->eval(slang::ast::ASTContext(
          symbol.getParentScope()->asSymbol(),
          slang::ast::LookupLocation::max));
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
  case SK::ClassMethod:
    return LK::Class;
  case SK::InterfacePort:
    return LK::Interface;
  case SK::Enum:
  case SK::TypeAlias:
    return LK::Enum;
  case SK::Struct:
    return LK::Struct;
  default:
    return LK::Variable;
  }
}

/// Calculate LSP Range from slang source range.
llvm::lsp::Range VerilogDocument::getLspRange(slang::SourceRange range) const {
  const auto &sm = getSlangSourceManager();
  int startLine = sm.getLineNumber(range.start()) - 1;
  int startCol = sm.getColumnNumber(range.start()) - 1;
  int endLine = sm.getLineNumber(range.end()) - 1;
  int endCol = sm.getColumnNumber(range.end()) - 1;
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
    int nameLine = sm.getLineNumber(body.location) - 1;
    int nameCol = sm.getColumnNumber(body.location) - 1;
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

        int pLine = sm.getLineNumber(port->location) - 1;
        int pCol = sm.getColumnNumber(port->location) - 1;
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
          auto cv = initExpr->eval(slang::ast::ASTContext(
              body, slang::ast::LookupLocation::max));
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
      default:
        continue; // Skip other member types
      }

      if (member.name.empty())
        continue;

      int mLine = sm.getLineNumber(member.location) - 1;
      int mCol = sm.getColumnNumber(member.location) - 1;
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
    int nameLine = sm.getLineNumber(pkg.location) - 1;
    int nameCol = sm.getColumnNumber(pkg.location) - 1;
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
      int mLine = sm.getLineNumber(member.location) - 1;
      int mCol = sm.getColumnNumber(member.location) - 1;
      llvm::lsp::Range mRange(llvm::lsp::Position(mLine, mCol),
                              llvm::lsp::Position(mLine, mCol + member.name.size()));

      llvm::lsp::DocumentSymbol memberSym(member.name, kind, mRange, mRange);
      children.push_back(std::move(memberSym));
    }

    pkgSym.children = std::move(children);
    symbols.push_back(std::move(pkgSym));
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

  // Visit top instances
  for (auto *inst : root.topInstances) {
    if (inst->body.location.buffer() != mainBufferId)
      continue;
    visitor.visit(inst->body);
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
  case SK::ClassMethod:
    return CK::Class;
  case SK::InterfacePort:
    return CK::Interface;
  case SK::Enum:
  case SK::TypeAlias:
    return CK::Enum;
  case SK::EnumValue:
    return CK::EnumMember;
  case SK::Struct:
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
      auto cv = initExpr->eval(slang::ast::ASTContext(
          symbol.getParentScope()->asSymbol(), slang::ast::LookupLocation::max));
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
    for (const auto &def : (*compilation)->getDefinitions()) {
      if (prefix.empty() ||
          llvm::StringRef(def.name).starts_with_insensitive(prefix)) {
        llvm::lsp::CompletionItem item;
        item.label = std::string(def.name);
        item.kind = llvm::lsp::CompletionItemKind::Module;
        item.detail = "module definition";
        item.insertText = std::string(def.name);
        item.insertTextFormat = llvm::lsp::InsertTextFormat::PlainText;
        completions.items.push_back(std::move(item));

        // Also add an instantiation snippet
        llvm::lsp::CompletionItem instSnippet;
        instSnippet.label = std::string(def.name) + " (instantiate)";
        instSnippet.kind = llvm::lsp::CompletionItemKind::Snippet;
        instSnippet.detail = "Module instantiation";

        std::string snippet = std::string(def.name) + " ${1:inst_name} (\n";
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
         message.contains("definition of") && message.contains("not found");
}

/// Generate quick fix code actions based on diagnostics.
static void generateQuickFixes(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Range &range,
    const std::vector<llvm::lsp::Diagnostic> &diagnostics,
    std::vector<llvm::lsp::CodeAction> &codeActions) {

  for (const auto &diag : diagnostics) {
    // Check for unknown identifier - suggest adding wire/logic declaration
    if (isUnknownIdentifierDiagnostic(diag.message)) {
      // Extract identifier name from the diagnostic message if possible
      // For now, provide generic suggestions

      llvm::lsp::CodeAction wireAction;
      wireAction.title = "Declare as wire";
      wireAction.kind = llvm::lsp::CodeAction::kQuickFix;
      wireAction.diagnostics = {diag};

      // The actual text edit would need the identifier name
      // For now, just provide the action structure
      llvm::lsp::WorkspaceEdit edit;
      // Insert "wire <name>;" at the beginning of the module
      wireAction.edit = edit;
      codeActions.push_back(std::move(wireAction));

      llvm::lsp::CodeAction logicAction;
      logicAction.title = "Declare as logic";
      logicAction.kind = llvm::lsp::CodeAction::kQuickFix;
      logicAction.diagnostics = {diag};
      logicAction.edit = llvm::lsp::WorkspaceEdit{};
      codeActions.push_back(std::move(logicAction));
    }

    // Check for width mismatch
    if (isWidthMismatchDiagnostic(diag.message)) {
      llvm::lsp::CodeAction truncateAction;
      truncateAction.title = "Add explicit truncation";
      truncateAction.kind = llvm::lsp::CodeAction::kQuickFix;
      truncateAction.diagnostics = {diag};
      truncateAction.edit = llvm::lsp::WorkspaceEdit{};
      codeActions.push_back(std::move(truncateAction));

      llvm::lsp::CodeAction extendAction;
      extendAction.title = "Add explicit extension";
      extendAction.kind = llvm::lsp::CodeAction::kQuickFix;
      extendAction.diagnostics = {diag};
      extendAction.edit = llvm::lsp::WorkspaceEdit{};
      codeActions.push_back(std::move(extendAction));
    }

    // Check for unknown module
    if (isUnknownModuleDiagnostic(diag.message)) {
      llvm::lsp::CodeAction createModuleAction;
      createModuleAction.title = "Create module stub";
      createModuleAction.kind = llvm::lsp::CodeAction::kQuickFix;
      createModuleAction.diagnostics = {diag};
      createModuleAction.edit = llvm::lsp::WorkspaceEdit{};
      codeActions.push_back(std::move(createModuleAction));
    }
  }
}

void VerilogDocument::getCodeActions(
    const llvm::lsp::URIForFile &uri, const llvm::lsp::Range &range,
    const std::vector<llvm::lsp::Diagnostic> &diagnostics,
    std::vector<llvm::lsp::CodeAction> &codeActions) {

  // Generate quick fixes based on diagnostics
  generateQuickFixes(uri, range, diagnostics, codeActions);

  // Add refactoring actions that are always available
  // Extract selection as new signal
  llvm::lsp::CodeAction extractSignalAction;
  extractSignalAction.title = "Extract to signal";
  extractSignalAction.kind = llvm::lsp::CodeAction::kRefactor;
  codeActions.push_back(std::move(extractSignalAction));

  // Add module instantiation template action
  if (succeeded(compilation)) {
    for (const auto &def : (*compilation)->getDefinitions()) {
      llvm::lsp::CodeAction instantiateAction;
      instantiateAction.title = "Insert " + std::string(def.name) + " instance";
      instantiateAction.kind = llvm::lsp::CodeAction::kRefactor;

      // Build the instantiation template
      std::string instTemplate = std::string(def.name) + " inst_" +
                                 std::string(def.name) + " (\n";

      // Get the ports from the definition
      bool first = true;
      // Try to get ports from the first instance if available
      const auto &root = (*compilation)->getRoot();
      for (auto *inst : root.topInstances) {
        if (&inst->getDefinition() == &def) {
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
