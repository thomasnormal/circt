//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the VerilogServer class, the top-level coordinator for
// the CIRCT Verilog LSP server. It manages the lifetime and synchronization of
// all open Verilog source files and provides the high-level entry points for
// Language Server Protocol (LSP) requests.
//
// Responsibilities:
//   * Maintain a map of open documents (`VerilogTextFile` instances) indexed by
//     their URI filenames.
//   * Create, update, and remove documents in response to LSP notifications
//     such as `textDocument/didOpen`, `didChange`, and `didClose`.
//   * Route definition and reference queries to the appropriate
//     VerilogTextFile, which in turn delegates to its `VerilogDocument`.
//   * Ensure each document remains synchronized with its latest text version
//     and up-to-date semantic index.
//
// Internal structure:
//   - Uses an internal `Impl` struct (pImpl pattern) to encapsulate the
//     server’s file map and its shared `VerilogServerContext`.
//   - Each document is owned via a `std::unique_ptr<VerilogTextFile>`,
//     allowing clean removal and automatic teardown when the file is closed.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/LSP/Protocol.h"

#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

#include "VerilogServer.h"
#include "VerilogServerContext.h"
#include "VerilogTextFile.h"
#include "Workspace.h"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>

using namespace llvm::lsp;

struct circt::lsp::VerilogServer::Impl {
  explicit Impl(const VerilogServerOptions &options) : context(options) {
    // Initialize workspace with global settings from command line
    workspace.setGlobalLibDirs(options.libDirs);
    workspace.setGlobalIncludeDirs(options.extraSourceLocationDirs);
    workspace.addCommandFiles(options.commandFiles);
  }

  /// The files held by the server, mapped by their URI file name.
  llvm::StringMap<std::unique_ptr<VerilogTextFile>> files;

  VerilogServerContext context;

  /// Workspace management for multi-root support.
  Workspace workspace;
};

circt::lsp::VerilogServer::VerilogServer(const VerilogServerOptions &options)
    : impl(std::make_unique<Impl>(options)) {}
circt::lsp::VerilogServer::~VerilogServer() = default;

void circt::lsp::VerilogServer::addDocument(
    const URIForFile &uri, StringRef contents, int64_t version,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {

  impl->files[uri.file()] = std::make_unique<VerilogTextFile>(
      impl->context, uri, contents, version, diagnostics);
}

void circt::lsp::VerilogServer::updateDocument(
    const URIForFile &uri,
    ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
    int64_t version, std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  // Check that we actually have a document for this uri.
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return;

  // Try to update the document. If we fail, erase the file from the server. A
  // failed updated generally means we've fallen out of sync somewhere.
  if (failed(it->second->update(uri, version, changes, diagnostics)))
    impl->files.erase(it);
}

std::optional<int64_t>
circt::lsp::VerilogServer::removeDocument(const URIForFile &uri) {
  auto it = impl->files.find(uri.file());
  if (it == impl->files.end())
    return std::nullopt;

  int64_t version = it->second->getVersion();
  impl->files.erase(it);
  return version;
}

void circt::lsp::VerilogServer::getLocationsOf(
    const URIForFile &uri, const Position &defPos,
    std::vector<llvm::lsp::Location> &locations) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getLocationsOf(uri, defPos, locations);
}

void circt::lsp::VerilogServer::findReferencesOf(
    const URIForFile &uri, const Position &pos, bool includeDeclaration,
    std::vector<llvm::lsp::Location> &references) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->findReferencesOf(uri, pos, includeDeclaration, references);
}

void circt::lsp::VerilogServer::getDocumentHighlights(
    const URIForFile &uri, const Position &pos,
    std::vector<DocumentHighlight> &highlights) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end()) {
    // Get highlights from VerilogDocument and convert to VerilogServer type
    std::vector<VerilogDocument::DocumentHighlight> docHighlights;
    fileIt->second->getDocumentHighlights(uri, pos, docHighlights);

    // Convert from VerilogDocument::DocumentHighlight to VerilogServer::DocumentHighlight
    for (const auto &dh : docHighlights) {
      DocumentHighlight highlight;
      highlight.range = dh.range;
      highlight.kind = static_cast<DocumentHighlightKind>(
          static_cast<int>(dh.kind));
      highlights.push_back(highlight);
    }
  }
}

std::optional<llvm::lsp::Hover>
circt::lsp::VerilogServer::getHover(const URIForFile &uri, const Position &pos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getHover(uri, pos);
  return std::nullopt;
}

void circt::lsp::VerilogServer::getDocumentSymbols(
    const URIForFile &uri, std::vector<llvm::lsp::DocumentSymbol> &symbols) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getDocumentSymbols(uri, symbols);
}

/// Compute a fuzzy match score for how well a name matches a query.
/// Returns -1 if no match, otherwise a score where higher is better.
/// The algorithm supports:
/// - Exact prefix matches (highest score)
/// - Substring matches
/// - Fuzzy subsequence matches (e.g., "abc" matches "AlphaBetaClass")
/// - CamelCase boundary matching (e.g., "ABC" matches "AlphaBetaClass")
static int computeFuzzyMatchScore(llvm::StringRef name, llvm::StringRef query) {
  if (query.empty())
    return 1000; // Empty query matches everything with high score
  if (name.empty())
    return -1;

  std::string nameLower = name.lower();
  std::string queryLower = query.lower();

  // Exact match gets highest score
  if (nameLower == queryLower)
    return 10000;

  // Exact prefix match gets very high score
  if (llvm::StringRef(nameLower).starts_with(queryLower))
    return 5000 + static_cast<int>(query.size()) * 10;

  // Substring match
  size_t substrPos = nameLower.find(queryLower);
  if (substrPos != std::string::npos) {
    // Earlier positions are better, shorter names are better
    int positionBonus = 100 - static_cast<int>(substrPos);
    int lengthBonus = 100 - static_cast<int>(name.size());
    return 2000 + positionBonus + lengthBonus;
  }

  // Fuzzy subsequence match - all query chars must appear in order
  size_t qi = 0; // Query index
  size_t ni = 0; // Name index
  int score = 0;
  bool lastWasBoundary = true; // Start counts as a boundary

  while (qi < queryLower.size() && ni < nameLower.size()) {
    char qc = queryLower[qi];
    char nc = nameLower[ni];

    if (qc == nc) {
      // Match found
      if (lastWasBoundary) {
        // CamelCase or underscore boundary match - bonus points
        score += 20;
      } else {
        // Consecutive match - small bonus
        score += (qi > 0 && static_cast<size_t>(qi - 1) < ni) ? 15 : 5;
      }
      qi++;
    }

    // Track CamelCase boundaries and underscore boundaries
    if (ni + 1 < name.size()) {
      char nextChar = name[ni + 1];
      char currChar = name[ni];
      lastWasBoundary =
          currChar == '_' ||
          (std::islower(static_cast<unsigned char>(currChar)) &&
           std::isupper(static_cast<unsigned char>(nextChar)));
    }
    ni++;
  }

  // All query characters must be found
  if (qi < queryLower.size())
    return -1;

  // Base score for fuzzy match, plus computed bonus
  return 500 + score - static_cast<int>(name.size()); // Prefer shorter names
}

static void collectWorkspaceSymbols(
    const llvm::lsp::URIForFile &uri, llvm::StringRef query,
    llvm::StringRef containerName,
    llvm::ArrayRef<llvm::lsp::DocumentSymbol> symbols,
    std::vector<circt::lsp::WorkspaceSymbol> &out) {
  for (const auto &symbol : symbols) {
    int score = computeFuzzyMatchScore(symbol.name, query);
    if (score >= 0) {
      circt::lsp::WorkspaceSymbol entry;
      entry.name = symbol.name;
      entry.kind = symbol.kind;
      entry.location = llvm::lsp::Location(uri, symbol.range);
      entry.containerName = containerName.str();
      entry.score = score;
      out.push_back(std::move(entry));
    }
    if (!symbol.children.empty())
      collectWorkspaceSymbols(uri, query, symbol.name, symbol.children, out);
  }
}

void circt::lsp::VerilogServer::getWorkspaceSymbols(
    llvm::StringRef query, std::vector<WorkspaceSymbol> &symbols) {
  llvm::StringSet<> seen;

  // Collect symbols from open documents (these have better semantic info)
  for (auto &entry : impl->files) {
    const auto &filePath = entry.first();
    auto uriOrErr = llvm::lsp::URIForFile::fromFile(filePath);
    if (!uriOrErr)
      continue;
    std::vector<llvm::lsp::DocumentSymbol> docSymbols;
    entry.second->getDocumentSymbols(*uriOrErr, docSymbols);
    collectWorkspaceSymbols(*uriOrErr, query, "", docSymbols, symbols);

    for (const auto &symbol : symbols) {
      std::string key = symbol.location.uri.file().str();
      key.append(":");
      key.append(symbol.name);
      key.append(":");
      key.append(std::to_string(symbol.location.range.start.line));
      key.append(":");
      key.append(std::to_string(symbol.location.range.start.character));
      seen.insert(std::move(key));
    }
  }

  // Collect symbols from workspace files (regex-based scanning)
  for (const auto &entry : impl->workspace.findAllSymbols()) {
    int score = computeFuzzyMatchScore(entry.name, query);
    if (score < 0)
      continue;
    auto uriOrErr = llvm::lsp::URIForFile::fromFile(entry.filePath);
    if (!uriOrErr)
      continue;
    WorkspaceSymbol symbol;
    symbol.name = entry.name;
    symbol.kind = entry.kind;
    symbol.location = llvm::lsp::Location(*uriOrErr, entry.range);
    symbol.score = score;

    std::string key = symbol.location.uri.file().str();
    key.append(":");
    key.append(symbol.name);
    key.append(":");
    key.append(std::to_string(symbol.location.range.start.line));
    key.append(":");
    key.append(std::to_string(symbol.location.range.start.character));
    if (seen.insert(key).second)
      symbols.push_back(std::move(symbol));
  }

  // Sort by score (descending) for best matches first
  std::stable_sort(symbols.begin(), symbols.end(),
                   [](const WorkspaceSymbol &a, const WorkspaceSymbol &b) {
                     return a.score > b.score;
                   });

  // Limit to reasonable number of results
  constexpr size_t maxResults = 100;
  if (symbols.size() > maxResults)
    symbols.resize(maxResults);
}

void circt::lsp::VerilogServer::getCompletions(
    const URIForFile &uri, const Position &pos,
    llvm::lsp::CompletionList &completions) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getCompletions(uri, pos, completions);
}

void circt::lsp::VerilogServer::getCodeActions(
    const URIForFile &uri, const llvm::lsp::Range &range,
    const std::vector<llvm::lsp::Diagnostic> &diagnostics,
    std::vector<llvm::lsp::CodeAction> &codeActions) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getCodeActions(uri, range, diagnostics, codeActions);
}

std::optional<std::pair<llvm::lsp::Range, std::string>>
circt::lsp::VerilogServer::prepareRename(const URIForFile &uri,
                                          const Position &pos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->prepareRename(uri, pos);
  return std::nullopt;
}

std::optional<llvm::lsp::WorkspaceEdit>
circt::lsp::VerilogServer::renameSymbol(const URIForFile &uri,
                                         const Position &pos,
                                         llvm::StringRef newName) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->renameSymbol(uri, pos, newName);
  return std::nullopt;
}

void circt::lsp::VerilogServer::getDocumentLinks(
    const URIForFile &uri, std::vector<llvm::lsp::DocumentLink> &links) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getDocumentLinks(uri, links);
}

void circt::lsp::VerilogServer::getSemanticTokens(
    const URIForFile &uri, std::vector<uint32_t> &data) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getSemanticTokens(uri, data);
}

void circt::lsp::VerilogServer::getInlayHints(
    const URIForFile &uri, const llvm::lsp::Range &range,
    std::vector<llvm::lsp::InlayHint> &hints) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    fileIt->second->getInlayHints(uri, range, hints);
}

llvm::lsp::SignatureHelp
circt::lsp::VerilogServer::getSignatureHelp(const URIForFile &uri,
                                            const Position &pos) {
  auto fileIt = impl->files.find(uri.file());
  if (fileIt != impl->files.end())
    return fileIt->second->getSignatureHelp(uri, pos);
  return llvm::lsp::SignatureHelp();
}

//===----------------------------------------------------------------------===//
// Workspace Management
//===----------------------------------------------------------------------===//

void circt::lsp::VerilogServer::initializeWorkspace(
    const llvm::json::Value &initParams) {
  impl->workspace.initializeFromParams(initParams);
}

void circt::lsp::VerilogServer::workspaceFoldersChanged(
    llvm::ArrayRef<std::string> added, llvm::ArrayRef<std::string> removed) {
  for (const auto &path : removed)
    impl->workspace.removeRoot(path);
  for (const auto &path : added)
    impl->workspace.addRoot(path);
}

void circt::lsp::VerilogServer::onFileChanged(const URIForFile &uri) {
  impl->workspace.onFileChanged(uri.file());
}

llvm::json::Value circt::lsp::VerilogServer::getWorkspaceConfiguration() const {
  llvm::json::Array roots;

  for (const auto &root : impl->workspace.getRoots()) {
    llvm::json::Object rootObj;
    rootObj["path"] = root->getRootPath();

    if (const auto *config = root->getProjectConfig()) {
      llvm::json::Object projectInfo;
      const auto &info = config->getProjectInfo();
      if (!info.name.empty())
        projectInfo["name"] = info.name;
      if (!info.topModule.empty())
        projectInfo["top"] = info.topModule;
      if (!info.version.empty())
        projectInfo["version"] = info.version;
      rootObj["project"] = std::move(projectInfo);

      llvm::json::Object lintInfo;
      lintInfo["enabled"] = config->getLintingConfig().enabled;
      if (!config->getLintingConfig().configFile.empty())
        lintInfo["configFile"] = config->getLintingConfig().configFile;
      rootObj["lint"] = std::move(lintInfo);
    }

    roots.push_back(std::move(rootObj));
  }

  return llvm::json::Object{{"workspaceRoots", std::move(roots)}};
}
