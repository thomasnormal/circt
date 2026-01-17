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

static bool matchesWorkspaceQuery(llvm::StringRef name,
                                  llvm::StringRef query) {
  if (query.empty())
    return true;
  std::string nameLower = name.lower();
  std::string queryLower = query.lower();
  return llvm::StringRef(nameLower).contains(queryLower);
}

static void collectWorkspaceSymbols(
    const llvm::lsp::URIForFile &uri, llvm::StringRef query,
    llvm::StringRef containerName,
    llvm::ArrayRef<llvm::lsp::DocumentSymbol> symbols,
    std::vector<circt::lsp::WorkspaceSymbol> &out) {
  for (const auto &symbol : symbols) {
    if (matchesWorkspaceQuery(symbol.name, query)) {
      circt::lsp::WorkspaceSymbol entry;
      entry.name = symbol.name;
      entry.kind = symbol.kind;
      entry.location = llvm::lsp::Location(uri, symbol.range);
      entry.containerName = containerName.str();
      out.push_back(std::move(entry));
    }
    if (!symbol.children.empty())
      collectWorkspaceSymbols(uri, query, symbol.name, symbol.children, out);
  }
}

void circt::lsp::VerilogServer::getWorkspaceSymbols(
    llvm::StringRef query, std::vector<WorkspaceSymbol> &symbols) {
  llvm::StringSet<> seen;

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

  for (const auto &entry : impl->workspace.findAllSymbols()) {
    if (!matchesWorkspaceQuery(entry.name, query))
      continue;
    auto uriOrErr = llvm::lsp::URIForFile::fromFile(entry.filePath);
    if (!uriOrErr)
      continue;
    WorkspaceSymbol symbol;
    symbol.name = entry.name;
    symbol.kind = entry.kind;
    symbol.location = llvm::lsp::Location(*uriOrErr, entry.range);

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
