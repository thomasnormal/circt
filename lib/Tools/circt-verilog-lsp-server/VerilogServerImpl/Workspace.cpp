//===- Workspace.cpp - LSP Workspace management ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Workspace.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace circt::lsp;

//===----------------------------------------------------------------------===//
// WorkspaceRoot Implementation
//===----------------------------------------------------------------------===//

WorkspaceRoot::WorkspaceRoot(llvm::StringRef rootPath) : rootPath(rootPath) {
  // Normalize the path
  llvm::SmallString<256> absPath(rootPath);
  llvm::sys::fs::make_absolute(absPath);
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);
  this->rootPath = std::string(absPath);

  // Create URI for this root
  auto uriOrErr = llvm::lsp::URIForFile::fromURI("file://" + this->rootPath);
  if (uriOrErr)
    rootUri = std::move(*uriOrErr);

  // Try to load project configuration
  reloadConfiguration();
}

bool WorkspaceRoot::containsFile(llvm::StringRef filePath) const {
  llvm::SmallString<256> absPath(filePath);
  llvm::sys::fs::make_absolute(absPath);
  llvm::sys::path::remove_dots(absPath, /*remove_dot_dot=*/true);

  return llvm::StringRef(absPath).starts_with(rootPath);
}

void WorkspaceRoot::reloadConfiguration() {
  auto configOrErr = ProjectConfig::findAndLoad(rootPath);
  if (configOrErr) {
    projectConfig = std::move(*configOrErr);
  } else {
    // No configuration file found - use defaults
    llvm::consumeError(configOrErr.takeError());
    projectConfig = ProjectConfig::createDefault();
    projectConfig->setRootDirectory(rootPath);
  }
}

std::vector<std::string> WorkspaceRoot::getIncludeDirs() const {
  if (!projectConfig)
    return {};

  return projectConfig->resolveIncludeDirs();
}

std::vector<std::string> WorkspaceRoot::getDefines() const {
  if (!projectConfig)
    return {};

  return projectConfig->getAllDefines();
}

llvm::Expected<std::vector<std::string>> WorkspaceRoot::getSourceFiles() const {
  if (!projectConfig)
    return std::vector<std::string>();

  return projectConfig->resolveSourceFiles();
}

bool WorkspaceRoot::isLintingEnabled() const {
  if (!projectConfig)
    return true; // Default to enabled

  return projectConfig->getLintingConfig().enabled;
}

std::string WorkspaceRoot::getLintConfigPath() const {
  if (!projectConfig)
    return "";

  const auto &lintConfig = projectConfig->getLintingConfig();
  if (lintConfig.configFile.empty())
    return "";

  // Resolve relative to workspace root
  if (llvm::sys::path::is_absolute(lintConfig.configFile))
    return lintConfig.configFile;

  llvm::SmallString<256> resolved(rootPath);
  llvm::sys::path::append(resolved, lintConfig.configFile);
  return std::string(resolved);
}

//===----------------------------------------------------------------------===//
// Workspace Implementation
//===----------------------------------------------------------------------===//

Workspace::Workspace() = default;
Workspace::~Workspace() = default;

void Workspace::addRoot(llvm::StringRef rootPath) {
  std::scoped_lock<std::mutex> lock(mutex);

  // Check if this root already exists
  for (const auto &root : roots) {
    if (root->getRootPath() == rootPath)
      return;
  }

  roots.push_back(std::make_unique<WorkspaceRoot>(rootPath));
}

void Workspace::initializeFromParams(const llvm::json::Value &initParams) {
  std::scoped_lock<std::mutex> lock(mutex);

  const auto *paramsObj = initParams.getAsObject();
  if (!paramsObj)
    return;

  // Check for workspaceFolders first (multi-root workspace)
  if (const auto *foldersVal = paramsObj->get("workspaceFolders")) {
    if (const auto *folders = foldersVal->getAsArray()) {
      for (const auto &folder : *folders) {
        if (const auto *folderObj = folder.getAsObject()) {
          if (const auto *uriVal = folderObj->get("uri")) {
            if (auto uriStr = uriVal->getAsString()) {
              // Parse file:// URI to path
              if (uriStr->starts_with("file://")) {
                llvm::StringRef path = uriStr->substr(7);
                roots.push_back(std::make_unique<WorkspaceRoot>(path));
              }
            }
          }
        }
      }
    }
  }

  // Fall back to rootUri or rootPath (single root workspace)
  if (roots.empty()) {
    if (const auto *rootUriVal = paramsObj->get("rootUri")) {
      if (auto rootUri = rootUriVal->getAsString()) {
        if (rootUri->starts_with("file://")) {
          llvm::StringRef path = rootUri->substr(7);
          roots.push_back(std::make_unique<WorkspaceRoot>(path));
        }
      }
    } else if (const auto *rootPathVal = paramsObj->get("rootPath")) {
      if (auto rootPath = rootPathVal->getAsString()) {
        roots.push_back(std::make_unique<WorkspaceRoot>(*rootPath));
      }
    }
  }
}

void Workspace::removeRoot(llvm::StringRef rootPath) {
  std::scoped_lock<std::mutex> lock(mutex);

  roots.erase(std::remove_if(roots.begin(), roots.end(),
                             [&](const std::unique_ptr<WorkspaceRoot> &root) {
                               return root->getRootPath() == rootPath;
                             }),
              roots.end());
}

const WorkspaceRoot *
Workspace::getRootForFile(llvm::StringRef filePath) const {
  std::scoped_lock<std::mutex> lock(mutex);

  // Find the most specific root that contains this file
  const WorkspaceRoot *bestMatch = nullptr;
  size_t bestMatchLen = 0;

  for (const auto &root : roots) {
    if (root->containsFile(filePath)) {
      size_t len = root->getRootPath().size();
      if (len > bestMatchLen) {
        bestMatch = root.get();
        bestMatchLen = len;
      }
    }
  }

  return bestMatch;
}

std::vector<std::string>
Workspace::getIncludeDirs(llvm::StringRef filePath) const {
  std::vector<std::string> dirs;

  // Start with global directories
  dirs.insert(dirs.end(), globalIncludeDirs.begin(), globalIncludeDirs.end());

  // Add workspace-specific directories
  if (const auto *root = getRootForFile(filePath)) {
    auto workspaceDirs = root->getIncludeDirs();
    dirs.insert(dirs.end(), workspaceDirs.begin(), workspaceDirs.end());
  }

  return dirs;
}

std::vector<std::string>
Workspace::getDefines(llvm::StringRef filePath) const {
  std::vector<std::string> defines;

  // Start with global defines
  defines.insert(defines.end(), globalDefines.begin(), globalDefines.end());

  // Add workspace-specific defines
  if (const auto *root = getRootForFile(filePath)) {
    auto workspaceDefines = root->getDefines();
    defines.insert(defines.end(), workspaceDefines.begin(),
                   workspaceDefines.end());
  }

  return defines;
}

bool Workspace::isLintingEnabled(llvm::StringRef filePath) const {
  if (const auto *root = getRootForFile(filePath))
    return root->isLintingEnabled();

  return true; // Default to enabled
}

void Workspace::setGlobalIncludeDirs(llvm::ArrayRef<std::string> dirs) {
  std::scoped_lock<std::mutex> lock(mutex);
  globalIncludeDirs.assign(dirs.begin(), dirs.end());
}

void Workspace::setGlobalDefines(llvm::ArrayRef<std::string> defines) {
  std::scoped_lock<std::mutex> lock(mutex);
  globalDefines.assign(defines.begin(), defines.end());
}

void Workspace::setGlobalLibDirs(llvm::ArrayRef<std::string> dirs) {
  std::scoped_lock<std::mutex> lock(mutex);
  globalLibDirs.assign(dirs.begin(), dirs.end());
}

void Workspace::addCommandFiles(llvm::ArrayRef<std::string> files) {
  std::scoped_lock<std::mutex> lock(mutex);

  for (const auto &file : files) {
    auto entriesOrErr = ProjectConfig::parseFileList(file);
    if (!entriesOrErr) {
      llvm::consumeError(entriesOrErr.takeError());
      continue;
    }

    for (const auto &entry : *entriesOrErr) {
      switch (entry.kind) {
      case FileListEntry::Kind::IncludeDir:
        globalIncludeDirs.push_back(entry.value);
        break;
      case FileListEntry::Kind::Define:
        globalDefines.push_back(formatDefine(entry.value, entry.extra));
        break;
      case FileListEntry::Kind::LibDir:
        globalLibDirs.push_back(entry.value);
        break;
      default:
        break;
      }
    }
  }
}

void Workspace::onFileChanged(llvm::StringRef filePath) {
  std::scoped_lock<std::mutex> lock(mutex);

  // Check if this is a project configuration file
  if (isProjectConfigFile(filePath)) {
    // Reload configuration for the affected workspace
    for (auto &root : roots) {
      if (root->containsFile(filePath)) {
        root->reloadConfiguration();
        break;
      }
    }
  }
}

std::vector<std::string> Workspace::getFilesToWatch() const {
  std::scoped_lock<std::mutex> lock(mutex);

  std::vector<std::string> files;

  // Watch project configuration files
  for (const auto &root : roots) {
    for (const auto &name : getProjectConfigFileNames()) {
      llvm::SmallString<256> path(root->getRootPath());
      llvm::sys::path::append(path, name);
      files.push_back(std::string(path));
    }
  }

  return files;
}

llvm::Expected<std::vector<std::string>> Workspace::getAllSourceFiles() const {
  std::scoped_lock<std::mutex> lock(mutex);

  std::vector<std::string> allFiles;

  for (const auto &root : roots) {
    auto filesOrErr = root->getSourceFiles();
    if (!filesOrErr)
      return filesOrErr.takeError();

    allFiles.insert(allFiles.end(), filesOrErr->begin(), filesOrErr->end());
  }

  return allFiles;
}

std::vector<std::pair<std::string, std::string>>
Workspace::findAllModules() const {
  std::scoped_lock<std::mutex> lock(mutex);

  std::vector<std::pair<std::string, std::string>> modules;

  // This would require parsing all source files to extract module names
  // For now, return empty - this can be populated as files are opened

  return modules;
}
