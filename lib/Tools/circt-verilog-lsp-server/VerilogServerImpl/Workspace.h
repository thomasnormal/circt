//===- Workspace.h - LSP Workspace management -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the Workspace class for managing multi-root workspaces
// and project configurations in the CIRCT Verilog LSP server.
//
// A workspace represents one or more project roots that share a common
// configuration. Each root can have its own circt-project.yaml configuration.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_WORKSPACE_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_WORKSPACE_H_

#include "circt/Support/ProjectConfig.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LSP/Protocol.h"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace circt {
namespace lsp {

//===----------------------------------------------------------------------===//
// WorkspaceRoot
//===----------------------------------------------------------------------===//

/// Represents a single workspace root directory with its project configuration.
class WorkspaceRoot {
public:
  WorkspaceRoot(llvm::StringRef rootPath);

  /// Get the root path of this workspace.
  llvm::StringRef getRootPath() const { return rootPath; }

  /// Get the URI for this workspace root.
  const llvm::lsp::URIForFile &getRootUri() const { return rootUri; }

  /// Get the project configuration for this workspace root.
  /// Returns nullptr if no configuration file was found.
  const ProjectConfig *getProjectConfig() const { return projectConfig.get(); }

  /// Check if a file path belongs to this workspace root.
  bool containsFile(llvm::StringRef filePath) const;

  /// Reload the project configuration from disk.
  void reloadConfiguration();

  /// Get include directories for this workspace.
  std::vector<std::string> getIncludeDirs() const;

  /// Get preprocessor defines for this workspace.
  std::vector<std::string> getDefines() const;

  /// Get source files for this workspace (resolved from patterns).
  llvm::Expected<std::vector<std::string>> getSourceFiles() const;

  /// Check if linting is enabled for this workspace.
  bool isLintingEnabled() const;

  /// Get the lint configuration file path.
  std::string getLintConfigPath() const;

private:
  std::string rootPath;
  llvm::lsp::URIForFile rootUri;
  std::unique_ptr<ProjectConfig> projectConfig;
};

//===----------------------------------------------------------------------===//
// Workspace
//===----------------------------------------------------------------------===//

/// Manages multiple workspace roots and their configurations.
class Workspace {
public:
  Workspace();
  ~Workspace();

  //===--------------------------------------------------------------------===//
  // Workspace Root Management
  //===--------------------------------------------------------------------===//

  /// Add a workspace root.
  void addRoot(llvm::StringRef rootPath);

  /// Add workspace roots from initialization parameters.
  void initializeFromParams(const llvm::json::Value &initParams);

  /// Remove a workspace root.
  void removeRoot(llvm::StringRef rootPath);

  /// Get the workspace root for a file.
  /// Returns nullptr if the file doesn't belong to any workspace root.
  const WorkspaceRoot *getRootForFile(llvm::StringRef filePath) const;

  /// Get all workspace roots.
  llvm::ArrayRef<std::unique_ptr<WorkspaceRoot>> getRoots() const {
    return roots;
  }

  /// Check if any workspace roots are configured.
  bool hasRoots() const { return !roots.empty(); }

  //===--------------------------------------------------------------------===//
  // Configuration Access
  //===--------------------------------------------------------------------===//

  /// Get include directories for a file.
  /// This returns the directories from the workspace containing the file,
  /// plus any global include directories.
  std::vector<std::string> getIncludeDirs(llvm::StringRef filePath) const;

  /// Get preprocessor defines for a file.
  std::vector<std::string> getDefines(llvm::StringRef filePath) const;

  /// Check if linting is enabled for a file.
  bool isLintingEnabled(llvm::StringRef filePath) const;

  //===--------------------------------------------------------------------===//
  // Global Configuration
  //===--------------------------------------------------------------------===//

  /// Set global include directories (from command line).
  void setGlobalIncludeDirs(llvm::ArrayRef<std::string> dirs);

  /// Set global defines (from command line).
  void setGlobalDefines(llvm::ArrayRef<std::string> defines);

  /// Set global library directories (from command line).
  void setGlobalLibDirs(llvm::ArrayRef<std::string> dirs);

  /// Add command files to parse.
  void addCommandFiles(llvm::ArrayRef<std::string> files);

  //===--------------------------------------------------------------------===//
  // File Watching
  //===--------------------------------------------------------------------===//

  /// Handle a file change notification.
  void onFileChanged(llvm::StringRef filePath);

  /// Get files that should be watched for changes.
  std::vector<std::string> getFilesToWatch() const;

  //===--------------------------------------------------------------------===//
  // Project-Wide Operations
  //===--------------------------------------------------------------------===//

  /// Find all source files across all workspaces.
  llvm::Expected<std::vector<std::string>> getAllSourceFiles() const;

  /// Find all modules across all workspaces.
  /// Returns a map from module name to file path.
  std::vector<std::pair<std::string, std::string>> findAllModules() const;

private:
  /// The workspace roots.
  llvm::SmallVector<std::unique_ptr<WorkspaceRoot>> roots;

  /// Global include directories (from command line).
  std::vector<std::string> globalIncludeDirs;

  /// Global defines (from command line).
  std::vector<std::string> globalDefines;

  /// Global library directories (from command line).
  std::vector<std::string> globalLibDirs;

  /// Mutex for thread-safe access.
  mutable std::mutex mutex;
};

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_WORKSPACE_H_
