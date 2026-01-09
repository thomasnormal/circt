//===- ProjectConfig.h - Project configuration support --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the ProjectConfig class for loading and managing CIRCT
// project configuration from YAML files (circt-project.yaml).
//
// Example configuration:
//
// ```yaml
// project:
//   name: "my_design"
//   top: "top_module"
//   version: "1.0.0"
//
// sources:
//   include_dirs:
//     - "rtl/"
//     - "includes/"
//   defines:
//     - "SYNTHESIS"
//     - "DEBUG_LEVEL=2"
//   files:
//     - "rtl/**/*.sv"
//     - "tb/**/*.sv"
//   file_lists:
//     - "project.f"
//
// lint:
//   enabled: true
//   config: "lint.yaml"
//
// simulation:
//   timescale: "1ns/1ps"
//   default_clocking: "clock"
//
// targets:
//   synthesis:
//     top: "chip_top"
//     defines: ["SYNTHESIS"]
//   simulation:
//     top: "tb_top"
//     defines: ["SIMULATION", "VCS"]
// ```
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PROJECTCONFIG_H
#define CIRCT_SUPPORT_PROJECTCONFIG_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace circt {

//===----------------------------------------------------------------------===//
// Project Info
//===----------------------------------------------------------------------===//

/// Basic project information.
struct ProjectInfo {
  /// Project name.
  std::string name;

  /// Top-level module name.
  std::string topModule;

  /// Project version string.
  std::string version;

  /// Project description.
  std::string description;
};

//===----------------------------------------------------------------------===//
// Source Configuration
//===----------------------------------------------------------------------===//

/// Configuration for source files and directories.
struct SourceConfig {
  /// Include directories for module lookup.
  std::vector<std::string> includeDirs;

  /// Preprocessor defines (VAR or VAR=VALUE).
  std::vector<std::string> defines;

  /// Source file patterns (supports glob patterns).
  std::vector<std::string> files;

  /// File list files (.f files) to parse.
  std::vector<std::string> fileLists;

  /// Library directories for library cell lookup.
  std::vector<std::string> libDirs;

  /// Library file patterns.
  std::vector<std::string> libFiles;

  /// File extensions to consider as SystemVerilog (default: .sv, .svh).
  std::vector<std::string> svExtensions;

  /// File extensions to consider as Verilog (default: .v, .vh).
  std::vector<std::string> verilogExtensions;
};

//===----------------------------------------------------------------------===//
// Lint Configuration
//===----------------------------------------------------------------------===//

/// Configuration for linting.
struct LintingConfig {
  /// Whether linting is enabled.
  bool enabled = true;

  /// Path to lint configuration file.
  std::string configFile;

  /// Rules to enable (in addition to config file).
  std::vector<std::string> enableRules;

  /// Rules to disable (overrides config file and enableRules).
  std::vector<std::string> disableRules;

  /// Files or patterns to exclude from linting.
  std::vector<std::string> excludePatterns;
};

//===----------------------------------------------------------------------===//
// Simulation Configuration
//===----------------------------------------------------------------------===//

/// Configuration for simulation.
struct SimulationConfig {
  /// Default timescale (e.g., "1ns/1ps").
  std::string timescale;

  /// Default clocking signal name.
  std::string defaultClocking;

  /// Default reset signal name.
  std::string defaultReset;

  /// Reset polarity (true = active high, false = active low).
  bool resetActiveHigh = true;

  /// Simulation-specific defines.
  std::vector<std::string> defines;
};

//===----------------------------------------------------------------------===//
// Target Configuration
//===----------------------------------------------------------------------===//

/// Configuration for a specific build target.
struct TargetConfig {
  /// Target name.
  std::string name;

  /// Top-level module override.
  std::string topModule;

  /// Additional defines for this target.
  std::vector<std::string> defines;

  /// Additional include directories.
  std::vector<std::string> includeDirs;

  /// Additional source files.
  std::vector<std::string> files;

  /// Whether linting is enabled for this target.
  std::optional<bool> lintEnabled;

  /// Output directory for this target.
  std::string outputDir;
};

//===----------------------------------------------------------------------===//
// File List Entry
//===----------------------------------------------------------------------===//

/// Represents a parsed entry from a file list (.f file).
struct FileListEntry {
  enum class Kind {
    File,       /// A source file path
    IncludeDir, /// +incdir+<path>
    Define,     /// +define+<macro> or +define+<macro>=<value>
    FileList,   /// -f <file>
    LibDir,     /// -y <dir>
    LibExt,     /// +libext+<ext>
    TopModule,  /// -top <module>
    Other       /// Unrecognized flag
  };

  Kind kind = Kind::File;
  std::string value;
  std::string extra; // For defines: the value part
};

//===----------------------------------------------------------------------===//
// ProjectConfig Class
//===----------------------------------------------------------------------===//

/// Main project configuration class.
class ProjectConfig {
public:
  ProjectConfig();
  ~ProjectConfig();

  //===--------------------------------------------------------------------===//
  // Loading Methods
  //===--------------------------------------------------------------------===//

  /// Load configuration from a YAML file.
  static llvm::Expected<std::unique_ptr<ProjectConfig>>
  loadFromFile(llvm::StringRef filePath);

  /// Load configuration from a YAML string.
  static llvm::Expected<std::unique_ptr<ProjectConfig>>
  loadFromYAML(llvm::StringRef yamlContent);

  /// Find and load project configuration from a directory.
  /// Searches for circt-project.yaml, .circt-project.yaml, circt.yaml.
  static llvm::Expected<std::unique_ptr<ProjectConfig>>
  findAndLoad(llvm::StringRef directory);

  /// Search parent directories for project configuration.
  static llvm::Expected<std::unique_ptr<ProjectConfig>>
  findAndLoadRecursive(llvm::StringRef startPath);

  //===--------------------------------------------------------------------===//
  // File List Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a file list (.f file).
  static llvm::Expected<std::vector<FileListEntry>>
  parseFileList(llvm::StringRef filePath);

  /// Parse file list content from a string.
  static llvm::Expected<std::vector<FileListEntry>>
  parseFileListContent(llvm::StringRef content, llvm::StringRef basePath = "");

  /// Expand a file list into this configuration.
  llvm::Error expandFileList(llvm::StringRef filePath);

  //===--------------------------------------------------------------------===//
  // Accessors
  //===--------------------------------------------------------------------===//

  /// Get project information.
  const ProjectInfo &getProjectInfo() const { return projectInfo; }

  /// Set project information.
  void setProjectInfo(const ProjectInfo &info) { projectInfo = info; }

  /// Get source configuration.
  const SourceConfig &getSourceConfig() const { return sourceConfig; }

  /// Set source configuration.
  void setSourceConfig(const SourceConfig &config) { sourceConfig = config; }

  /// Get linting configuration.
  const LintingConfig &getLintingConfig() const { return lintingConfig; }

  /// Set linting configuration.
  void setLintingConfig(const LintingConfig &config) { lintingConfig = config; }

  /// Get simulation configuration.
  const SimulationConfig &getSimulationConfig() const {
    return simulationConfig;
  }

  /// Set simulation configuration.
  void setSimulationConfig(const SimulationConfig &config) {
    simulationConfig = config;
  }

  /// Get a target configuration by name.
  const TargetConfig *getTarget(llvm::StringRef name) const;

  /// Get all target names.
  std::vector<std::string> getTargetNames() const;

  /// Add a target configuration.
  void addTarget(TargetConfig target);

  /// Get the project root directory.
  llvm::StringRef getRootDirectory() const { return rootDirectory; }

  /// Set the project root directory.
  void setRootDirectory(llvm::StringRef dir) { rootDirectory = dir.str(); }

  //===--------------------------------------------------------------------===//
  // Resolution Methods
  //===--------------------------------------------------------------------===//

  /// Resolve all file patterns to actual file paths.
  llvm::Expected<std::vector<std::string>> resolveSourceFiles() const;

  /// Resolve include directories to absolute paths.
  std::vector<std::string> resolveIncludeDirs() const;

  /// Get all defines (project + simulation merged).
  std::vector<std::string> getAllDefines() const;

  /// Get effective configuration for a target.
  /// Merges project defaults with target-specific settings.
  llvm::Expected<std::unique_ptr<ProjectConfig>>
  getEffectiveConfig(llvm::StringRef targetName) const;

  //===--------------------------------------------------------------------===//
  // Validation
  //===--------------------------------------------------------------------===//

  /// Validate the configuration.
  llvm::Error validate() const;

  /// Check if the configuration is empty/default.
  bool isEmpty() const;

  //===--------------------------------------------------------------------===//
  // Serialization
  //===--------------------------------------------------------------------===//

  /// Serialize the configuration to YAML.
  std::string toYAML() const;

  /// Create a default project configuration.
  static std::unique_ptr<ProjectConfig> createDefault();

private:
  ProjectInfo projectInfo;
  SourceConfig sourceConfig;
  LintingConfig lintingConfig;
  SimulationConfig simulationConfig;
  llvm::StringMap<TargetConfig> targets;
  std::string rootDirectory;

  /// Helper to expand glob patterns.
  llvm::Expected<std::vector<std::string>>
  expandGlob(llvm::StringRef pattern) const;

  /// Helper to resolve a path relative to project root.
  std::string resolvePath(llvm::StringRef path) const;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Get the list of recognized project configuration file names.
llvm::ArrayRef<llvm::StringRef> getProjectConfigFileNames();

/// Check if a file is a recognized project configuration file.
bool isProjectConfigFile(llvm::StringRef filename);

/// Parse a define string into name and optional value.
/// Returns (name, value) where value is empty if not present.
std::pair<std::string, std::string> parseDefine(llvm::StringRef define);

/// Format a define as a command-line argument.
std::string formatDefine(llvm::StringRef name, llvm::StringRef value = "");

} // namespace circt

#endif // CIRCT_SUPPORT_PROJECTCONFIG_H
