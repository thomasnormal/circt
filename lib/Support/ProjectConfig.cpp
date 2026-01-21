//===- ProjectConfig.cpp - Project configuration support ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ProjectConfig class for loading and managing CIRCT
// project configuration from YAML files.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ProjectConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GlobPattern.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Project Configuration File Names
//===----------------------------------------------------------------------===//

static const llvm::StringRef configFileNames[] = {
    "circt-project.yaml", ".circt-project.yaml", "circt.yaml", ".circt.yaml",
    "circt-project.yml",  ".circt-project.yml",  "circt.yml",  ".circt.yml"};

llvm::ArrayRef<llvm::StringRef> circt::getProjectConfigFileNames() {
  return configFileNames;
}

bool circt::isProjectConfigFile(llvm::StringRef filename) {
  llvm::StringRef basename = llvm::sys::path::filename(filename);
  for (const auto &name : configFileNames) {
    if (basename == name)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Define Parsing
//===----------------------------------------------------------------------===//

std::pair<std::string, std::string> circt::parseDefine(llvm::StringRef define) {
  auto pos = define.find('=');
  if (pos == llvm::StringRef::npos)
    return {define.str(), ""};
  return {define.substr(0, pos).str(), define.substr(pos + 1).str()};
}

std::string circt::formatDefine(llvm::StringRef name, llvm::StringRef value) {
  if (value.empty())
    return name.str();
  return (name + "=" + value).str();
}

//===----------------------------------------------------------------------===//
// YAML Parsing Helpers
//===----------------------------------------------------------------------===//

namespace {

/// Get scalar value from a YAML node.
llvm::StringRef getScalar(llvm::yaml::Node *node,
                          llvm::SmallVectorImpl<char> &storage) {
  if (auto *scalar = llvm::dyn_cast<llvm::yaml::ScalarNode>(node))
    return scalar->getValue(storage);
  return "";
}

/// Get boolean value from a YAML node.
bool getBool(llvm::yaml::Node *node) {
  llvm::SmallString<16> storage;
  auto val = getScalar(node, storage);
  return val == "true" || val == "yes" || val == "1" || val == "on";
}

/// Parse a string sequence from a YAML node.
void parseStringSequence(llvm::yaml::Node *node,
                         std::vector<std::string> &out) {
  if (auto *seq = llvm::dyn_cast<llvm::yaml::SequenceNode>(node)) {
    for (auto &item : *seq) {
      llvm::SmallString<128> storage;
      auto val = getScalar(&item, storage);
      if (!val.empty())
        out.push_back(val.str());
    }
  }
}

/// Parse a YAML mapping node with a callback for each key-value pair.
bool parseMapping(llvm::yaml::MappingNode *mapping,
                  std::function<bool(llvm::StringRef, llvm::yaml::Node *)> cb) {
  for (auto &entry : *mapping) {
    auto *keyNode = llvm::dyn_cast<llvm::yaml::ScalarNode>(entry.getKey());
    if (!keyNode)
      continue;

    llvm::SmallString<64> keyStorage;
    llvm::StringRef key = keyNode->getValue(keyStorage);

    if (!cb(key, entry.getValue()))
      return false;
  }
  return true;
}

/// Parse project info section.
void parseProjectInfo(llvm::yaml::MappingNode *node, ProjectInfo &info) {
  parseMapping(node, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    llvm::SmallString<128> storage;
    if (key == "name")
      info.name = getScalar(value, storage).str();
    else if (key == "top" || key == "top_module")
      info.topModule = getScalar(value, storage).str();
    else if (key == "version")
      info.version = getScalar(value, storage).str();
    else if (key == "description")
      info.description = getScalar(value, storage).str();
    return true;
  });
}

/// Parse source configuration section.
void parseSourceConfig(llvm::yaml::MappingNode *node, SourceConfig &config) {
  parseMapping(node, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    if (key == "include_dirs" || key == "includes")
      parseStringSequence(value, config.includeDirs);
    else if (key == "defines")
      parseStringSequence(value, config.defines);
    else if (key == "files" || key == "sources")
      parseStringSequence(value, config.files);
    else if (key == "file_lists" || key == "f_files")
      parseStringSequence(value, config.fileLists);
    else if (key == "lib_dirs" || key == "libraries")
      parseStringSequence(value, config.libDirs);
    else if (key == "lib_files")
      parseStringSequence(value, config.libFiles);
    else if (key == "sv_extensions")
      parseStringSequence(value, config.svExtensions);
    else if (key == "verilog_extensions")
      parseStringSequence(value, config.verilogExtensions);
    return true;
  });
}

/// Parse lint configuration section.
void parseLintConfig(llvm::yaml::MappingNode *node, LintingConfig &config) {
  parseMapping(node, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    llvm::SmallString<256> storage;
    if (key == "enabled")
      config.enabled = getBool(value);
    else if (key == "config" || key == "config_file")
      config.configFile = getScalar(value, storage).str();
    else if (key == "enable" || key == "enable_rules")
      parseStringSequence(value, config.enableRules);
    else if (key == "disable" || key == "disable_rules")
      parseStringSequence(value, config.disableRules);
    else if (key == "exclude" || key == "exclude_patterns")
      parseStringSequence(value, config.excludePatterns);
    return true;
  });
}

/// Parse simulation configuration section.
void parseSimulationConfig(llvm::yaml::MappingNode *node,
                           SimulationConfig &config) {
  parseMapping(node, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    llvm::SmallString<64> storage;
    if (key == "timescale")
      config.timescale = getScalar(value, storage).str();
    else if (key == "default_clocking" || key == "clock")
      config.defaultClocking = getScalar(value, storage).str();
    else if (key == "default_reset" || key == "reset")
      config.defaultReset = getScalar(value, storage).str();
    else if (key == "reset_active_high")
      config.resetActiveHigh = getBool(value);
    else if (key == "defines")
      parseStringSequence(value, config.defines);
    return true;
  });
}

/// Parse a target configuration.
TargetConfig parseTargetConfig(llvm::StringRef name,
                               llvm::yaml::MappingNode *node) {
  TargetConfig target;
  target.name = name.str();

  parseMapping(node, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    llvm::SmallString<256> storage;
    if (key == "top" || key == "top_module")
      target.topModule = getScalar(value, storage).str();
    else if (key == "defines")
      parseStringSequence(value, target.defines);
    else if (key == "include_dirs" || key == "includes")
      parseStringSequence(value, target.includeDirs);
    else if (key == "files" || key == "sources")
      parseStringSequence(value, target.files);
    else if (key == "lint" || key == "lint_enabled")
      target.lintEnabled = getBool(value);
    else if (key == "output_dir" || key == "output")
      target.outputDir = getScalar(value, storage).str();
    return true;
  });

  return target;
}

} // namespace

//===----------------------------------------------------------------------===//
// ProjectConfig Implementation
//===----------------------------------------------------------------------===//

ProjectConfig::ProjectConfig() = default;
ProjectConfig::~ProjectConfig() = default;

llvm::Expected<std::unique_ptr<ProjectConfig>>
ProjectConfig::loadFromFile(llvm::StringRef filePath) {
  auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (auto ec = fileOrErr.getError())
    return llvm::createStringError(ec, "failed to open project config file: %s",
                                   filePath.str().c_str());

  auto result = loadFromYAML((*fileOrErr)->getBuffer());
  if (!result)
    return result.takeError();

  // Set root directory to the directory containing the config file
  llvm::SmallString<256> absPath(filePath);
  llvm::sys::fs::make_absolute(absPath);
  (*result)->setRootDirectory(llvm::sys::path::parent_path(absPath));

  return result;
}

llvm::Expected<std::unique_ptr<ProjectConfig>>
ProjectConfig::loadFromYAML(llvm::StringRef yamlContent) {
  auto config = std::make_unique<ProjectConfig>();

  // Handle empty content as valid empty config
  if (yamlContent.empty() || yamlContent.trim().empty())
    return std::move(config);

  llvm::SourceMgr srcMgr;
  llvm::yaml::Stream stream(yamlContent, srcMgr);

  auto docIt = stream.begin();
  if (docIt == stream.end())
    return std::move(config); // Empty config is valid

  auto *root = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(docIt->getRoot());
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "project config root must be a mapping");

  parseMapping(root, [&](llvm::StringRef key, llvm::yaml::Node *value) {
    auto *mapping = llvm::dyn_cast<llvm::yaml::MappingNode>(value);

    if (key == "project" && mapping)
      parseProjectInfo(mapping, config->projectInfo);
    else if ((key == "sources" || key == "source") && mapping)
      parseSourceConfig(mapping, config->sourceConfig);
    else if (key == "lint" && mapping)
      parseLintConfig(mapping, config->lintingConfig);
    else if (key == "simulation" && mapping)
      parseSimulationConfig(mapping, config->simulationConfig);
    else if (key == "targets" && mapping) {
      parseMapping(mapping,
                   [&](llvm::StringRef targetName, llvm::yaml::Node *targetVal) {
                     if (auto *targetMap =
                             llvm::dyn_cast<llvm::yaml::MappingNode>(targetVal)) {
                       config->addTarget(
                           parseTargetConfig(targetName, targetMap));
                     }
                     return true;
                   });
    }
    return true;
  });

  return std::move(config);
}

llvm::Expected<std::unique_ptr<ProjectConfig>>
ProjectConfig::findAndLoad(llvm::StringRef directory) {
  llvm::SmallString<256> path;

  for (const auto &name : configFileNames) {
    path = directory;
    llvm::sys::path::append(path, name);

    if (llvm::sys::fs::exists(path))
      return loadFromFile(path);
  }

  return llvm::createStringError(std::errc::no_such_file_or_directory,
                                 "no project configuration file found in: %s",
                                 directory.str().c_str());
}

llvm::Expected<std::unique_ptr<ProjectConfig>>
ProjectConfig::findAndLoadRecursive(llvm::StringRef startPath) {
  llvm::SmallString<256> current(startPath);
  llvm::sys::fs::make_absolute(current);

  // Walk up the directory tree
  while (!current.empty()) {
    auto result = findAndLoad(current);
    if (result)
      return result;

    // Move to parent directory
    auto parent = llvm::sys::path::parent_path(current);
    if (parent == current)
      break; // Reached root
    current = parent;
  }

  return llvm::createStringError(
      std::errc::no_such_file_or_directory,
      "no project configuration file found in directory tree starting from: %s",
      startPath.str().c_str());
}

//===----------------------------------------------------------------------===//
// File List Parsing
//===----------------------------------------------------------------------===//

llvm::Expected<std::vector<FileListEntry>>
ProjectConfig::parseFileList(llvm::StringRef filePath) {
  auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (auto ec = fileOrErr.getError())
    return llvm::createStringError(ec, "failed to open file list: %s",
                                   filePath.str().c_str());

  llvm::SmallString<256> absPath(filePath);
  llvm::sys::fs::make_absolute(absPath);
  llvm::StringRef basePath = llvm::sys::path::parent_path(absPath);

  return parseFileListContent((*fileOrErr)->getBuffer(), basePath);
}

llvm::Expected<std::vector<FileListEntry>>
ProjectConfig::parseFileListContent(llvm::StringRef content,
                                    llvm::StringRef basePath) {
  std::vector<FileListEntry> entries;

  // Parse line by line
  llvm::SmallVector<llvm::StringRef, 128> lines;
  content.split(lines, '\n');

  for (auto line : lines) {
    // Trim whitespace
    line = line.trim();

    // Skip empty lines and comments
    if (line.empty() || line.starts_with("//") || line.starts_with("#"))
      continue;

    FileListEntry entry;

    // Handle +incdir+<path>
    if (line.starts_with("+incdir+")) {
      entry.kind = FileListEntry::Kind::IncludeDir;
      entry.value = line.substr(8).str();
    }
    // Handle +define+<macro>[=<value>]
    else if (line.starts_with("+define+")) {
      entry.kind = FileListEntry::Kind::Define;
      auto defStr = line.substr(8);
      auto [name, val] = parseDefine(defStr);
      entry.value = name;
      entry.extra = val;
    }
    // Handle +libext+<ext>
    else if (line.starts_with("+libext+")) {
      entry.kind = FileListEntry::Kind::LibExt;
      entry.value = line.substr(8).str();
    }
    // Handle -f <file>
    else if (line.starts_with("-f ") || line.starts_with("-F ")) {
      entry.kind = FileListEntry::Kind::FileList;
      entry.value = line.substr(3).trim().str();
    }
    // Handle -y <dir>
    else if (line.starts_with("-y ")) {
      entry.kind = FileListEntry::Kind::LibDir;
      entry.value = line.substr(3).trim().str();
    }
    // Handle -top <module>
    else if (line.starts_with("-top ")) {
      entry.kind = FileListEntry::Kind::TopModule;
      entry.value = line.substr(5).trim().str();
    }
    // Handle other flags starting with - or +
    else if (line.starts_with("-") || line.starts_with("+")) {
      entry.kind = FileListEntry::Kind::Other;
      entry.value = line.str();
    }
    // Regular file path
    else {
      entry.kind = FileListEntry::Kind::File;
      // Resolve relative paths
      if (!basePath.empty() && !llvm::sys::path::is_absolute(line)) {
        llvm::SmallString<256> resolved(basePath);
        llvm::sys::path::append(resolved, line);
        entry.value = resolved.str().str();
      } else {
        entry.value = line.str();
      }
    }

    entries.push_back(std::move(entry));
  }

  return entries;
}

llvm::Error ProjectConfig::expandFileList(llvm::StringRef filePath) {
  auto entriesOrErr = parseFileList(filePath);
  if (!entriesOrErr)
    return entriesOrErr.takeError();

  for (const auto &entry : *entriesOrErr) {
    switch (entry.kind) {
    case FileListEntry::Kind::File:
      sourceConfig.files.push_back(entry.value);
      break;
    case FileListEntry::Kind::IncludeDir:
      sourceConfig.includeDirs.push_back(entry.value);
      break;
    case FileListEntry::Kind::Define:
      sourceConfig.defines.push_back(formatDefine(entry.value, entry.extra));
      break;
    case FileListEntry::Kind::LibDir:
      sourceConfig.libDirs.push_back(entry.value);
      break;
    case FileListEntry::Kind::FileList:
      // Recursively expand nested file lists
      if (auto err = expandFileList(entry.value))
        return err;
      break;
    case FileListEntry::Kind::TopModule:
      if (projectInfo.topModule.empty())
        projectInfo.topModule = entry.value;
      break;
    default:
      // Ignore unrecognized entries
      break;
    }
  }

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Accessors
//===----------------------------------------------------------------------===//

const TargetConfig *ProjectConfig::getTarget(llvm::StringRef name) const {
  auto it = targets.find(name);
  if (it != targets.end())
    return &it->second;
  return nullptr;
}

std::vector<std::string> ProjectConfig::getTargetNames() const {
  std::vector<std::string> names;
  names.reserve(targets.size());
  for (const auto &entry : targets)
    names.push_back(entry.first().str());
  return names;
}

void ProjectConfig::addTarget(TargetConfig target) {
  targets[target.name] = std::move(target);
}

//===----------------------------------------------------------------------===//
// Resolution Methods
//===----------------------------------------------------------------------===//

std::string ProjectConfig::resolvePath(llvm::StringRef path) const {
  if (llvm::sys::path::is_absolute(path))
    return path.str();

  if (rootDirectory.empty())
    return path.str();

  llvm::SmallString<256> resolved(rootDirectory);
  llvm::sys::path::append(resolved, path);
  return resolved.str().str();
}

llvm::Expected<std::vector<std::string>>
ProjectConfig::expandGlob(llvm::StringRef pattern) const {
  std::vector<std::string> result;

  // Check if pattern contains glob characters
  bool isGlob = pattern.contains('*') || pattern.contains('?') ||
                pattern.contains('[') || pattern.contains('{');

  if (!isGlob) {
    // Not a glob, just return the resolved path
    std::string resolved = resolvePath(pattern);
    if (llvm::sys::fs::exists(resolved))
      result.push_back(resolved);
    return result;
  }

  // For globs, we need to walk the directory tree
  // Split pattern into base directory and glob part
  llvm::StringRef patternPath = pattern;
  llvm::SmallString<256> baseDir(rootDirectory);

  // Find the first glob character
  size_t globPos = patternPath.find_first_of("*?[{");
  if (globPos != llvm::StringRef::npos) {
    // Find the last path separator before the glob
    size_t sepPos = patternPath.substr(0, globPos).rfind('/');
    if (sepPos != llvm::StringRef::npos) {
      llvm::sys::path::append(baseDir, patternPath.substr(0, sepPos));
      patternPath = patternPath.substr(sepPos + 1);
    }
  }

  // Create glob pattern
  auto globOrErr = llvm::GlobPattern::create(patternPath);
  if (!globOrErr)
    return globOrErr.takeError();

  // Walk directory tree
  std::error_code ec;
  for (llvm::sys::fs::recursive_directory_iterator it(baseDir, ec), end;
       it != end && !ec; it.increment(ec)) {
    llvm::StringRef filePath = it->path();
    llvm::StringRef relPath = filePath;

    // Make path relative to base directory for matching
    if (filePath.starts_with(baseDir))
      relPath = filePath.substr(baseDir.size() + 1);

    if (globOrErr->match(relPath))
      result.push_back(filePath.str());
  }

  return result;
}

llvm::Expected<std::vector<std::string>>
ProjectConfig::resolveSourceFiles() const {
  std::vector<std::string> files;

  // First expand file lists
  for (const auto &fileList : sourceConfig.fileLists) {
    std::string resolved = resolvePath(fileList);
    auto entriesOrErr = parseFileList(resolved);
    if (!entriesOrErr)
      return entriesOrErr.takeError();

    for (const auto &entry : *entriesOrErr) {
      if (entry.kind == FileListEntry::Kind::File) {
        // Expand globs in file entries
        auto expanded = expandGlob(entry.value);
        if (!expanded)
          return expanded.takeError();
        files.insert(files.end(), expanded->begin(), expanded->end());
      }
    }
  }

  // Then expand file patterns
  for (const auto &pattern : sourceConfig.files) {
    auto expanded = expandGlob(pattern);
    if (!expanded)
      return expanded.takeError();
    files.insert(files.end(), expanded->begin(), expanded->end());
  }

  return files;
}

std::vector<std::string> ProjectConfig::resolveIncludeDirs() const {
  std::vector<std::string> dirs;
  dirs.reserve(sourceConfig.includeDirs.size());

  for (const auto &dir : sourceConfig.includeDirs)
    dirs.push_back(resolvePath(dir));

  return dirs;
}

std::vector<std::string> ProjectConfig::getAllDefines() const {
  std::vector<std::string> defines;
  defines.reserve(sourceConfig.defines.size() + simulationConfig.defines.size());

  defines.insert(defines.end(), sourceConfig.defines.begin(),
                 sourceConfig.defines.end());
  defines.insert(defines.end(), simulationConfig.defines.begin(),
                 simulationConfig.defines.end());

  return defines;
}

llvm::Expected<std::unique_ptr<ProjectConfig>>
ProjectConfig::getEffectiveConfig(llvm::StringRef targetName) const {
  auto target = getTarget(targetName);
  if (!target)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "unknown target: %s",
                                   targetName.str().c_str());

  auto effective = std::make_unique<ProjectConfig>(*this);

  // Override project info
  if (!target->topModule.empty())
    effective->projectInfo.topModule = target->topModule;

  // Add target-specific defines
  effective->sourceConfig.defines.insert(
      effective->sourceConfig.defines.end(), target->defines.begin(),
      target->defines.end());

  // Add target-specific include dirs
  effective->sourceConfig.includeDirs.insert(
      effective->sourceConfig.includeDirs.end(), target->includeDirs.begin(),
      target->includeDirs.end());

  // Add target-specific files
  effective->sourceConfig.files.insert(effective->sourceConfig.files.end(),
                                       target->files.begin(),
                                       target->files.end());

  // Override lint setting
  if (target->lintEnabled.has_value())
    effective->lintingConfig.enabled = *target->lintEnabled;

  return effective;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

llvm::Error ProjectConfig::validate() const {
  // Check that required directories exist
  for (const auto &dir : sourceConfig.includeDirs) {
    std::string resolved = resolvePath(dir);
    if (!llvm::sys::fs::exists(resolved)) {
      // Warning, not error - directory might be created later
    }
  }

  // Validate lint config file exists if specified
  if (!lintingConfig.configFile.empty()) {
    std::string resolved = resolvePath(lintingConfig.configFile);
    if (!llvm::sys::fs::exists(resolved)) {
      return llvm::createStringError(std::errc::no_such_file_or_directory,
                                     "lint config file not found: %s",
                                     resolved.c_str());
    }
  }

  return llvm::Error::success();
}

bool ProjectConfig::isEmpty() const {
  return projectInfo.name.empty() && projectInfo.topModule.empty() &&
         sourceConfig.files.empty() && sourceConfig.includeDirs.empty() &&
         sourceConfig.defines.empty() && sourceConfig.fileLists.empty() &&
         targets.empty();
}

//===----------------------------------------------------------------------===//
// Serialization
//===----------------------------------------------------------------------===//

std::string ProjectConfig::toYAML() const {
  std::string yaml;
  llvm::raw_string_ostream os(yaml);

  os << "# CIRCT Project Configuration\n";
  os << "# Generated by CIRCT\n\n";

  // Project section
  os << "project:\n";
  if (!projectInfo.name.empty())
    os << "  name: \"" << projectInfo.name << "\"\n";
  if (!projectInfo.topModule.empty())
    os << "  top: \"" << projectInfo.topModule << "\"\n";
  if (!projectInfo.version.empty())
    os << "  version: \"" << projectInfo.version << "\"\n";
  if (!projectInfo.description.empty())
    os << "  description: \"" << projectInfo.description << "\"\n";

  // Sources section
  os << "\nsources:\n";
  if (!sourceConfig.includeDirs.empty()) {
    os << "  include_dirs:\n";
    for (const auto &dir : sourceConfig.includeDirs)
      os << "    - \"" << dir << "\"\n";
  }
  if (!sourceConfig.defines.empty()) {
    os << "  defines:\n";
    for (const auto &def : sourceConfig.defines)
      os << "    - \"" << def << "\"\n";
  }
  if (!sourceConfig.files.empty()) {
    os << "  files:\n";
    for (const auto &file : sourceConfig.files)
      os << "    - \"" << file << "\"\n";
  }
  if (!sourceConfig.fileLists.empty()) {
    os << "  file_lists:\n";
    for (const auto &fl : sourceConfig.fileLists)
      os << "    - \"" << fl << "\"\n";
  }

  // Lint section
  os << "\nlint:\n";
  os << "  enabled: " << (lintingConfig.enabled ? "true" : "false") << "\n";
  if (!lintingConfig.configFile.empty())
    os << "  config: \"" << lintingConfig.configFile << "\"\n";

  // Simulation section
  if (!simulationConfig.timescale.empty() ||
      !simulationConfig.defaultClocking.empty()) {
    os << "\nsimulation:\n";
    if (!simulationConfig.timescale.empty())
      os << "  timescale: \"" << simulationConfig.timescale << "\"\n";
    if (!simulationConfig.defaultClocking.empty())
      os << "  default_clocking: \"" << simulationConfig.defaultClocking
         << "\"\n";
  }

  // Targets section
  if (!targets.empty()) {
    os << "\ntargets:\n";
    for (const auto &entry : targets) {
      const auto &target = entry.second;
      os << "  " << target.name << ":\n";
      if (!target.topModule.empty())
        os << "    top: \"" << target.topModule << "\"\n";
      if (!target.defines.empty()) {
        os << "    defines:\n";
        for (const auto &def : target.defines)
          os << "      - \"" << def << "\"\n";
      }
    }
  }

  return yaml;
}

std::unique_ptr<ProjectConfig> ProjectConfig::createDefault() {
  auto config = std::make_unique<ProjectConfig>();

  // Set up default file extensions
  config->sourceConfig.svExtensions = {".sv", ".svh"};
  config->sourceConfig.verilogExtensions = {".v", ".vh"};

  // Default timescale
  config->simulationConfig.timescale = "1ns/1ps";

  return config;
}
