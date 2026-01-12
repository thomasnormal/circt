//===- LintConfig.cpp - Verilog/SystemVerilog lint configuration ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LintConfig class for loading and managing lint
// configuration from YAML files.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/Linting/LintConfig.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

#include <regex>

using namespace circt::lint;

//===----------------------------------------------------------------------===//
// Severity Helpers
//===----------------------------------------------------------------------===//

llvm::StringRef circt::lint::severityToString(LintSeverity severity) {
  switch (severity) {
  case LintSeverity::Ignore:
    return "ignore";
  case LintSeverity::Hint:
    return "hint";
  case LintSeverity::Warning:
    return "warning";
  case LintSeverity::Error:
    return "error";
  }
  llvm_unreachable("invalid severity");
}

std::optional<LintSeverity> circt::lint::parseSeverity(llvm::StringRef str) {
  auto lower = str.lower();
  if (lower == "ignore" || lower == "off" || lower == "disabled" ||
      lower == "none")
    return LintSeverity::Ignore;
  if (lower == "hint" || lower == "info" || lower == "information")
    return LintSeverity::Hint;
  if (lower == "warning" || lower == "warn")
    return LintSeverity::Warning;
  if (lower == "error" || lower == "err")
    return LintSeverity::Error;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// LintConfig Implementation
//===----------------------------------------------------------------------===//

LintConfig::LintConfig() = default;
LintConfig::~LintConfig() = default;

namespace {

/// Helper to parse a YAML mapping node.
bool parseYAMLMapping(
    llvm::yaml::MappingNode *mapping,
    std::function<bool(llvm::StringRef, llvm::yaml::Node *)> callback) {
  for (auto &entry : *mapping) {
    auto *keyNode = dyn_cast<llvm::yaml::ScalarNode>(entry.getKey());
    if (!keyNode)
      return false;

    llvm::SmallString<32> keyStorage;
    llvm::StringRef key = keyNode->getValue(keyStorage);

    if (!callback(key, entry.getValue()))
      return false;
  }
  return true;
}

/// Get scalar value from a YAML node.
llvm::StringRef getScalarValue(llvm::yaml::Node *node,
                               llvm::SmallVectorImpl<char> &storage) {
  if (auto *scalar = dyn_cast<llvm::yaml::ScalarNode>(node))
    return scalar->getValue(storage);
  return "";
}

} // namespace

llvm::Expected<std::unique_ptr<LintConfig>>
LintConfig::loadFromFile(llvm::StringRef filePath) {
  auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (auto ec = fileOrErr.getError())
    return llvm::createStringError(ec, "failed to open lint config file: %s",
                                   filePath.str().c_str());

  return loadFromYAML((*fileOrErr)->getBuffer());
}

llvm::Expected<std::unique_ptr<LintConfig>>
LintConfig::loadFromYAML(llvm::StringRef yamlContent) {
  auto config = std::make_unique<LintConfig>();

  llvm::SourceMgr srcMgr;
  llvm::yaml::Stream stream(yamlContent, srcMgr);

  auto docIt = stream.begin();
  if (docIt == stream.end())
    return std::move(config); // Empty config is valid

  auto *root = dyn_cast_or_null<llvm::yaml::MappingNode>(docIt->getRoot());
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "lint config root must be a mapping");

  llvm::SmallString<64> storage;

  bool success = parseYAMLMapping(root, [&](llvm::StringRef key,
                                            llvm::yaml::Node *value) -> bool {
    if (key == "rules") {
      auto *rulesMap = dyn_cast<llvm::yaml::MappingNode>(value);
      if (!rulesMap)
        return false;

      return parseYAMLMapping(
          rulesMap, [&](llvm::StringRef ruleName, llvm::yaml::Node *ruleValue) {
            LintRuleConfig ruleConfig;

            // Rule value can be either a severity string or a mapping
            if (auto *scalarNode =
                    dyn_cast<llvm::yaml::ScalarNode>(ruleValue)) {
              llvm::SmallString<32> valueStorage;
              llvm::StringRef severityStr =
                  scalarNode->getValue(valueStorage);
              if (auto severity = parseSeverity(severityStr)) {
                ruleConfig.severity = *severity;
                ruleConfig.enabled = (*severity != LintSeverity::Ignore);
              }
            } else if (auto *mappingNode =
                           dyn_cast<llvm::yaml::MappingNode>(ruleValue)) {
              parseYAMLMapping(
                  mappingNode,
                  [&](llvm::StringRef optKey, llvm::yaml::Node *optValue) {
                    llvm::SmallString<64> optStorage;

                    if (optKey == "severity") {
                      auto severityStr = getScalarValue(optValue, optStorage);
                      if (auto severity = parseSeverity(severityStr)) {
                        ruleConfig.severity = *severity;
                        ruleConfig.enabled =
                            (*severity != LintSeverity::Ignore);
                      }
                    } else if (optKey == "enabled") {
                      auto enabledStr = getScalarValue(optValue, optStorage);
                      ruleConfig.enabled =
                          (enabledStr == "true" || enabledStr == "yes" ||
                           enabledStr == "1");
                    } else if (optKey == "pattern" || optKey == "regex") {
                      ruleConfig.stringOptions[optKey.str()] =
                          getScalarValue(optValue, optStorage).str();
                    } else {
                      // Store as generic string option
                      ruleConfig.stringOptions[optKey.str()] =
                          getScalarValue(optValue, optStorage).str();
                    }
                    return true;
                  });
            }

            config->setRuleConfig(ruleName, ruleConfig);
            return true;
          });
    } else if (key == "naming") {
      auto *namingMap = dyn_cast<llvm::yaml::MappingNode>(value);
      if (!namingMap)
        return false;

      NamingConventionConfig namingConfig;
      parseYAMLMapping(namingMap,
                       [&](llvm::StringRef optKey, llvm::yaml::Node *optValue) {
                         llvm::SmallString<128> optStorage;
                         auto patternStr = getScalarValue(optValue, optStorage);

                         if (optKey == "module_pattern")
                           namingConfig.modulePattern = patternStr.str();
                         else if (optKey == "signal_pattern")
                           namingConfig.signalPattern = patternStr.str();
                         else if (optKey == "parameter_pattern")
                           namingConfig.parameterPattern = patternStr.str();
                         else if (optKey == "port_pattern")
                           namingConfig.portPattern = patternStr.str();
                         else if (optKey == "instance_pattern")
                           namingConfig.instancePattern = patternStr.str();
                         else if (optKey == "constant_pattern")
                           namingConfig.constantPattern = patternStr.str();

                         return true;
                       });
      config->setNamingConfig(namingConfig);
    } else if (key == "exclude") {
      if (auto *excludeSeq = dyn_cast<llvm::yaml::SequenceNode>(value)) {
        for (auto &item : *excludeSeq) {
          if (auto *scalarNode = dyn_cast<llvm::yaml::ScalarNode>(&item)) {
            llvm::SmallString<128> itemStorage;
            config->addExcludePattern(scalarNode->getValue(itemStorage));
          }
        }
      }
    }

    return true;
  });

  if (!success)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse lint configuration");

  return std::move(config);
}

const LintRuleConfig &
LintConfig::getRuleConfig(llvm::StringRef ruleName) const {
  auto it = ruleConfigs.find(ruleName);
  if (it != ruleConfigs.end())
    return it->second;
  return defaultConfig;
}

void LintConfig::setRuleConfig(llvm::StringRef ruleName, LintRuleConfig config) {
  ruleConfigs[ruleName] = std::move(config);
}

bool LintConfig::isRuleEnabled(llvm::StringRef ruleName) const {
  return getRuleConfig(ruleName).enabled;
}

void LintConfig::setRuleEnabled(llvm::StringRef ruleName, bool enabled) {
  ruleConfigs[ruleName].enabled = enabled;
}

LintSeverity LintConfig::getRuleSeverity(llvm::StringRef ruleName) const {
  return getRuleConfig(ruleName).severity;
}

void LintConfig::setRuleSeverity(llvm::StringRef ruleName,
                                  LintSeverity severity) {
  ruleConfigs[ruleName].severity = severity;
  ruleConfigs[ruleName].enabled = (severity != LintSeverity::Ignore);
}

void LintConfig::addExcludePattern(llvm::StringRef pattern) {
  excludePatterns.push_back(pattern.str());
}

bool LintConfig::shouldExcludeFile(llvm::StringRef filePath) const {
  for (const auto &pattern : excludePatterns) {
    // Simple glob matching (supports * and ?)
    try {
      // Convert glob pattern to regex
      std::string regexPattern = "^";
      for (char c : pattern) {
        switch (c) {
        case '*':
          regexPattern += ".*";
          break;
        case '?':
          regexPattern += ".";
          break;
        case '.':
          regexPattern += "\\.";
          break;
        case '/':
        case '\\':
          regexPattern += "[/\\\\]";
          break;
        default:
          regexPattern += c;
          break;
        }
      }
      regexPattern += "$";

      std::regex re(regexPattern);
      if (std::regex_match(filePath.str(), re))
        return true;
    } catch (const std::regex_error &) {
      // If regex compilation fails, try simple substring match
      if (filePath.contains(pattern))
        return true;
    }
  }
  return false;
}

std::vector<std::string> LintConfig::getConfiguredRules() const {
  std::vector<std::string> names;
  names.reserve(ruleConfigs.size());
  for (const auto &entry : ruleConfigs)
    names.push_back(entry.first().str());
  return names;
}

void LintConfig::merge(const LintConfig &other) {
  // Merge rule configs
  for (const auto &entry : other.ruleConfigs)
    ruleConfigs[entry.first()] = entry.second;

  // Take naming config from other
  namingConfig = other.namingConfig;

  // Merge exclude patterns
  for (const auto &pattern : other.excludePatterns)
    excludePatterns.push_back(pattern);
}

std::unique_ptr<LintConfig> LintConfig::createDefault() {
  auto config = std::make_unique<LintConfig>();
  config->enableAllRules(LintSeverity::Warning);
  return config;
}

void LintConfig::enableAllRules(LintSeverity severity) {
  // Enable common rules at the specified severity
  const char *ruleNames[] = {
      "unused_signal",          "unused_parameter",
      "undriven_signal",        "unread_signal",
      "implicit_width",         "blocking_in_sequential",
      "nonblocking_in_combinational", "missing_default",
      "naming_convention",      "incomplete_case",
      "multiple_drivers",       "latch_inference",
  };

  for (const char *name : ruleNames) {
    LintRuleConfig ruleConfig;
    ruleConfig.severity = severity;
    ruleConfig.enabled = true;
    ruleConfigs[name] = ruleConfig;
  }
}

void LintConfig::disableAllRules() {
  for (auto &entry : ruleConfigs) {
    entry.second.enabled = false;
    entry.second.severity = LintSeverity::Ignore;
  }
}
