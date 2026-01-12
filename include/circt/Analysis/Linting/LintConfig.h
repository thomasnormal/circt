//===- LintConfig.h - Verilog/SystemVerilog lint configuration --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the LintConfig class and related types for configuring
// Verilog/SystemVerilog linting rules. The configuration can be loaded from
// YAML files (e.g., circt-lint.yaml) and supports enabling/disabling rules,
// setting severity levels, and configuring rule-specific options.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_LINTING_LINTCONFIG_H
#define CIRCT_ANALYSIS_LINTING_LINTCONFIG_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace circt {
namespace lint {

/// Severity levels for lint diagnostics.
enum class LintSeverity {
  Ignore,  /// Rule is disabled
  Hint,    /// Informational hint
  Warning, /// Warning level
  Error    /// Error level (may fail compilation)
};

/// Convert severity enum to string.
llvm::StringRef severityToString(LintSeverity severity);

/// Parse severity from string.
std::optional<LintSeverity> parseSeverity(llvm::StringRef str);

/// Configuration for a single lint rule.
struct LintRuleConfig {
  /// The severity level for this rule.
  LintSeverity severity = LintSeverity::Warning;

  /// Whether the rule is enabled.
  bool enabled = true;

  /// Rule-specific string options (e.g., regex patterns).
  llvm::StringMap<std::string> stringOptions;

  /// Rule-specific integer options.
  llvm::StringMap<int64_t> intOptions;

  /// Rule-specific boolean options.
  llvm::StringMap<bool> boolOptions;
};

/// Configuration for naming convention rules.
struct NamingConventionConfig {
  /// Regex pattern for module names.
  std::string modulePattern = "^[A-Z][a-zA-Z0-9_]*$";

  /// Regex pattern for signal/variable names.
  std::string signalPattern = "^[a-z][a-z0-9_]*$";

  /// Regex pattern for parameter names.
  std::string parameterPattern = "^[A-Z][A-Z0-9_]*$";

  /// Regex pattern for port names.
  std::string portPattern = "^[a-z][a-z0-9_]*(_i|_o|_io)?$";

  /// Regex pattern for instance names.
  std::string instancePattern = "^[a-z][a-z0-9_]*$";

  /// Regex pattern for constant/localparam names.
  std::string constantPattern = "^[A-Z][A-Z0-9_]*$";
};

/// Main lint configuration class.
class LintConfig {
public:
  LintConfig();
  ~LintConfig();

  /// Load configuration from a YAML file.
  static llvm::Expected<std::unique_ptr<LintConfig>>
  loadFromFile(llvm::StringRef filePath);

  /// Load configuration from a YAML string.
  static llvm::Expected<std::unique_ptr<LintConfig>>
  loadFromYAML(llvm::StringRef yamlContent);

  /// Get the configuration for a specific rule by name.
  /// Returns default configuration if the rule is not explicitly configured.
  const LintRuleConfig &getRuleConfig(llvm::StringRef ruleName) const;

  /// Set the configuration for a specific rule.
  void setRuleConfig(llvm::StringRef ruleName, LintRuleConfig config);

  /// Check if a rule is enabled.
  bool isRuleEnabled(llvm::StringRef ruleName) const;

  /// Enable or disable a rule.
  void setRuleEnabled(llvm::StringRef ruleName, bool enabled);

  /// Get the severity for a rule.
  LintSeverity getRuleSeverity(llvm::StringRef ruleName) const;

  /// Set the severity for a rule.
  void setRuleSeverity(llvm::StringRef ruleName, LintSeverity severity);

  /// Get the naming convention configuration.
  const NamingConventionConfig &getNamingConfig() const { return namingConfig; }

  /// Set the naming convention configuration.
  void setNamingConfig(const NamingConventionConfig &config) {
    namingConfig = config;
  }

  /// Get the list of file patterns to exclude from linting.
  const std::vector<std::string> &getExcludePatterns() const {
    return excludePatterns;
  }

  /// Add a file pattern to exclude from linting.
  void addExcludePattern(llvm::StringRef pattern);

  /// Check if a file should be excluded from linting.
  bool shouldExcludeFile(llvm::StringRef filePath) const;

  /// Get all configured rule names.
  std::vector<std::string> getConfiguredRules() const;

  /// Merge another configuration into this one.
  /// Rules from 'other' take precedence.
  void merge(const LintConfig &other);

  /// Create a default configuration with all rules enabled at warning level.
  static std::unique_ptr<LintConfig> createDefault();

  /// Enable all rules at the specified severity level.
  void enableAllRules(LintSeverity severity = LintSeverity::Warning);

  /// Disable all rules.
  void disableAllRules();

private:
  /// Rule configurations keyed by rule name.
  llvm::StringMap<LintRuleConfig> ruleConfigs;

  /// Default configuration for unconfigured rules.
  LintRuleConfig defaultConfig;

  /// Naming convention configuration.
  NamingConventionConfig namingConfig;

  /// File patterns to exclude from linting.
  std::vector<std::string> excludePatterns;
};

} // namespace lint
} // namespace circt

#endif // CIRCT_ANALYSIS_LINTING_LINTCONFIG_H
