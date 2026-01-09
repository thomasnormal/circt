//===- LintRules.h - Verilog/SystemVerilog lint rules -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the lint rule infrastructure for Verilog/SystemVerilog.
// It provides a base class for lint rules and concrete implementations for
// common linting checks like unused signals, naming conventions, etc.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_LINTING_LINTRULES_H
#define CIRCT_ANALYSIS_LINTING_LINTRULES_H

#include "circt/Analysis/Linting/LintConfig.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>
#include <vector>

// Forward declarations for slang types
namespace slang {
namespace ast {
class Compilation;
class Symbol;
class Type;
class Expression;
class Statement;
} // namespace ast
class SourceRange;
class SourceLocation;
class SourceManager;
} // namespace slang

namespace circt {
namespace lint {

/// A single lint diagnostic produced by a lint rule.
struct LintDiagnostic {
  /// The rule that produced this diagnostic.
  std::string ruleName;

  /// The diagnostic message.
  std::string message;

  /// The severity of the diagnostic.
  LintSeverity severity;

  /// The source location of the issue.
  slang::SourceLocation location;

  /// The source range of the issue (may be larger than location).
  std::optional<slang::SourceRange> range;

  /// Optional related locations for additional context.
  std::vector<std::pair<slang::SourceLocation, std::string>> relatedLocations;

  /// Optional fix suggestion.
  std::optional<std::string> fixSuggestion;

  /// Optional code for the diagnostic (for filtering/grouping).
  std::optional<std::string> code;
};

/// Base class for all lint rules.
class LintRule {
public:
  LintRule(llvm::StringRef name, llvm::StringRef description);
  virtual ~LintRule();

  /// Get the unique name of this rule (e.g., "unused_signal").
  llvm::StringRef getName() const { return name; }

  /// Get a human-readable description of this rule.
  llvm::StringRef getDescription() const { return description; }

  /// Get the default severity for this rule.
  virtual LintSeverity getDefaultSeverity() const {
    return LintSeverity::Warning;
  }

  /// Get the category of this rule (e.g., "style", "error-prone", "naming").
  virtual llvm::StringRef getCategory() const { return "general"; }

  /// Run this rule on the given compilation.
  /// Returns a list of diagnostics produced.
  virtual std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) = 0;

  /// Check if this rule is applicable to the given compilation.
  /// Subclasses can override to skip certain compilations.
  virtual bool isApplicable(const slang::ast::Compilation &compilation) const {
    return true;
  }

protected:
  std::string name;
  std::string description;
};

//===----------------------------------------------------------------------===//
// Concrete Lint Rules
//===----------------------------------------------------------------------===//

/// Rule: Detect unused signals (declared but never read).
class UnusedSignalRule : public LintRule {
public:
  UnusedSignalRule();

  llvm::StringRef getCategory() const override { return "unused"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect unused parameters (declared but never referenced).
class UnusedParameterRule : public LintRule {
public:
  UnusedParameterRule();

  llvm::StringRef getCategory() const override { return "unused"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect undriven signals (read but never assigned).
class UndrivenSignalRule : public LintRule {
public:
  UndrivenSignalRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }
  LintSeverity getDefaultSeverity() const override {
    return LintSeverity::Warning;
  }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect unread signals (assigned but never read).
class UnreadSignalRule : public LintRule {
public:
  UnreadSignalRule();

  llvm::StringRef getCategory() const override { return "unused"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect implicit width conversions/truncations.
class ImplicitWidthConversionRule : public LintRule {
public:
  ImplicitWidthConversionRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect blocking assignments in sequential (always_ff) blocks.
class BlockingInSequentialRule : public LintRule {
public:
  BlockingInSequentialRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }
  LintSeverity getDefaultSeverity() const override {
    return LintSeverity::Warning;
  }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect non-blocking assignments in combinational (always_comb) blocks.
class NonBlockingInCombinationalRule : public LintRule {
public:
  NonBlockingInCombinationalRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }
  LintSeverity getDefaultSeverity() const override {
    return LintSeverity::Warning;
  }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect missing default case in case statements.
class MissingDefaultCaseRule : public LintRule {
public:
  MissingDefaultCaseRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Check naming conventions against configurable patterns.
class NamingConventionRule : public LintRule {
public:
  NamingConventionRule();

  llvm::StringRef getCategory() const override { return "naming"; }
  LintSeverity getDefaultSeverity() const override { return LintSeverity::Hint; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect case statements that don't cover all enum values.
class IncompleteCaseRule : public LintRule {
public:
  IncompleteCaseRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect signals driven in multiple always blocks.
class MultipleDriversRule : public LintRule {
public:
  MultipleDriversRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }
  LintSeverity getDefaultSeverity() const override {
    return LintSeverity::Warning;
  }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

/// Rule: Detect latches (combinational logic with incomplete sensitivity).
class LatchInferenceRule : public LintRule {
public:
  LatchInferenceRule();

  llvm::StringRef getCategory() const override { return "error-prone"; }
  LintSeverity getDefaultSeverity() const override {
    return LintSeverity::Warning;
  }

  std::vector<LintDiagnostic>
  check(const slang::ast::Compilation &compilation,
        const LintRuleConfig &config) override;
};

//===----------------------------------------------------------------------===//
// Lint Rule Registry
//===----------------------------------------------------------------------===//

/// Registry for lint rules.
class LintRuleRegistry {
public:
  /// Get the global rule registry.
  static LintRuleRegistry &getInstance();

  /// Register a new lint rule.
  void registerRule(std::unique_ptr<LintRule> rule);

  /// Get a rule by name.
  LintRule *getRule(llvm::StringRef name) const;

  /// Get all registered rules.
  const std::vector<std::unique_ptr<LintRule>> &getAllRules() const {
    return rules;
  }

  /// Get all rule names.
  std::vector<std::string> getRuleNames() const;

  /// Get rules by category.
  std::vector<LintRule *> getRulesByCategory(llvm::StringRef category) const;

private:
  LintRuleRegistry();
  ~LintRuleRegistry();

  /// Register all built-in rules.
  void registerBuiltinRules();

  std::vector<std::unique_ptr<LintRule>> rules;
  llvm::StringMap<LintRule *> ruleMap;
};

//===----------------------------------------------------------------------===//
// Lint Runner
//===----------------------------------------------------------------------===//

/// Results from running lint rules.
struct LintResults {
  /// All diagnostics produced.
  std::vector<LintDiagnostic> diagnostics;

  /// Number of errors.
  size_t errorCount = 0;

  /// Number of warnings.
  size_t warningCount = 0;

  /// Number of hints.
  size_t hintCount = 0;

  /// Whether linting succeeded (no errors).
  bool success() const { return errorCount == 0; }
};

/// Runs lint rules on a compilation.
class LintRunner {
public:
  explicit LintRunner(const LintConfig &config);

  /// Run all enabled lint rules on the given compilation.
  LintResults run(const slang::ast::Compilation &compilation);

  /// Run a specific rule by name.
  std::vector<LintDiagnostic>
  runRule(llvm::StringRef ruleName,
          const slang::ast::Compilation &compilation);

private:
  const LintConfig &config;
};

} // namespace lint
} // namespace circt

#endif // CIRCT_ANALYSIS_LINTING_LINTRULES_H
