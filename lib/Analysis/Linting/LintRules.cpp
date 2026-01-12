//===- LintRules.cpp - Verilog/SystemVerilog lint rules --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lint rules for Verilog/SystemVerilog code quality
// analysis. The rules check for common issues like unused signals, naming
// convention violations, and potential synthesis problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/Linting/LintRules.h"

#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Compilation.h"
#include "slang/ast/Expression.h"
#include "slang/ast/Statements.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/ParameterSymbols.h"
#include "slang/ast/symbols/PortSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Regex.h"

#include <regex>

using namespace circt::lint;

//===----------------------------------------------------------------------===//
// LintRule Base Class
//===----------------------------------------------------------------------===//

LintRule::LintRule(llvm::StringRef name, llvm::StringRef description)
    : name(name.str()), description(description.str()) {}

LintRule::~LintRule() = default;

//===----------------------------------------------------------------------===//
// Signal Usage Tracking Visitor
//===----------------------------------------------------------------------===//

namespace {

/// Tracks signal reads and writes throughout a module.
class SignalUsageVisitor
    : public slang::ast::ASTVisitor<SignalUsageVisitor, true, true> {
public:
  llvm::DenseSet<const slang::ast::ValueSymbol *> readSignals;
  llvm::DenseSet<const slang::ast::ValueSymbol *> writtenSignals;
  llvm::DenseSet<const slang::ast::Symbol *> referencedSymbols;

  void handle(const slang::ast::NamedValueExpression &expr) {
    referencedSymbols.insert(&expr.symbol);
    if (auto *valueSymbol =
            expr.symbol.as_if<slang::ast::ValueSymbol>()) {
      // Track whether the value is being read or written based on context
      // For simplicity, we mark all uses as reads initially
      readSignals.insert(valueSymbol);
    }
  }

  void handle(const slang::ast::AssignmentExpression &expr) {
    // The left-hand side is being written
    if (auto *namedValue =
            expr.left().as_if<slang::ast::NamedValueExpression>()) {
      if (auto *valueSymbol =
              namedValue->symbol.as_if<slang::ast::ValueSymbol>()) {
        writtenSignals.insert(valueSymbol);
      }
      referencedSymbols.insert(&namedValue->symbol);
    }
    // Right-hand side is being read
    visitDefault(expr.right());
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

/// Visitor to find blocking/non-blocking assignments in procedural blocks.
class AssignmentTypeVisitor
    : public slang::ast::ASTVisitor<AssignmentTypeVisitor, true, true> {
public:
  std::vector<std::pair<slang::SourceLocation, bool>> assignments;

  void handle(const slang::ast::AssignmentExpression &expr) {
    // isNonBlocking() indicates if this is a non-blocking assignment (<=)
    bool isNonBlocking = expr.isNonBlocking();
    assignments.emplace_back(expr.sourceRange.start(), isNonBlocking);
    visitDefault(expr);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

/// Visitor to find case statements without default.
class CaseStatementVisitor
    : public slang::ast::ASTVisitor<CaseStatementVisitor, true, true> {
public:
  std::vector<const slang::ast::CaseStatement *> caseStatementsWithoutDefault;

  void handle(const slang::ast::CaseStatement &stmt) {
    bool hasDefault = false;
    for (const auto &item : stmt.items) {
      if (item.expressions.empty()) {
        // Empty expressions list means this is the default case
        hasDefault = true;
        break;
      }
    }
    if (stmt.defaultCase)
      hasDefault = true;

    if (!hasDefault)
      caseStatementsWithoutDefault.push_back(&stmt);

    visitDefault(stmt);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

/// Check if a symbol name matches a regex pattern.
bool matchesPattern(llvm::StringRef name, llvm::StringRef pattern) {
  if (pattern.empty())
    return true;
  try {
    std::regex re(pattern.str());
    return std::regex_match(name.str(), re);
  } catch (const std::regex_error &) {
    return true; // If pattern is invalid, don't report violations
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// UnusedSignalRule Implementation
//===----------------------------------------------------------------------===//

UnusedSignalRule::UnusedSignalRule()
    : LintRule("unused_signal",
               "Detects signals that are declared but never used") {}

std::vector<LintDiagnostic>
UnusedSignalRule::check(const slang::ast::Compilation &compilation,
                        const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  for (auto *inst : compilation.getRoot().topInstances) {
    // Collect all signals in the module
    llvm::DenseSet<const slang::ast::ValueSymbol *> allSignals;

    for (const auto &member : inst->body.members()) {
      if (auto *var = member.as_if<slang::ast::VariableSymbol>())
        allSignals.insert(var);
      else if (auto *net = member.as_if<slang::ast::NetSymbol>())
        allSignals.insert(net);
    }

    // Track signal usage
    SignalUsageVisitor usageVisitor;
    inst->body.visit(usageVisitor);

    // Find unused signals
    for (const auto *signal : allSignals) {
      if (!usageVisitor.readSignals.contains(signal) &&
          !usageVisitor.writtenSignals.contains(signal)) {
        LintDiagnostic diag;
        diag.ruleName = getName().str();
        diag.message = "signal '" + std::string(signal->name) +
                       "' is declared but never used";
        diag.severity = config.severity;
        diag.location = signal->location;
        diag.code = "unused_signal";
        diagnostics.push_back(std::move(diag));
      }
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// UnusedParameterRule Implementation
//===----------------------------------------------------------------------===//

UnusedParameterRule::UnusedParameterRule()
    : LintRule("unused_parameter",
               "Detects parameters that are declared but never referenced") {}

std::vector<LintDiagnostic>
UnusedParameterRule::check(const slang::ast::Compilation &compilation,
                           const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  for (auto *inst : compilation.getRoot().topInstances) {
    // Collect all parameters
    llvm::DenseSet<const slang::ast::Symbol *> allParams;

    for (const auto &member : inst->body.members()) {
      if (member.kind == slang::ast::SymbolKind::Parameter ||
          member.kind == slang::ast::SymbolKind::TypeParameter)
        allParams.insert(&member);
    }

    // Track parameter usage
    SignalUsageVisitor usageVisitor;
    inst->body.visit(usageVisitor);

    // Find unused parameters
    for (const auto *param : allParams) {
      if (!usageVisitor.referencedSymbols.contains(param)) {
        LintDiagnostic diag;
        diag.ruleName = getName().str();
        diag.message = "parameter '" + std::string(param->name) +
                       "' is declared but never used";
        diag.severity = config.severity;
        diag.location = param->location;
        diag.code = "unused_parameter";
        diagnostics.push_back(std::move(diag));
      }
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// UndrivenSignalRule Implementation
//===----------------------------------------------------------------------===//

UndrivenSignalRule::UndrivenSignalRule()
    : LintRule("undriven_signal",
               "Detects signals that are read but never assigned") {}

std::vector<LintDiagnostic>
UndrivenSignalRule::check(const slang::ast::Compilation &compilation,
                          const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  for (auto *inst : compilation.getRoot().topInstances) {
    // Collect internal signals (not ports)
    llvm::DenseSet<const slang::ast::ValueSymbol *> internalSignals;

    for (const auto &member : inst->body.members()) {
      if (auto *var = member.as_if<slang::ast::VariableSymbol>())
        internalSignals.insert(var);
      else if (auto *net = member.as_if<slang::ast::NetSymbol>())
        internalSignals.insert(net);
    }

    // Track signal usage
    SignalUsageVisitor usageVisitor;
    inst->body.visit(usageVisitor);

    // Find undriven signals (read but never written)
    for (const auto *signal : internalSignals) {
      if (usageVisitor.readSignals.contains(signal) &&
          !usageVisitor.writtenSignals.contains(signal)) {
        LintDiagnostic diag;
        diag.ruleName = getName().str();
        diag.message = "signal '" + std::string(signal->name) +
                       "' is read but never assigned a value";
        diag.severity = config.severity;
        diag.location = signal->location;
        diag.code = "undriven_signal";
        diagnostics.push_back(std::move(diag));
      }
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// UnreadSignalRule Implementation
//===----------------------------------------------------------------------===//

UnreadSignalRule::UnreadSignalRule()
    : LintRule("unread_signal",
               "Detects signals that are assigned but never read") {}

std::vector<LintDiagnostic>
UnreadSignalRule::check(const slang::ast::Compilation &compilation,
                        const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  for (auto *inst : compilation.getRoot().topInstances) {
    // Collect internal signals (not output ports)
    llvm::DenseSet<const slang::ast::ValueSymbol *> internalSignals;

    for (const auto &member : inst->body.members()) {
      if (auto *var = member.as_if<slang::ast::VariableSymbol>())
        internalSignals.insert(var);
      // Exclude output wires since they're meant to be read externally
      else if (auto *net = member.as_if<slang::ast::NetSymbol>())
        internalSignals.insert(net);
    }

    // Track signal usage
    SignalUsageVisitor usageVisitor;
    inst->body.visit(usageVisitor);

    // Find unread signals (written but never read internally)
    for (const auto *signal : internalSignals) {
      if (!usageVisitor.readSignals.contains(signal) &&
          usageVisitor.writtenSignals.contains(signal)) {
        LintDiagnostic diag;
        diag.ruleName = getName().str();
        diag.message = "signal '" + std::string(signal->name) +
                       "' is assigned but never read";
        diag.severity = config.severity;
        diag.location = signal->location;
        diag.code = "unread_signal";
        diagnostics.push_back(std::move(diag));
      }
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// ImplicitWidthConversionRule Implementation
//===----------------------------------------------------------------------===//

ImplicitWidthConversionRule::ImplicitWidthConversionRule()
    : LintRule("implicit_width",
               "Detects implicit width conversions that may cause truncation") {
}

namespace {

class WidthConversionVisitor
    : public slang::ast::ASTVisitor<WidthConversionVisitor, true, true> {
public:
  std::vector<LintDiagnostic> diagnostics;
  LintSeverity severity;
  std::string ruleName;

  WidthConversionVisitor(LintSeverity sev, llvm::StringRef name)
      : severity(sev), ruleName(name.str()) {}

  void handle(const slang::ast::AssignmentExpression &expr) {
    auto &lhs = expr.left();
    auto &rhs = expr.right();

    if (!lhs.type->isIntegral() || !rhs.type->isIntegral()) {
      visitDefault(expr);
      return;
    }

    auto lhsWidth = lhs.type->getBitWidth();
    auto rhsWidth = rhs.type->getBitWidth();

    if (rhsWidth > lhsWidth) {
      LintDiagnostic diag;
      diag.ruleName = ruleName;
      diag.message = "implicit truncation from " + std::to_string(rhsWidth) +
                     " bits to " + std::to_string(lhsWidth) + " bits";
      diag.severity = severity;
      diag.location = expr.sourceRange.start();
      diag.range = expr.sourceRange;
      diag.code = "implicit_width";
      diag.fixSuggestion =
          "Use explicit truncation: " + std::to_string(lhsWidth - 1) + ":0";
      diagnostics.push_back(std::move(diag));
    }

    visitDefault(expr);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

} // namespace

std::vector<LintDiagnostic>
ImplicitWidthConversionRule::check(const slang::ast::Compilation &compilation,
                                   const LintRuleConfig &config) {
  WidthConversionVisitor visitor(config.severity, getName());

  for (auto *inst : compilation.getRoot().topInstances) {
    inst->body.visit(visitor);
  }

  return std::move(visitor.diagnostics);
}

//===----------------------------------------------------------------------===//
// BlockingInSequentialRule Implementation
//===----------------------------------------------------------------------===//

BlockingInSequentialRule::BlockingInSequentialRule()
    : LintRule("blocking_in_sequential",
               "Detects blocking assignments in sequential always blocks") {}

namespace {

class ProceduralBlockVisitor
    : public slang::ast::ASTVisitor<ProceduralBlockVisitor, true, true> {
public:
  std::vector<LintDiagnostic> diagnostics;
  LintSeverity severity;
  std::string ruleName;
  bool checkBlocking; // true = check for blocking in sequential
                      // false = check for non-blocking in combinational

  ProceduralBlockVisitor(LintSeverity sev, llvm::StringRef name, bool blocking)
      : severity(sev), ruleName(name.str()), checkBlocking(blocking) {}

  void handle(const slang::ast::ProceduralBlockSymbol &block) {
    bool isSequential =
        (block.procedureKind == slang::ast::ProceduralBlockKind::AlwaysFF);
    bool isCombinational =
        (block.procedureKind == slang::ast::ProceduralBlockKind::AlwaysComb);

    if ((checkBlocking && isSequential) ||
        (!checkBlocking && isCombinational)) {
      AssignmentTypeVisitor assignVisitor;
      block.visit(assignVisitor);

      for (const auto &[loc, isNonBlocking] : assignVisitor.assignments) {
        if (checkBlocking && !isNonBlocking) {
          // Blocking assignment in sequential block
          LintDiagnostic diag;
          diag.ruleName = ruleName;
          diag.message =
              "blocking assignment (=) used in sequential always_ff block";
          diag.severity = severity;
          diag.location = loc;
          diag.code = "blocking_in_sequential";
          diag.fixSuggestion = "Use non-blocking assignment (<=) instead";
          diagnostics.push_back(std::move(diag));
        } else if (!checkBlocking && isNonBlocking) {
          // Non-blocking assignment in combinational block
          LintDiagnostic diag;
          diag.ruleName = ruleName;
          diag.message =
              "non-blocking assignment (<=) used in combinational always_comb "
              "block";
          diag.severity = severity;
          diag.location = loc;
          diag.code = "nonblocking_in_combinational";
          diag.fixSuggestion = "Use blocking assignment (=) instead";
          diagnostics.push_back(std::move(diag));
        }
      }
    }

    visitDefault(block);
  }

  template <typename T>
  void handle(const T &node) {
    visitDefault(node);
  }
};

} // namespace

std::vector<LintDiagnostic>
BlockingInSequentialRule::check(const slang::ast::Compilation &compilation,
                                const LintRuleConfig &config) {
  ProceduralBlockVisitor visitor(config.severity, getName(),
                                 /*checkBlocking=*/true);

  for (auto *inst : compilation.getRoot().topInstances) {
    inst->body.visit(visitor);
  }

  return std::move(visitor.diagnostics);
}

//===----------------------------------------------------------------------===//
// NonBlockingInCombinationalRule Implementation
//===----------------------------------------------------------------------===//

NonBlockingInCombinationalRule::NonBlockingInCombinationalRule()
    : LintRule(
          "nonblocking_in_combinational",
          "Detects non-blocking assignments in combinational always blocks") {}

std::vector<LintDiagnostic> NonBlockingInCombinationalRule::check(
    const slang::ast::Compilation &compilation, const LintRuleConfig &config) {
  ProceduralBlockVisitor visitor(config.severity, getName(),
                                 /*checkBlocking=*/false);

  for (auto *inst : compilation.getRoot().topInstances) {
    inst->body.visit(visitor);
  }

  return std::move(visitor.diagnostics);
}

//===----------------------------------------------------------------------===//
// MissingDefaultCaseRule Implementation
//===----------------------------------------------------------------------===//

MissingDefaultCaseRule::MissingDefaultCaseRule()
    : LintRule("missing_default",
               "Detects case statements without a default clause") {}

std::vector<LintDiagnostic>
MissingDefaultCaseRule::check(const slang::ast::Compilation &compilation,
                              const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  for (auto *inst : compilation.getRoot().topInstances) {
    CaseStatementVisitor visitor;
    inst->body.visit(visitor);

    for (const auto *caseStmt : visitor.caseStatementsWithoutDefault) {
      LintDiagnostic diag;
      diag.ruleName = getName().str();
      diag.message = "case statement does not have a default clause";
      diag.severity = config.severity;
      diag.location = caseStmt->sourceRange.start();
      diag.range = caseStmt->sourceRange;
      diag.code = "missing_default";
      diag.fixSuggestion = "Add a 'default:' clause to handle all other cases";
      diagnostics.push_back(std::move(diag));
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// NamingConventionRule Implementation
//===----------------------------------------------------------------------===//

NamingConventionRule::NamingConventionRule()
    : LintRule("naming_convention",
               "Checks naming conventions against configurable patterns") {}

std::vector<LintDiagnostic>
NamingConventionRule::check(const slang::ast::Compilation &compilation,
                            const LintRuleConfig &config) {
  std::vector<LintDiagnostic> diagnostics;

  // Get naming patterns from config
  std::string modulePattern = "^[A-Z][a-zA-Z0-9_]*$";
  std::string signalPattern = "^[a-z][a-z0-9_]*$";
  std::string parameterPattern = "^[A-Z][A-Z0-9_]*$";
  std::string portPattern = "^[a-z][a-z0-9_]*(_i|_o|_io)?$";

  auto it = config.stringOptions.find("module_pattern");
  if (it != config.stringOptions.end())
    modulePattern = it->second;

  it = config.stringOptions.find("signal_pattern");
  if (it != config.stringOptions.end())
    signalPattern = it->second;

  it = config.stringOptions.find("parameter_pattern");
  if (it != config.stringOptions.end())
    parameterPattern = it->second;

  it = config.stringOptions.find("port_pattern");
  if (it != config.stringOptions.end())
    portPattern = it->second;

  // Also check for single "pattern" option that applies to all
  it = config.stringOptions.find("pattern");
  if (it != config.stringOptions.end()) {
    modulePattern = signalPattern = parameterPattern = portPattern = it->second;
  }

  for (auto *inst : compilation.getRoot().topInstances) {
    // Check module name
    if (!matchesPattern(inst->name, modulePattern)) {
      LintDiagnostic diag;
      diag.ruleName = getName().str();
      diag.message = "module name '" + std::string(inst->name) +
                     "' does not match naming convention";
      diag.severity = config.severity;
      diag.location = inst->location;
      diag.code = "naming_convention";
      diagnostics.push_back(std::move(diag));
    }

    // Check ports
    for (const auto *portSym : inst->body.getPortList()) {
      if (const auto *port = portSym->as_if<slang::ast::PortSymbol>()) {
        if (!matchesPattern(port->name, portPattern)) {
          LintDiagnostic diag;
          diag.ruleName = getName().str();
          diag.message = "port name '" + std::string(port->name) +
                         "' does not match naming convention";
          diag.severity = config.severity;
          diag.location = port->location;
          diag.code = "naming_convention";
          diagnostics.push_back(std::move(diag));
        }
      }
    }

    // Check members
    for (const auto &member : inst->body.members()) {
      if (member.name.empty())
        continue;

      bool matches = true;
      std::string symbolType;

      switch (member.kind) {
      case slang::ast::SymbolKind::Variable:
      case slang::ast::SymbolKind::Net:
        matches = matchesPattern(member.name, signalPattern);
        symbolType = "signal";
        break;
      case slang::ast::SymbolKind::Parameter:
        matches = matchesPattern(member.name, parameterPattern);
        symbolType = "parameter";
        break;
      default:
        continue;
      }

      if (!matches) {
        LintDiagnostic diag;
        diag.ruleName = getName().str();
        diag.message = symbolType + " name '" + std::string(member.name) +
                       "' does not match naming convention";
        diag.severity = config.severity;
        diag.location = member.location;
        diag.code = "naming_convention";
        diagnostics.push_back(std::move(diag));
      }
    }
  }

  return diagnostics;
}

//===----------------------------------------------------------------------===//
// IncompleteCaseRule Implementation
//===----------------------------------------------------------------------===//

IncompleteCaseRule::IncompleteCaseRule()
    : LintRule("incomplete_case",
               "Detects case statements that don't cover all enum values") {}

std::vector<LintDiagnostic>
IncompleteCaseRule::check(const slang::ast::Compilation &compilation,
                          const LintRuleConfig &config) {
  // This is a more complex check that requires analyzing enum coverage
  // For now, we rely on the missing_default check
  return {};
}

//===----------------------------------------------------------------------===//
// MultipleDriversRule Implementation
//===----------------------------------------------------------------------===//

MultipleDriversRule::MultipleDriversRule()
    : LintRule("multiple_drivers",
               "Detects signals driven in multiple always blocks") {}

std::vector<LintDiagnostic>
MultipleDriversRule::check(const slang::ast::Compilation &compilation,
                           const LintRuleConfig &config) {
  // This check requires tracking drivers across procedural blocks
  // which is complex to implement correctly
  return {};
}

//===----------------------------------------------------------------------===//
// LatchInferenceRule Implementation
//===----------------------------------------------------------------------===//

LatchInferenceRule::LatchInferenceRule()
    : LintRule("latch_inference",
               "Detects potential unintended latch inference") {}

std::vector<LintDiagnostic>
LatchInferenceRule::check(const slang::ast::Compilation &compilation,
                          const LintRuleConfig &config) {
  // Latch inference detection requires control flow analysis
  // This is a placeholder for future implementation
  return {};
}

//===----------------------------------------------------------------------===//
// LintRuleRegistry Implementation
//===----------------------------------------------------------------------===//

LintRuleRegistry::LintRuleRegistry() { registerBuiltinRules(); }

LintRuleRegistry::~LintRuleRegistry() = default;

LintRuleRegistry &LintRuleRegistry::getInstance() {
  static LintRuleRegistry instance;
  return instance;
}

void LintRuleRegistry::registerRule(std::unique_ptr<LintRule> rule) {
  ruleMap[rule->getName()] = rule.get();
  rules.push_back(std::move(rule));
}

LintRule *LintRuleRegistry::getRule(llvm::StringRef name) const {
  auto it = ruleMap.find(name);
  if (it != ruleMap.end())
    return it->second;
  return nullptr;
}

std::vector<std::string> LintRuleRegistry::getRuleNames() const {
  std::vector<std::string> names;
  names.reserve(rules.size());
  for (const auto &rule : rules)
    names.push_back(rule->getName().str());
  return names;
}

std::vector<LintRule *>
LintRuleRegistry::getRulesByCategory(llvm::StringRef category) const {
  std::vector<LintRule *> result;
  for (const auto &rule : rules) {
    if (rule->getCategory() == category)
      result.push_back(rule.get());
  }
  return result;
}

void LintRuleRegistry::registerBuiltinRules() {
  registerRule(std::make_unique<UnusedSignalRule>());
  registerRule(std::make_unique<UnusedParameterRule>());
  registerRule(std::make_unique<UndrivenSignalRule>());
  registerRule(std::make_unique<UnreadSignalRule>());
  registerRule(std::make_unique<ImplicitWidthConversionRule>());
  registerRule(std::make_unique<BlockingInSequentialRule>());
  registerRule(std::make_unique<NonBlockingInCombinationalRule>());
  registerRule(std::make_unique<MissingDefaultCaseRule>());
  registerRule(std::make_unique<NamingConventionRule>());
  registerRule(std::make_unique<IncompleteCaseRule>());
  registerRule(std::make_unique<MultipleDriversRule>());
  registerRule(std::make_unique<LatchInferenceRule>());
}

//===----------------------------------------------------------------------===//
// LintRunner Implementation
//===----------------------------------------------------------------------===//

LintRunner::LintRunner(const LintConfig &config) : config(config) {}

LintResults LintRunner::run(const slang::ast::Compilation &compilation) {
  LintResults results;

  auto &registry = LintRuleRegistry::getInstance();

  for (const auto &rule : registry.getAllRules()) {
    if (!config.isRuleEnabled(rule->getName()))
      continue;

    if (!rule->isApplicable(compilation))
      continue;

    auto ruleConfig = config.getRuleConfig(rule->getName());
    auto ruleDiags = rule->check(compilation, ruleConfig);

    for (auto &diag : ruleDiags) {
      switch (diag.severity) {
      case LintSeverity::Error:
        ++results.errorCount;
        break;
      case LintSeverity::Warning:
        ++results.warningCount;
        break;
      case LintSeverity::Hint:
        ++results.hintCount;
        break;
      case LintSeverity::Ignore:
        continue; // Skip ignored diagnostics
      }
      results.diagnostics.push_back(std::move(diag));
    }
  }

  return results;
}

std::vector<LintDiagnostic>
LintRunner::runRule(llvm::StringRef ruleName,
                    const slang::ast::Compilation &compilation) {
  auto &registry = LintRuleRegistry::getInstance();
  auto *rule = registry.getRule(ruleName);

  if (!rule)
    return {};

  auto ruleConfig = config.getRuleConfig(ruleName);
  return rule->check(compilation, ruleConfig);
}
