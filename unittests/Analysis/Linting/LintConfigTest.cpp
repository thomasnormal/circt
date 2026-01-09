//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "circt/Analysis/Linting/LintConfig.h"

using namespace circt::lint;

namespace {

/// Test default configuration creation.
TEST(LintConfigTest, DefaultConfig) {
  auto config = LintConfig::createDefault();

  EXPECT_TRUE(config->isRuleEnabled("unused_signal"));
  EXPECT_TRUE(config->isRuleEnabled("implicit_width"));
  EXPECT_EQ(config->getRuleSeverity("unused_signal"), LintSeverity::Warning);
}

/// Test setting and getting rule configuration.
TEST(LintConfigTest, SetRuleConfig) {
  LintConfig config;

  LintRuleConfig ruleConfig;
  ruleConfig.severity = LintSeverity::Error;
  ruleConfig.enabled = true;
  config.setRuleConfig("test_rule", ruleConfig);

  EXPECT_TRUE(config.isRuleEnabled("test_rule"));
  EXPECT_EQ(config.getRuleSeverity("test_rule"), LintSeverity::Error);
}

/// Test enabling and disabling rules.
TEST(LintConfigTest, EnableDisableRules) {
  LintConfig config;

  config.setRuleEnabled("my_rule", true);
  EXPECT_TRUE(config.isRuleEnabled("my_rule"));

  config.setRuleEnabled("my_rule", false);
  EXPECT_FALSE(config.isRuleEnabled("my_rule"));
}

/// Test severity changes.
TEST(LintConfigTest, SeverityChanges) {
  LintConfig config;

  config.setRuleSeverity("test", LintSeverity::Warning);
  EXPECT_EQ(config.getRuleSeverity("test"), LintSeverity::Warning);
  EXPECT_TRUE(config.isRuleEnabled("test"));

  config.setRuleSeverity("test", LintSeverity::Ignore);
  EXPECT_EQ(config.getRuleSeverity("test"), LintSeverity::Ignore);
  EXPECT_FALSE(config.isRuleEnabled("test"));
}

/// Test severity string parsing.
TEST(LintConfigTest, ParseSeverity) {
  EXPECT_EQ(parseSeverity("warning"), LintSeverity::Warning);
  EXPECT_EQ(parseSeverity("Warning"), LintSeverity::Warning);
  EXPECT_EQ(parseSeverity("WARN"), LintSeverity::Warning);
  EXPECT_EQ(parseSeverity("error"), LintSeverity::Error);
  EXPECT_EQ(parseSeverity("hint"), LintSeverity::Hint);
  EXPECT_EQ(parseSeverity("info"), LintSeverity::Hint);
  EXPECT_EQ(parseSeverity("ignore"), LintSeverity::Ignore);
  EXPECT_EQ(parseSeverity("off"), LintSeverity::Ignore);
  EXPECT_EQ(parseSeverity("disabled"), LintSeverity::Ignore);
  EXPECT_EQ(parseSeverity("invalid"), std::nullopt);
}

/// Test severity to string conversion.
TEST(LintConfigTest, SeverityToString) {
  EXPECT_EQ(severityToString(LintSeverity::Warning), "warning");
  EXPECT_EQ(severityToString(LintSeverity::Error), "error");
  EXPECT_EQ(severityToString(LintSeverity::Hint), "hint");
  EXPECT_EQ(severityToString(LintSeverity::Ignore), "ignore");
}

/// Test exclude pattern matching.
TEST(LintConfigTest, ExcludePatterns) {
  LintConfig config;

  config.addExcludePattern("*.gen.sv");
  config.addExcludePattern("test_*");

  EXPECT_TRUE(config.shouldExcludeFile("module.gen.sv"));
  EXPECT_TRUE(config.shouldExcludeFile("test_foo.sv"));
  EXPECT_FALSE(config.shouldExcludeFile("module.sv"));
  EXPECT_FALSE(config.shouldExcludeFile("foo_test.sv"));
}

/// Test YAML loading.
TEST(LintConfigTest, LoadFromYAML) {
  const char *yaml = R"(
rules:
  unused_signal: warning
  implicit_width: error
  naming_convention:
    severity: hint
    pattern: "^[a-z][a-z0-9_]*$"
naming:
  module_pattern: "^[A-Z][a-zA-Z0-9_]*$"
  signal_pattern: "^[a-z][a-z0-9_]*$"
exclude:
  - "*.gen.sv"
  - "testbench/*"
)";

  auto configOrErr = LintConfig::loadFromYAML(yaml);
  ASSERT_FALSE(!configOrErr);

  auto &config = *configOrErr;
  EXPECT_EQ(config->getRuleSeverity("unused_signal"), LintSeverity::Warning);
  EXPECT_EQ(config->getRuleSeverity("implicit_width"), LintSeverity::Error);
  EXPECT_EQ(config->getRuleSeverity("naming_convention"), LintSeverity::Hint);

  const auto &namingConfig = config->getNamingConfig();
  EXPECT_EQ(namingConfig.modulePattern, "^[A-Z][a-zA-Z0-9_]*$");
  EXPECT_EQ(namingConfig.signalPattern, "^[a-z][a-z0-9_]*$");

  EXPECT_TRUE(config->shouldExcludeFile("module.gen.sv"));
}

/// Test configuration merging.
TEST(LintConfigTest, MergeConfig) {
  LintConfig base;
  base.setRuleSeverity("rule1", LintSeverity::Warning);
  base.setRuleSeverity("rule2", LintSeverity::Error);

  LintConfig override;
  override.setRuleSeverity("rule2", LintSeverity::Hint);
  override.setRuleSeverity("rule3", LintSeverity::Warning);

  base.merge(override);

  EXPECT_EQ(base.getRuleSeverity("rule1"), LintSeverity::Warning);
  EXPECT_EQ(base.getRuleSeverity("rule2"), LintSeverity::Hint);  // Overridden
  EXPECT_EQ(base.getRuleSeverity("rule3"), LintSeverity::Warning);  // Added
}

/// Test disable all rules.
TEST(LintConfigTest, DisableAllRules) {
  auto config = LintConfig::createDefault();

  config->disableAllRules();

  EXPECT_FALSE(config->isRuleEnabled("unused_signal"));
  EXPECT_FALSE(config->isRuleEnabled("implicit_width"));
}

/// Test enable all rules at specific severity.
TEST(LintConfigTest, EnableAllRulesAtSeverity) {
  LintConfig config;

  config.enableAllRules(LintSeverity::Error);

  EXPECT_TRUE(config.isRuleEnabled("unused_signal"));
  EXPECT_EQ(config.getRuleSeverity("unused_signal"), LintSeverity::Error);
}

/// Test empty YAML produces valid empty config.
TEST(LintConfigTest, EmptyYAML) {
  auto configOrErr = LintConfig::loadFromYAML("");
  ASSERT_FALSE(!configOrErr);
}

/// Test rule-specific options.
TEST(LintConfigTest, RuleSpecificOptions) {
  const char *yaml = R"(
rules:
  naming_convention:
    severity: hint
    pattern: "^m_.*$"
    max_length: 30
)";

  auto configOrErr = LintConfig::loadFromYAML(yaml);
  ASSERT_FALSE(!configOrErr);

  auto &config = *configOrErr;
  const auto &ruleConfig = config->getRuleConfig("naming_convention");

  auto patternIt = ruleConfig.stringOptions.find("pattern");
  ASSERT_NE(patternIt, ruleConfig.stringOptions.end());
  EXPECT_EQ(patternIt->second, "^m_.*$");
}

/// Test getting configured rules list.
TEST(LintConfigTest, GetConfiguredRules) {
  LintConfig config;
  config.setRuleSeverity("rule_a", LintSeverity::Warning);
  config.setRuleSeverity("rule_b", LintSeverity::Error);

  auto rules = config.getConfiguredRules();

  EXPECT_EQ(rules.size(), 2u);
  EXPECT_TRUE(std::find(rules.begin(), rules.end(), "rule_a") != rules.end());
  EXPECT_TRUE(std::find(rules.begin(), rules.end(), "rule_b") != rules.end());
}

} // namespace
