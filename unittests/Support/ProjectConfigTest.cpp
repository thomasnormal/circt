//===- ProjectConfigTest.cpp - Unit tests for ProjectConfig ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ProjectConfig.h"
#include "gtest/gtest.h"

using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// YAML Parsing Tests
//===----------------------------------------------------------------------===//

TEST(ProjectConfigTest, EmptyConfig) {
  auto configOrErr = ProjectConfig::loadFromYAML("");
  ASSERT_TRUE(static_cast<bool>(configOrErr));
  EXPECT_TRUE((*configOrErr)->isEmpty());
}

TEST(ProjectConfigTest, BasicProjectInfo) {
  const char *yaml = R"(
project:
  name: "my_design"
  top: "top_module"
  version: "1.0.0"
  description: "A test design"
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  const auto &info = (*configOrErr)->getProjectInfo();
  EXPECT_EQ(info.name, "my_design");
  EXPECT_EQ(info.topModule, "top_module");
  EXPECT_EQ(info.version, "1.0.0");
  EXPECT_EQ(info.description, "A test design");
}

TEST(ProjectConfigTest, SourceConfiguration) {
  const char *yaml = R"(
sources:
  include_dirs:
    - "rtl/"
    - "includes/"
  defines:
    - "SYNTHESIS"
    - "DEBUG_LEVEL=2"
  files:
    - "rtl/*.sv"
    - "tb/*.sv"
  file_lists:
    - "project.f"
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  const auto &src = (*configOrErr)->getSourceConfig();
  ASSERT_EQ(src.includeDirs.size(), 2u);
  EXPECT_EQ(src.includeDirs[0], "rtl/");
  EXPECT_EQ(src.includeDirs[1], "includes/");

  ASSERT_EQ(src.defines.size(), 2u);
  EXPECT_EQ(src.defines[0], "SYNTHESIS");
  EXPECT_EQ(src.defines[1], "DEBUG_LEVEL=2");

  ASSERT_EQ(src.files.size(), 2u);
  EXPECT_EQ(src.files[0], "rtl/*.sv");
  EXPECT_EQ(src.files[1], "tb/*.sv");

  ASSERT_EQ(src.fileLists.size(), 1u);
  EXPECT_EQ(src.fileLists[0], "project.f");
}

TEST(ProjectConfigTest, LintConfiguration) {
  const char *yaml = R"(
lint:
  enabled: true
  config: "lint.yaml"
  enable_rules:
    - "unused_signal"
    - "naming_convention"
  disable_rules:
    - "implicit_width"
  exclude_patterns:
    - "generated/**"
    - "third_party/**"
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  const auto &lint = (*configOrErr)->getLintingConfig();
  EXPECT_TRUE(lint.enabled);
  EXPECT_EQ(lint.configFile, "lint.yaml");

  ASSERT_EQ(lint.enableRules.size(), 2u);
  EXPECT_EQ(lint.enableRules[0], "unused_signal");
  EXPECT_EQ(lint.enableRules[1], "naming_convention");

  ASSERT_EQ(lint.disableRules.size(), 1u);
  EXPECT_EQ(lint.disableRules[0], "implicit_width");

  ASSERT_EQ(lint.excludePatterns.size(), 2u);
  EXPECT_EQ(lint.excludePatterns[0], "generated/**");
}

TEST(ProjectConfigTest, SimulationConfiguration) {
  const char *yaml = R"(
simulation:
  timescale: "1ns/1ps"
  default_clocking: "clk"
  default_reset: "rst_n"
  reset_active_high: false
  defines:
    - "SIMULATION"
    - "VCS"
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  const auto &sim = (*configOrErr)->getSimulationConfig();
  EXPECT_EQ(sim.timescale, "1ns/1ps");
  EXPECT_EQ(sim.defaultClocking, "clk");
  EXPECT_EQ(sim.defaultReset, "rst_n");
  EXPECT_FALSE(sim.resetActiveHigh);

  ASSERT_EQ(sim.defines.size(), 2u);
  EXPECT_EQ(sim.defines[0], "SIMULATION");
  EXPECT_EQ(sim.defines[1], "VCS");
}

TEST(ProjectConfigTest, TargetConfiguration) {
  const char *yaml = R"(
targets:
  synthesis:
    top: "chip_top"
    defines:
      - "SYNTHESIS"
    lint_enabled: false
  simulation:
    top: "tb_top"
    defines:
      - "SIMULATION"
      - "VCS"
    include_dirs:
      - "tb/includes"
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  auto names = (*configOrErr)->getTargetNames();
  EXPECT_EQ(names.size(), 2u);

  const auto *synth = (*configOrErr)->getTarget("synthesis");
  ASSERT_NE(synth, nullptr);
  EXPECT_EQ(synth->name, "synthesis");
  EXPECT_EQ(synth->topModule, "chip_top");
  ASSERT_EQ(synth->defines.size(), 1u);
  EXPECT_EQ(synth->defines[0], "SYNTHESIS");
  ASSERT_TRUE(synth->lintEnabled.has_value());
  EXPECT_FALSE(*synth->lintEnabled);

  const auto *sim = (*configOrErr)->getTarget("simulation");
  ASSERT_NE(sim, nullptr);
  EXPECT_EQ(sim->name, "simulation");
  EXPECT_EQ(sim->topModule, "tb_top");
  ASSERT_EQ(sim->defines.size(), 2u);
  ASSERT_EQ(sim->includeDirs.size(), 1u);
}

TEST(ProjectConfigTest, FullConfiguration) {
  const char *yaml = R"(
project:
  name: "full_project"
  top: "main_top"
  version: "2.0.0"

sources:
  include_dirs:
    - "src/includes"
  defines:
    - "FPGA"
  files:
    - "src/**/*.sv"

lint:
  enabled: true
  config: "lint-rules.yaml"

simulation:
  timescale: "10ns/1ps"

targets:
  fpga:
    top: "fpga_top"
    defines: ["XILINX"]
)";

  auto configOrErr = ProjectConfig::loadFromYAML(yaml);
  ASSERT_TRUE(static_cast<bool>(configOrErr));

  EXPECT_EQ((*configOrErr)->getProjectInfo().name, "full_project");
  EXPECT_EQ((*configOrErr)->getSourceConfig().includeDirs.size(), 1u);
  EXPECT_TRUE((*configOrErr)->getLintingConfig().enabled);
  EXPECT_EQ((*configOrErr)->getSimulationConfig().timescale, "10ns/1ps");
  EXPECT_NE((*configOrErr)->getTarget("fpga"), nullptr);
}

//===----------------------------------------------------------------------===//
// File List Parsing Tests
//===----------------------------------------------------------------------===//

TEST(ProjectConfigTest, FileListParsing) {
  const char *content = R"(
// Comment line
# Another comment
+incdir+rtl/includes
+incdir+tb/includes
+define+SYNTHESIS
+define+DEBUG_LEVEL=2
-y lib/cells
-f dependencies.f
-top top_module
rtl/module1.sv
rtl/module2.sv
tb/testbench.sv
)";

  auto entriesOrErr = ProjectConfig::parseFileListContent(content);
  ASSERT_TRUE(static_cast<bool>(entriesOrErr));

  const auto &entries = *entriesOrErr;
  // Count each type
  int fileCount = 0, incCount = 0, defCount = 0;
  int libCount = 0, fListCount = 0, topCount = 0;

  for (const auto &e : entries) {
    switch (e.kind) {
    case FileListEntry::Kind::File:
      fileCount++;
      break;
    case FileListEntry::Kind::IncludeDir:
      incCount++;
      break;
    case FileListEntry::Kind::Define:
      defCount++;
      break;
    case FileListEntry::Kind::LibDir:
      libCount++;
      break;
    case FileListEntry::Kind::FileList:
      fListCount++;
      break;
    case FileListEntry::Kind::TopModule:
      topCount++;
      break;
    default:
      break;
    }
  }

  EXPECT_EQ(fileCount, 3);
  EXPECT_EQ(incCount, 2);
  EXPECT_EQ(defCount, 2);
  EXPECT_EQ(libCount, 1);
  EXPECT_EQ(fListCount, 1);
  EXPECT_EQ(topCount, 1);
}

TEST(ProjectConfigTest, FileListDefineWithValue) {
  const char *content = "+define+SOME_VALUE=42\n";

  auto entriesOrErr = ProjectConfig::parseFileListContent(content);
  ASSERT_TRUE(static_cast<bool>(entriesOrErr));

  ASSERT_EQ(entriesOrErr->size(), 1u);
  EXPECT_EQ((*entriesOrErr)[0].kind, FileListEntry::Kind::Define);
  EXPECT_EQ((*entriesOrErr)[0].value, "SOME_VALUE");
  EXPECT_EQ((*entriesOrErr)[0].extra, "42");
}

//===----------------------------------------------------------------------===//
// Define Parsing Tests
//===----------------------------------------------------------------------===//

TEST(ProjectConfigTest, ParseDefineWithoutValue) {
  auto [name, value] = parseDefine("SYNTHESIS");
  EXPECT_EQ(name, "SYNTHESIS");
  EXPECT_EQ(value, "");
}

TEST(ProjectConfigTest, ParseDefineWithValue) {
  auto [name, value] = parseDefine("DEBUG_LEVEL=2");
  EXPECT_EQ(name, "DEBUG_LEVEL");
  EXPECT_EQ(value, "2");
}

TEST(ProjectConfigTest, FormatDefine) {
  EXPECT_EQ(formatDefine("SYNTHESIS"), "SYNTHESIS");
  EXPECT_EQ(formatDefine("DEBUG", "1"), "DEBUG=1");
}

//===----------------------------------------------------------------------===//
// Serialization Tests
//===----------------------------------------------------------------------===//

TEST(ProjectConfigTest, ToYAML) {
  auto config = ProjectConfig::createDefault();

  ProjectInfo info;
  info.name = "test_project";
  info.topModule = "top";
  config->setProjectInfo(info);

  SourceConfig src;
  src.includeDirs = {"rtl/", "includes/"};
  src.defines = {"SYNTHESIS"};
  config->setSourceConfig(src);

  std::string yaml = config->toYAML();

  // Verify the YAML contains expected content
  EXPECT_NE(yaml.find("test_project"), std::string::npos);
  EXPECT_NE(yaml.find("top"), std::string::npos);
  EXPECT_NE(yaml.find("rtl/"), std::string::npos);
  EXPECT_NE(yaml.find("SYNTHESIS"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Utility Tests
//===----------------------------------------------------------------------===//

TEST(ProjectConfigTest, IsProjectConfigFile) {
  EXPECT_TRUE(isProjectConfigFile("circt-project.yaml"));
  EXPECT_TRUE(isProjectConfigFile(".circt-project.yaml"));
  EXPECT_TRUE(isProjectConfigFile("circt.yaml"));
  EXPECT_TRUE(isProjectConfigFile("circt.yml"));
  EXPECT_FALSE(isProjectConfigFile("project.yaml"));
  EXPECT_FALSE(isProjectConfigFile("circt.txt"));
}

TEST(ProjectConfigTest, GetAllDefines) {
  auto config = std::make_unique<ProjectConfig>();

  SourceConfig src;
  src.defines = {"FPGA", "DEBUG=1"};
  config->setSourceConfig(src);

  SimulationConfig sim;
  sim.defines = {"SIMULATION"};
  config->setSimulationConfig(sim);

  auto defines = config->getAllDefines();
  ASSERT_EQ(defines.size(), 3u);
  EXPECT_EQ(defines[0], "FPGA");
  EXPECT_EQ(defines[1], "DEBUG=1");
  EXPECT_EQ(defines[2], "SIMULATION");
}

} // namespace
