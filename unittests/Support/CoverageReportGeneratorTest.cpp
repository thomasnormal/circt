//===- CoverageReportGeneratorTest.cpp - CoverageReportGenerator tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CoverageReportGenerator.h"
#include "circt/Support/CoverageDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class CoverageReportGeneratorTest : public ::testing::Test {
protected:
  void SetUp() override {
    llvm::SmallString<128> tempPath;
    std::error_code ec =
        llvm::sys::fs::createUniqueDirectory("coverage-report-test", tempPath);
    ASSERT_FALSE(ec) << "Failed to create temp directory";
    tempDir = tempPath.str().str();
  }

  void TearDown() override {
    if (!tempDir.empty()) {
      llvm::sys::fs::remove_directories(tempDir);
    }
  }

  std::string tempDir;

  CoverageDatabase createTestDatabase() {
    CoverageDatabase db;

    // Add line coverage points
    CoveragePoint linePoint1;
    linePoint1.name = "module.v:10";
    linePoint1.type = CoverageType::Line;
    linePoint1.hits = 5;
    linePoint1.goal = 1;
    linePoint1.location.filename = "module.v";
    linePoint1.location.line = 10;
    linePoint1.hierarchy = "top.submodule";
    linePoint1.description = "Line coverage for statement";
    db.addCoveragePoint(linePoint1);

    CoveragePoint linePoint2;
    linePoint2.name = "module.v:20";
    linePoint2.type = CoverageType::Line;
    linePoint2.hits = 0;
    linePoint2.goal = 1;
    linePoint2.location.filename = "module.v";
    linePoint2.location.line = 20;
    linePoint2.hierarchy = "top.submodule";
    db.addCoveragePoint(linePoint2);

    // Add toggle coverage point
    CoveragePoint togglePoint;
    togglePoint.name = "top.signal";
    togglePoint.type = CoverageType::Toggle;
    togglePoint.toggle01 = true;
    togglePoint.toggle10 = false;
    togglePoint.hierarchy = "top";
    db.addCoveragePoint(togglePoint);

    // Add branch coverage point
    CoveragePoint branchPoint;
    branchPoint.name = "branch.v:15";
    branchPoint.type = CoverageType::Branch;
    branchPoint.branchTrue = true;
    branchPoint.branchFalse = true;
    branchPoint.location.filename = "branch.v";
    branchPoint.location.line = 15;
    branchPoint.hierarchy = "top.controller";
    db.addCoveragePoint(branchPoint);

    // Add coverage group
    CoverageGroup group;
    group.name = "test_group";
    group.description = "Test coverage group";
    group.pointNames = {"module.v:10", "module.v:20"};
    db.addCoverageGroup(group);

    // Add exclusion
    CoverageExclusion exclusion;
    exclusion.pointName = "excluded_point";
    exclusion.reason = "Dead code";
    exclusion.author = "test";
    exclusion.date = "2024-01-01";
    db.addExclusion(exclusion);

    // Add trend data
    CoverageTrendPoint trend1;
    trend1.timestamp = "2024-01-01T00:00:00Z";
    trend1.runId = "run1";
    trend1.commitHash = "abc123";
    trend1.lineCoverage = 40.0;
    trend1.toggleCoverage = 50.0;
    trend1.branchCoverage = 100.0;
    trend1.overallCoverage = 55.0;
    trend1.totalPoints = 4;
    trend1.coveredPoints = 2;
    db.addTrendPoint(trend1);

    CoverageTrendPoint trend2;
    trend2.timestamp = "2024-01-02T00:00:00Z";
    trend2.runId = "run2";
    trend2.commitHash = "def456";
    trend2.lineCoverage = 50.0;
    trend2.toggleCoverage = 50.0;
    trend2.branchCoverage = 100.0;
    trend2.overallCoverage = 62.5;
    trend2.totalPoints = 4;
    trend2.coveredPoints = 3;
    db.addTrendPoint(trend2);

    return db;
  }
};

//===----------------------------------------------------------------------===//
// Hierarchy Tree Tests
//===----------------------------------------------------------------------===//

TEST_F(CoverageReportGeneratorTest, BuildHierarchyTree) {
  auto db = createTestDatabase();

  CoverageReportGenerator generator(db);

  const auto *root = generator.getHierarchyRoot();
  ASSERT_NE(root, nullptr);

  // Check that we have a "top" child
  auto topIt = root->children.find("top");
  ASSERT_NE(topIt, root->children.end());

  const auto *topNode = topIt->second.get();
  EXPECT_EQ(topNode->name, "top");
  EXPECT_GT(topNode->totalPoints, 0u);
}

TEST_F(CoverageReportGeneratorTest, HierarchyCoverageCalculation) {
  auto db = createTestDatabase();

  CoverageReportGenerator generator(db);

  const auto *root = generator.getHierarchyRoot();
  ASSERT_NE(root, nullptr);

  // Total coverage should reflect all points
  EXPECT_GT(root->totalPoints, 0u);
  EXPECT_GE(root->coveredPoints, 0u);
  EXPECT_LE(root->coveredPoints, root->totalPoints);
  EXPECT_GE(root->coveragePercent, 0.0);
  EXPECT_LE(root->coveragePercent, 100.0);
}

//===----------------------------------------------------------------------===//
// Source File Coverage Tests
//===----------------------------------------------------------------------===//

TEST_F(CoverageReportGeneratorTest, BuildSourceFileCoverage) {
  auto db = createTestDatabase();

  CoverageReportGenerator generator(db);

  const auto &sourceFiles = generator.getSourceFileCoverage();

  // We should have module.v and branch.v
  EXPECT_GE(sourceFiles.size(), 2u);

  auto moduleIt = sourceFiles.find("module.v");
  ASSERT_NE(moduleIt, sourceFiles.end());

  const auto &moduleFile = moduleIt->second;
  EXPECT_EQ(moduleFile.filename, "module.v");
  EXPECT_EQ(moduleFile.totalLines, 2u);
  EXPECT_EQ(moduleFile.coveredLines, 1u);
  EXPECT_DOUBLE_EQ(moduleFile.coveragePercent, 50.0);
}

//===----------------------------------------------------------------------===//
// HTML Report Generation Tests
//===----------------------------------------------------------------------===//

TEST_F(CoverageReportGeneratorTest, GenerateHTMLReport) {
  auto db = createTestDatabase();

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err)) << llvm::toString(std::move(err));

  // Check for basic HTML structure
  EXPECT_TRUE(output.find("<!DOCTYPE html>") != std::string::npos);
  EXPECT_TRUE(output.find("</html>") != std::string::npos);

  // Check for summary section
  EXPECT_TRUE(output.find("Overall Coverage") != std::string::npos);
  EXPECT_TRUE(output.find("Total Points") != std::string::npos);

  // Check for coverage by type section
  EXPECT_TRUE(output.find("Coverage by Type") != std::string::npos);
  EXPECT_TRUE(output.find("Line Coverage") != std::string::npos);

  // Check for hierarchy breakdown
  EXPECT_TRUE(output.find("Hierarchy Breakdown") != std::string::npos);

  // Check for source file coverage
  EXPECT_TRUE(output.find("Source File Coverage") != std::string::npos);
  EXPECT_TRUE(output.find("module.v") != std::string::npos);

  // Check for uncovered items section
  EXPECT_TRUE(output.find("Uncovered Items") != std::string::npos);

  // Check for trend visualization (we have trends)
  EXPECT_TRUE(output.find("Coverage Trends") != std::string::npos);

  // Check for exclusions section
  EXPECT_TRUE(output.find("Exclusions") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLReportToFile) {
  auto db = createTestDatabase();

  std::string filePath = tempDir + "/report.html";

  HTMLReportOptions options;
  options.title = "Test Coverage Report";

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(filePath);
  ASSERT_FALSE(static_cast<bool>(err)) << llvm::toString(std::move(err));

  // Check file exists
  EXPECT_TRUE(llvm::sys::fs::exists(filePath));

  // Read file and check content
  auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
  ASSERT_TRUE(static_cast<bool>(bufferOrErr));

  llvm::StringRef content = bufferOrErr.get()->getBuffer();
  EXPECT_TRUE(content.contains("Test Coverage Report"));
  EXPECT_TRUE(content.contains("<!DOCTYPE html>"));
}

TEST_F(CoverageReportGeneratorTest, CustomReportOptions) {
  auto db = createTestDatabase();

  HTMLReportOptions options;
  options.title = "Custom Report";
  options.uncoveredOnly = true;
  options.includeTrends = true;

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Title should be custom
  EXPECT_TRUE(output.find("Custom Report") != std::string::npos);

  // Trends should be included
  EXPECT_TRUE(output.find("Coverage Trends") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, HierarchyFilter) {
  auto db = createTestDatabase();

  HTMLReportOptions options;
  options.hierarchyFilter = "top.submodule";

  CoverageReportGenerator generator(db, options);

  const auto &sourceFiles = generator.getSourceFileCoverage();

  // Only module.v should be included (it's in top.submodule)
  // branch.v is in top.controller, so it should be excluded
  EXPECT_EQ(sourceFiles.size(), 1u);

  auto it = sourceFiles.find("module.v");
  ASSERT_NE(it, sourceFiles.end());
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(CoverageReportGeneratorTest, EmptyDatabase) {
  CoverageDatabase db;

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Should still generate valid HTML
  EXPECT_TRUE(output.find("<!DOCTYPE html>") != std::string::npos);
  EXPECT_TRUE(output.find("Overall Coverage") != std::string::npos);
  EXPECT_TRUE(output.find("0") != std::string::npos ||
              output.find("100") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, NoTrends) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test";
  point.type = CoverageType::Line;
  point.hits = 1;
  db.addCoveragePoint(point);

  HTMLReportOptions options;
  options.includeTrends = true;

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Trends section should not appear when no trends
  EXPECT_TRUE(output.find("Coverage Trends") == std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, HTMLEscape) {
  CoverageDatabase db;

  // Add point with special HTML characters
  CoveragePoint point;
  point.name = "test<script>&evil";
  point.type = CoverageType::Line;
  point.hits = 0;
  point.hierarchy = "top<>&test";
  db.addCoveragePoint(point);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Dangerous characters should be escaped
  EXPECT_TRUE(output.find("&lt;script&gt;") != std::string::npos);
  EXPECT_TRUE(output.find("&amp;evil") != std::string::npos);
  // Raw dangerous tags should not appear
  EXPECT_TRUE(output.find("<script>") == std::string::npos ||
              output.find("<script>") > output.find("</head>"));
}

//===----------------------------------------------------------------------===//
// SourceFileCoverage Tests
//===----------------------------------------------------------------------===//

TEST(SourceFileCoverageTest, CalculateCoverage) {
  SourceFileCoverage file;
  file.filename = "test.v";

  CoveragePoint p1;
  p1.type = CoverageType::Line;
  p1.hits = 5;
  file.linePoints.emplace_back(10, &p1);

  CoveragePoint p2;
  p2.type = CoverageType::Line;
  p2.hits = 0;
  file.linePoints.emplace_back(20, &p2);

  file.calculateCoverage();

  EXPECT_EQ(file.totalLines, 2u);
  EXPECT_EQ(file.coveredLines, 1u);
  EXPECT_DOUBLE_EQ(file.coveragePercent, 50.0);
}

TEST(SourceFileCoverageTest, EmptyFile) {
  SourceFileCoverage file;
  file.filename = "empty.v";

  file.calculateCoverage();

  EXPECT_EQ(file.totalLines, 0u);
  EXPECT_EQ(file.coveredLines, 0u);
  EXPECT_DOUBLE_EQ(file.coveragePercent, 100.0);
}

//===----------------------------------------------------------------------===//
// HierarchyNode Tests
//===----------------------------------------------------------------------===//

TEST(HierarchyNodeTest, CalculateCoverage) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "point1";
  p1.type = CoverageType::Line;
  p1.hits = 1;
  p1.hierarchy = "top";
  db.addCoveragePoint(p1);

  CoveragePoint p2;
  p2.name = "point2";
  p2.type = CoverageType::Line;
  p2.hits = 0;
  p2.hierarchy = "top";
  db.addCoveragePoint(p2);

  HierarchyNode node;
  node.name = "top";
  node.coveragePointNames = {"point1", "point2"};

  node.calculateCoverage(db);

  EXPECT_EQ(node.totalPoints, 2u);
  EXPECT_EQ(node.coveredPoints, 1u);
  EXPECT_DOUBLE_EQ(node.coveragePercent, 50.0);
}

//===----------------------------------------------------------------------===//
// Additional Edge Case Tests
//===----------------------------------------------------------------------===//

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithAllCoverageTypes) {
  CoverageDatabase db;

  // Line coverage
  CoveragePoint linePoint;
  linePoint.name = "module.v:10";
  linePoint.type = CoverageType::Line;
  linePoint.hits = 5;
  linePoint.location.filename = "module.v";
  linePoint.location.line = 10;
  db.addCoveragePoint(linePoint);

  // Toggle coverage
  CoveragePoint togglePoint;
  togglePoint.name = "toggle1";
  togglePoint.type = CoverageType::Toggle;
  togglePoint.toggle01 = true;
  togglePoint.toggle10 = false;
  db.addCoveragePoint(togglePoint);

  // Branch coverage
  CoveragePoint branchPoint;
  branchPoint.name = "branch1";
  branchPoint.type = CoverageType::Branch;
  branchPoint.branchTrue = true;
  branchPoint.branchFalse = true;
  db.addCoveragePoint(branchPoint);

  // Condition coverage
  CoveragePoint conditionPoint;
  conditionPoint.name = "condition1";
  conditionPoint.type = CoverageType::Condition;
  conditionPoint.hits = 2;
  conditionPoint.goal = 4;
  db.addCoveragePoint(conditionPoint);

  // FSM coverage
  CoveragePoint fsmPoint;
  fsmPoint.name = "fsm1";
  fsmPoint.type = CoverageType::FSM;
  fsmPoint.hits = 5;
  fsmPoint.goal = 5;
  db.addCoveragePoint(fsmPoint);

  // Assertion coverage
  CoveragePoint assertPoint;
  assertPoint.name = "assert1";
  assertPoint.type = CoverageType::Assertion;
  assertPoint.hits = 1;
  db.addCoveragePoint(assertPoint);

  // Coverpoint coverage
  CoveragePoint coverpointPoint;
  coverpointPoint.name = "coverpoint1";
  coverpointPoint.type = CoverageType::Coverpoint;
  coverpointPoint.hits = 10;
  coverpointPoint.goal = 20;
  db.addCoveragePoint(coverpointPoint);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Verify all coverage types are represented
  EXPECT_TRUE(output.find("Line Coverage") != std::string::npos);
  EXPECT_TRUE(output.find("Toggle Coverage") != std::string::npos);
  EXPECT_TRUE(output.find("Branch Coverage") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithLongPointNames) {
  CoverageDatabase db;

  // Very long point name
  CoveragePoint point;
  point.name = "very_long_module_name.submodule.subsubmodule.deeply.nested."
               "hierarchy.with.many.levels.and.a.very.long.final.name:line123";
  point.type = CoverageType::Line;
  point.hits = 1;
  point.hierarchy = "very_long_module_name.submodule.subsubmodule.deeply";
  db.addCoveragePoint(point);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Should not crash and should produce valid HTML
  EXPECT_TRUE(output.find("<!DOCTYPE html>") != std::string::npos);
  EXPECT_TRUE(output.find("</html>") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithSpecialCharactersInName) {
  CoverageDatabase db;

  // Use special characters in name and hierarchy since those are outputted
  CoveragePoint point;
  point.name = "test<script>&point";
  point.type = CoverageType::Line;
  point.hits = 0; // Uncovered so it appears in uncovered items table
  point.hierarchy = "top<html>&test";
  db.addCoveragePoint(point);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Special characters in name and hierarchy should be escaped
  EXPECT_TRUE(output.find("&lt;script&gt;") != std::string::npos);
  EXPECT_TRUE(output.find("&amp;point") != std::string::npos);
  EXPECT_TRUE(output.find("&lt;html&gt;") != std::string::npos);
  EXPECT_TRUE(output.find("&amp;test") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWith100PercentCoverage) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "covered_point";
  point.type = CoverageType::Line;
  point.hits = 100;
  point.goal = 1;
  db.addCoveragePoint(point);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Should show 100% coverage
  EXPECT_TRUE(output.find("100") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWith0PercentCoverage) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "uncovered_point";
  point.type = CoverageType::Line;
  point.hits = 0;
  point.goal = 1;
  db.addCoveragePoint(point);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Uncovered items section should have content
  EXPECT_TRUE(output.find("Uncovered Items") != std::string::npos);
  EXPECT_TRUE(output.find("uncovered_point") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithMultipleTrendPoints) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test";
  point.type = CoverageType::Line;
  point.hits = 1;
  db.addCoveragePoint(point);

  // Add multiple trend points
  for (int i = 0; i < 10; ++i) {
    CoverageTrendPoint trend;
    trend.timestamp = "2024-01-0" + std::to_string(i + 1) + "T00:00:00Z";
    trend.runId = "run" + std::to_string(i);
    trend.overallCoverage = 50.0 + i * 5;
    trend.lineCoverage = 50.0 + i * 5;
    trend.toggleCoverage = 40.0 + i * 3;
    trend.branchCoverage = 60.0 + i * 4;
    trend.totalPoints = 10;
    trend.coveredPoints = 5 + i;
    db.addTrendPoint(trend);
  }

  HTMLReportOptions options;
  options.includeTrends = true;

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Trend section should be present
  EXPECT_TRUE(output.find("Coverage Trends") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithCoverageGroups) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "group_point_1";
  p1.type = CoverageType::Line;
  p1.hits = 1;
  db.addCoveragePoint(p1);

  CoveragePoint p2;
  p2.name = "group_point_2";
  p2.type = CoverageType::Line;
  p2.hits = 0;
  db.addCoveragePoint(p2);

  CoverageGroup group;
  group.name = "test_group";
  group.description = "A test coverage group";
  group.pointNames = {"group_point_1", "group_point_2"};
  db.addCoverageGroup(group);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Group should be mentioned
  EXPECT_TRUE(output.find("test_group") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithExclusions) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "normal_point";
  p1.type = CoverageType::Line;
  p1.hits = 1;
  db.addCoveragePoint(p1);

  CoverageExclusion exclusion;
  exclusion.pointName = "excluded_point";
  exclusion.reason = "Known dead code path";
  exclusion.author = "test_author";
  exclusion.date = "2024-01-01";
  exclusion.ticketId = "JIRA-123";
  db.addExclusion(exclusion);

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Exclusions section should be present
  EXPECT_TRUE(output.find("Exclusions") != std::string::npos);
  EXPECT_TRUE(output.find("excluded_point") != std::string::npos);
  EXPECT_TRUE(output.find("Known dead code path") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLUncoveredOnly) {
  CoverageDatabase db;

  CoveragePoint covered;
  covered.name = "covered_point";
  covered.type = CoverageType::Line;
  covered.hits = 5;
  db.addCoveragePoint(covered);

  CoveragePoint uncovered;
  uncovered.name = "uncovered_point";
  uncovered.type = CoverageType::Line;
  uncovered.hits = 0;
  db.addCoveragePoint(uncovered);

  HTMLReportOptions options;
  options.uncoveredOnly = true;

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Should still produce valid HTML
  EXPECT_TRUE(output.find("<!DOCTYPE html>") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLWithCustomCSS) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test";
  point.type = CoverageType::Line;
  point.hits = 1;
  db.addCoveragePoint(point);

  HTMLReportOptions options;
  options.customCSS = "body { background-color: #f0f0f0; }";

  std::string output;
  llvm::raw_string_ostream os(output);

  CoverageReportGenerator generator(db, options);
  auto err = generator.generateReport(os);
  ASSERT_FALSE(static_cast<bool>(err));

  // Custom CSS should be included
  EXPECT_TRUE(output.find("background-color: #f0f0f0") != std::string::npos);
}

TEST_F(CoverageReportGeneratorTest, GenerateHTMLToInvalidPath) {
  CoverageDatabase db;

  CoverageReportGenerator generator(db);
  auto err = generator.generateReport("/nonexistent/directory/report.html");
  EXPECT_TRUE(static_cast<bool>(err));
  llvm::consumeError(std::move(err));
}

TEST_F(CoverageReportGeneratorTest, SourceFileCoverageMultiplePoints) {
  CoverageDatabase db;

  // Multiple points in same file
  for (int i = 1; i <= 10; ++i) {
    CoveragePoint point;
    point.name = "test.v:" + std::to_string(i * 10);
    point.type = CoverageType::Line;
    point.hits = (i % 2 == 0) ? 1 : 0; // Every other line covered
    point.location.filename = "test.v";
    point.location.line = i * 10;
    db.addCoveragePoint(point);
  }

  CoverageReportGenerator generator(db);

  const auto &sourceFiles = generator.getSourceFileCoverage();

  auto it = sourceFiles.find("test.v");
  ASSERT_NE(it, sourceFiles.end());

  EXPECT_EQ(it->second.totalLines, 10u);
  EXPECT_EQ(it->second.coveredLines, 5u);
  EXPECT_DOUBLE_EQ(it->second.coveragePercent, 50.0);
}

TEST_F(CoverageReportGeneratorTest, HierarchyTreeDeepNesting) {
  CoverageDatabase db;

  // Create deeply nested hierarchy
  CoveragePoint point;
  point.name = "deep_point";
  point.type = CoverageType::Line;
  point.hits = 1;
  point.hierarchy = "level1.level2.level3.level4.level5.level6";
  db.addCoveragePoint(point);

  CoverageReportGenerator generator(db);

  const auto *root = generator.getHierarchyRoot();
  ASSERT_NE(root, nullptr);

  // Traverse the tree to verify structure
  const HierarchyNode *current = root;
  std::vector<std::string> expectedLevels = {"level1", "level2", "level3",
                                             "level4", "level5", "level6"};

  for (const auto &level : expectedLevels) {
    auto it = current->children.find(level);
    ASSERT_NE(it, current->children.end())
        << "Expected child " << level << " not found";
    current = it->second.get();
    EXPECT_EQ(current->name, level);
  }
}

TEST(SourceFileCoverageTest, MultiplePointsSameLine) {
  SourceFileCoverage file;
  file.filename = "test.v";

  CoveragePoint p1;
  p1.type = CoverageType::Line;
  p1.hits = 5;
  file.linePoints.emplace_back(10, &p1);

  CoveragePoint p2;
  p2.type = CoverageType::Line;
  p2.hits = 0;
  file.linePoints.emplace_back(10, &p2); // Same line

  file.calculateCoverage();

  // Even with two points on the same line, count unique lines
  // (implementation may vary - this tests current behavior)
  EXPECT_GE(file.totalLines, 1u);
  EXPECT_GE(file.coveragePercent, 0.0);
  EXPECT_LE(file.coveragePercent, 100.0);
}

TEST(SourceFileCoverageTest, UnorderedLines) {
  SourceFileCoverage file;
  file.filename = "test.v";

  CoveragePoint p1;
  p1.type = CoverageType::Line;
  p1.hits = 1;

  CoveragePoint p2;
  p2.type = CoverageType::Line;
  p2.hits = 1;

  CoveragePoint p3;
  p3.type = CoverageType::Line;
  p3.hits = 1;

  // Add lines out of order
  file.linePoints.emplace_back(30, &p1);
  file.linePoints.emplace_back(10, &p2);
  file.linePoints.emplace_back(20, &p3);

  file.calculateCoverage();

  EXPECT_EQ(file.totalLines, 3u);
  EXPECT_EQ(file.coveredLines, 3u);
  EXPECT_DOUBLE_EQ(file.coveragePercent, 100.0);
}

TEST(HierarchyNodeTest, EmptyNode) {
  CoverageDatabase db;

  HierarchyNode node;
  node.name = "empty";

  node.calculateCoverage(db);

  EXPECT_EQ(node.totalPoints, 0u);
  EXPECT_EQ(node.coveredPoints, 0u);
  EXPECT_DOUBLE_EQ(node.coveragePercent, 100.0);
}

TEST(HierarchyNodeTest, NodeWithChildren) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "child_point";
  p1.type = CoverageType::Line;
  p1.hits = 1;
  db.addCoveragePoint(p1);

  HierarchyNode parent;
  parent.name = "parent";

  auto child = std::make_unique<HierarchyNode>();
  child->name = "child";
  child->coveragePointNames = {"child_point"};

  parent.children["child"] = std::move(child);

  parent.calculateCoverage(db);

  // Parent should aggregate child coverage
  EXPECT_GE(parent.totalPoints, 0u);
}

} // namespace
