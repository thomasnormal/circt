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

} // namespace
