//===- CoverageDatabaseTest.cpp - CoverageDatabase unit tests -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CoverageDatabase.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// CoveragePoint Tests
//===----------------------------------------------------------------------===//

TEST(CoveragePointTest, LineCoverage) {
  CoveragePoint point;
  point.name = "test::line1";
  point.type = CoverageType::Line;
  point.hits = 0;
  point.goal = 1;

  EXPECT_FALSE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 0.0);

  point.hits = 1;
  EXPECT_TRUE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 100.0);

  point.hits = 5;
  EXPECT_TRUE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 100.0); // Capped at 100%
}

TEST(CoveragePointTest, ToggleCoverage) {
  CoveragePoint point;
  point.name = "test::toggle1";
  point.type = CoverageType::Toggle;
  point.toggle01 = false;
  point.toggle10 = false;

  EXPECT_FALSE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 0.0);

  point.toggle01 = true;
  EXPECT_FALSE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 50.0);

  point.toggle10 = true;
  EXPECT_TRUE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 100.0);
}

TEST(CoveragePointTest, BranchCoverage) {
  CoveragePoint point;
  point.name = "test::branch1";
  point.type = CoverageType::Branch;
  point.branchTrue = false;
  point.branchFalse = false;

  EXPECT_FALSE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 0.0);

  point.branchTrue = true;
  EXPECT_FALSE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 50.0);

  point.branchFalse = true;
  EXPECT_TRUE(point.isCovered());
  EXPECT_DOUBLE_EQ(point.getCoveragePercent(), 100.0);
}

TEST(CoveragePointTest, Merge) {
  CoveragePoint p1;
  p1.name = "test::point1";
  p1.type = CoverageType::Line;
  p1.hits = 5;
  p1.toggle01 = true;
  p1.branchTrue = true;

  CoveragePoint p2;
  p2.name = "test::point1";
  p2.type = CoverageType::Line;
  p2.hits = 3;
  p2.toggle10 = true;
  p2.branchFalse = true;

  p1.merge(p2);

  EXPECT_EQ(p1.hits, 8u);
  EXPECT_TRUE(p1.toggle01);
  EXPECT_TRUE(p1.toggle10);
  EXPECT_TRUE(p1.branchTrue);
  EXPECT_TRUE(p1.branchFalse);
}

//===----------------------------------------------------------------------===//
// CoverageDatabase Tests
//===----------------------------------------------------------------------===//

TEST(CoverageDatabaseTest, AddAndGetCoveragePoints) {
  CoverageDatabase db;

  CoveragePoint point1;
  point1.name = "module.v:10";
  point1.type = CoverageType::Line;
  point1.hits = 1;

  CoveragePoint point2;
  point2.name = "module.v:20";
  point2.type = CoverageType::Line;
  point2.hits = 0;

  db.addCoveragePoint(point1);
  db.addCoveragePoint(point2);

  EXPECT_EQ(db.getTotalPointCount(), 2u);
  EXPECT_TRUE(db.hasCoveragePoint("module.v:10"));
  EXPECT_TRUE(db.hasCoveragePoint("module.v:20"));
  EXPECT_FALSE(db.hasCoveragePoint("module.v:30"));

  const auto *retrieved = db.getCoveragePoint("module.v:10");
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->hits, 1u);
}

TEST(CoverageDatabaseTest, RecordHit) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test::point";
  point.type = CoverageType::Line;
  point.hits = 0;

  db.addCoveragePoint(point);

  db.recordHit("test::point", 5);
  const auto *retrieved = db.getCoveragePoint("test::point");
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->hits, 5u);

  db.recordHit("test::point");
  retrieved = db.getCoveragePoint("test::point");
  EXPECT_EQ(retrieved->hits, 6u);
}

TEST(CoverageDatabaseTest, GetByType) {
  CoverageDatabase db;

  CoveragePoint linePoint;
  linePoint.name = "line1";
  linePoint.type = CoverageType::Line;
  db.addCoveragePoint(linePoint);

  CoveragePoint togglePoint;
  togglePoint.name = "toggle1";
  togglePoint.type = CoverageType::Toggle;
  db.addCoveragePoint(togglePoint);

  togglePoint.name = "toggle2";
  db.addCoveragePoint(togglePoint);

  auto linePoints = db.getCoveragePointsByType(CoverageType::Line);
  EXPECT_EQ(linePoints.size(), 1u);

  auto togglePoints = db.getCoveragePointsByType(CoverageType::Toggle);
  EXPECT_EQ(togglePoints.size(), 2u);

  auto branchPoints = db.getCoveragePointsByType(CoverageType::Branch);
  EXPECT_EQ(branchPoints.size(), 0u);
}

TEST(CoverageDatabaseTest, GetByHierarchy) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "p1";
  p1.hierarchy = "top.mod1.submod";
  db.addCoveragePoint(p1);

  CoveragePoint p2;
  p2.name = "p2";
  p2.hierarchy = "top.mod1.other";
  db.addCoveragePoint(p2);

  CoveragePoint p3;
  p3.name = "p3";
  p3.hierarchy = "top.mod2.submod";
  db.addCoveragePoint(p3);

  auto mod1Points = db.getCoveragePointsByHierarchy("top.mod1");
  EXPECT_EQ(mod1Points.size(), 2u);

  auto topPoints = db.getCoveragePointsByHierarchy("top");
  EXPECT_EQ(topPoints.size(), 3u);

  auto mod2Points = db.getCoveragePointsByHierarchy("top.mod2");
  EXPECT_EQ(mod2Points.size(), 1u);
}

TEST(CoverageDatabaseTest, CoverageGroups) {
  CoverageDatabase db;

  CoveragePoint p1;
  p1.name = "point1";
  p1.type = CoverageType::Line;
  p1.hits = 1;
  db.addCoveragePoint(p1);

  CoveragePoint p2;
  p2.name = "point2";
  p2.type = CoverageType::Line;
  p2.hits = 0;
  db.addCoveragePoint(p2);

  CoverageGroup group;
  group.name = "group1";
  group.pointNames = {"point1", "point2"};
  db.addCoverageGroup(group);

  const auto *retrieved = db.getCoverageGroup("group1");
  ASSERT_NE(retrieved, nullptr);
  EXPECT_DOUBLE_EQ(retrieved->getCoveragePercent(db.getCoveragePoints()), 50.0);
}

TEST(CoverageDatabaseTest, Exclusions) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test::point";
  point.type = CoverageType::Line;
  point.hits = 0;
  db.addCoveragePoint(point);

  EXPECT_FALSE(db.isExcluded("test::point"));

  CoverageExclusion exclusion;
  exclusion.pointName = "test::point";
  exclusion.reason = "Known unreachable code";
  exclusion.author = "test_user";
  db.addExclusion(exclusion);

  EXPECT_TRUE(db.isExcluded("test::point"));

  const auto *retrieved = db.getExclusion("test::point");
  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->reason, "Known unreachable code");
}

TEST(CoverageDatabaseTest, CoverageMetrics) {
  CoverageDatabase db;

  // Add some covered points
  CoveragePoint covered;
  covered.type = CoverageType::Line;
  covered.hits = 1;

  covered.name = "covered1";
  db.addCoveragePoint(covered);
  covered.name = "covered2";
  db.addCoveragePoint(covered);
  covered.name = "covered3";
  db.addCoveragePoint(covered);

  // Add an uncovered point
  CoveragePoint uncovered;
  uncovered.name = "uncovered1";
  uncovered.type = CoverageType::Line;
  uncovered.hits = 0;
  db.addCoveragePoint(uncovered);

  EXPECT_EQ(db.getTotalPointCount(), 4u);
  EXPECT_EQ(db.getCoveredPointCount(), 3u);
  EXPECT_DOUBLE_EQ(db.getOverallCoverage(), 75.0);
  EXPECT_DOUBLE_EQ(db.getCoverageByType(CoverageType::Line), 75.0);
}

TEST(CoverageDatabaseTest, Merge) {
  CoverageDatabase db1;
  CoverageDatabase db2;

  CoveragePoint p1;
  p1.name = "common::point";
  p1.type = CoverageType::Line;
  p1.hits = 5;
  db1.addCoveragePoint(p1);

  CoveragePoint p1b;
  p1b.name = "common::point";
  p1b.type = CoverageType::Line;
  p1b.hits = 3;
  db2.addCoveragePoint(p1b);

  CoveragePoint p2;
  p2.name = "db1::point";
  p2.type = CoverageType::Line;
  p2.hits = 1;
  db1.addCoveragePoint(p2);

  CoveragePoint p3;
  p3.name = "db2::point";
  p3.type = CoverageType::Line;
  p3.hits = 2;
  db2.addCoveragePoint(p3);

  db1.merge(db2);

  EXPECT_EQ(db1.getTotalPointCount(), 3u);

  const auto *common = db1.getCoveragePoint("common::point");
  ASSERT_NE(common, nullptr);
  EXPECT_EQ(common->hits, 8u); // 5 + 3

  EXPECT_TRUE(db1.hasCoveragePoint("db1::point"));
  EXPECT_TRUE(db1.hasCoveragePoint("db2::point"));
}

TEST(CoverageDatabaseTest, Diff) {
  CoverageDatabase newDb;
  CoverageDatabase oldDb;

  // Common point - covered in new, uncovered in old
  CoveragePoint newlyCoveredPoint;
  newlyCoveredPoint.name = "newly_covered";
  newlyCoveredPoint.type = CoverageType::Line;
  newlyCoveredPoint.hits = 1;
  newDb.addCoveragePoint(newlyCoveredPoint);

  newlyCoveredPoint.hits = 0;
  oldDb.addCoveragePoint(newlyCoveredPoint);

  // Point only in new
  CoveragePoint onlyNewPoint;
  onlyNewPoint.name = "only_new";
  onlyNewPoint.type = CoverageType::Line;
  onlyNewPoint.hits = 1;
  newDb.addCoveragePoint(onlyNewPoint);

  // Point only in old
  CoveragePoint onlyOldPoint;
  onlyOldPoint.name = "only_old";
  onlyOldPoint.type = CoverageType::Line;
  onlyOldPoint.hits = 1;
  oldDb.addCoveragePoint(onlyOldPoint);

  auto result = newDb.diff(oldDb);

  EXPECT_EQ(result.newlyCovered.size(), 1u);
  EXPECT_EQ(result.newlyCovered[0], "newly_covered");

  EXPECT_EQ(result.onlyInThis.size(), 1u);
  EXPECT_EQ(result.onlyInThis[0], "only_new");

  EXPECT_EQ(result.onlyInOther.size(), 1u);
  EXPECT_EQ(result.onlyInOther[0], "only_old");

  EXPECT_GT(result.coverageDelta, 0.0);
}

TEST(CoverageDatabaseTest, TrendTracking) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test";
  point.type = CoverageType::Line;
  point.hits = 1;
  db.addCoveragePoint(point);

  auto trend = db.createCurrentTrendPoint("run1", "abc123");

  EXPECT_EQ(trend.runId, "run1");
  EXPECT_EQ(trend.commitHash, "abc123");
  EXPECT_FALSE(trend.timestamp.empty());
  EXPECT_DOUBLE_EQ(trend.overallCoverage, 100.0);
  EXPECT_EQ(trend.totalPoints, 1u);
  EXPECT_EQ(trend.coveredPoints, 1u);

  db.addTrendPoint(trend);
  EXPECT_EQ(db.getTrends().size(), 1u);
}

//===----------------------------------------------------------------------===//
// Serialization Tests
//===----------------------------------------------------------------------===//

class CoverageDatabaseSerializationTest : public ::testing::Test {
protected:
  void SetUp() override {
    llvm::SmallString<128> tempPath;
    std::error_code ec =
        llvm::sys::fs::createUniqueDirectory("coverage-test", tempPath);
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

    CoveragePoint linePoint;
    linePoint.name = "module.v:10";
    linePoint.type = CoverageType::Line;
    linePoint.hits = 5;
    linePoint.goal = 1;
    linePoint.location.filename = "module.v";
    linePoint.location.line = 10;
    linePoint.hierarchy = "top.sub";
    linePoint.description = "Test line coverage";
    db.addCoveragePoint(linePoint);

    CoveragePoint togglePoint;
    togglePoint.name = "top.signal";
    togglePoint.type = CoverageType::Toggle;
    togglePoint.toggle01 = true;
    togglePoint.toggle10 = false;
    db.addCoveragePoint(togglePoint);

    CoverageGroup group;
    group.name = "test_group";
    group.description = "Test coverage group";
    group.pointNames = {"module.v:10", "top.signal"};
    db.addCoverageGroup(group);

    CoverageExclusion exclusion;
    exclusion.pointName = "excluded_point";
    exclusion.reason = "Dead code";
    exclusion.author = "test";
    exclusion.date = "2024-01-01";
    db.addExclusion(exclusion);

    db.setMetadata("version", "1.0");
    db.setMetadata("project", "test");

    return db;
  }
};

TEST_F(CoverageDatabaseSerializationTest, BinaryRoundTrip) {
  auto db = createTestDatabase();

  std::string filePath = tempDir + "/test.cov";

  auto writeErr = db.writeToFile(filePath);
  ASSERT_FALSE(static_cast<bool>(writeErr)) << llvm::toString(std::move(writeErr));

  auto readResult = CoverageDatabase::readFromFile(filePath);
  ASSERT_TRUE(static_cast<bool>(readResult)) << llvm::toString(readResult.takeError());

  CoverageDatabase &loaded = *readResult;

  EXPECT_EQ(loaded.getTotalPointCount(), db.getTotalPointCount());

  const auto *linePoint = loaded.getCoveragePoint("module.v:10");
  ASSERT_NE(linePoint, nullptr);
  EXPECT_EQ(linePoint->hits, 5u);
  EXPECT_EQ(linePoint->location.filename, "module.v");
  EXPECT_EQ(linePoint->location.line, 10u);

  const auto *togglePoint = loaded.getCoveragePoint("top.signal");
  ASSERT_NE(togglePoint, nullptr);
  EXPECT_TRUE(togglePoint->toggle01);
  EXPECT_FALSE(togglePoint->toggle10);

  const auto *group = loaded.getCoverageGroup("test_group");
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->pointNames.size(), 2u);

  EXPECT_TRUE(loaded.isExcluded("excluded_point"));

  auto version = loaded.getMetadata("version");
  ASSERT_TRUE(version.has_value());
  EXPECT_EQ(*version, "1.0");
}

TEST_F(CoverageDatabaseSerializationTest, JSONRoundTrip) {
  auto db = createTestDatabase();

  std::string filePath = tempDir + "/test.json";

  auto writeErr = db.writeToJSON(filePath);
  ASSERT_FALSE(static_cast<bool>(writeErr)) << llvm::toString(std::move(writeErr));

  auto readResult = CoverageDatabase::readFromJSON(filePath);
  ASSERT_TRUE(static_cast<bool>(readResult)) << llvm::toString(readResult.takeError());

  CoverageDatabase &loaded = *readResult;

  EXPECT_EQ(loaded.getTotalPointCount(), db.getTotalPointCount());

  const auto *linePoint = loaded.getCoveragePoint("module.v:10");
  ASSERT_NE(linePoint, nullptr);
  EXPECT_EQ(linePoint->hits, 5u);
}

TEST_F(CoverageDatabaseSerializationTest, JSONToObject) {
  auto db = createTestDatabase();

  auto json = db.toJSON();

  auto *root = json.getAsObject();
  ASSERT_NE(root, nullptr);

  auto *summary = root->getObject("summary");
  ASSERT_NE(summary, nullptr);

  auto totalPoints = summary->getInteger("total_points");
  ASSERT_TRUE(totalPoints.has_value());
  EXPECT_EQ(*totalPoints, 2);

  auto *points = root->getArray("coverage_points");
  ASSERT_NE(points, nullptr);
  EXPECT_EQ(points->size(), 2u);
}

//===----------------------------------------------------------------------===//
// Helper Function Tests
//===----------------------------------------------------------------------===//

TEST(CoverageTypeTest, NameParsing) {
  EXPECT_EQ(getCoverageTypeName(CoverageType::Line), "line");
  EXPECT_EQ(getCoverageTypeName(CoverageType::Toggle), "toggle");
  EXPECT_EQ(getCoverageTypeName(CoverageType::Branch), "branch");

  EXPECT_EQ(parseCoverageType("line"), CoverageType::Line);
  EXPECT_EQ(parseCoverageType("toggle"), CoverageType::Toggle);
  EXPECT_EQ(parseCoverageType("branch"), CoverageType::Branch);

  EXPECT_FALSE(parseCoverageType("invalid").has_value());
}

TEST(CoverageDatabaseTest, Metadata) {
  CoverageDatabase db;

  db.setMetadata("key1", "value1");
  db.setMetadata("key2", "value2");

  auto val1 = db.getMetadata("key1");
  ASSERT_TRUE(val1.has_value());
  EXPECT_EQ(*val1, "value1");

  auto val2 = db.getMetadata("key2");
  ASSERT_TRUE(val2.has_value());
  EXPECT_EQ(*val2, "value2");

  auto missing = db.getMetadata("missing");
  EXPECT_FALSE(missing.has_value());

  const auto &allMeta = db.getAllMetadata();
  EXPECT_EQ(allMeta.size(), 2u);
}

TEST(CoverageDatabaseTest, Clear) {
  CoverageDatabase db;

  CoveragePoint point;
  point.name = "test";
  db.addCoveragePoint(point);

  CoverageGroup group;
  group.name = "group";
  db.addCoverageGroup(group);

  CoverageExclusion exclusion;
  exclusion.pointName = "exc";
  db.addExclusion(exclusion);

  db.setMetadata("key", "value");

  EXPECT_GT(db.getTotalPointCount(), 0u);

  db.clear();

  EXPECT_EQ(db.getTotalPointCount(), 0u);
  EXPECT_EQ(db.getCoverageGroups().size(), 0u);
  EXPECT_EQ(db.getExclusions().size(), 0u);
  EXPECT_EQ(db.getAllMetadata().size(), 0u);
}

} // namespace
