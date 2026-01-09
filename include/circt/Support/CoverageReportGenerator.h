//===- CoverageReportGenerator.h - Coverage report generation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CoverageReportGenerator class for generating
// comprehensive HTML coverage reports with drill-down, source annotations,
// and trend visualization.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_COVERAGEREPORTGENERATOR_H
#define CIRCT_SUPPORT_COVERAGEREPORTGENERATOR_H

#include "circt/Support/CoverageDatabase.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace circt {

/// Configuration options for HTML report generation.
struct HTMLReportOptions {
  /// Title for the report.
  std::string title = "CIRCT Coverage Report";

  /// Whether to include source file annotations.
  bool includeSourceAnnotations = true;

  /// Whether to include trend visualization.
  bool includeTrends = true;

  /// Whether to show only uncovered points in details.
  bool uncoveredOnly = false;

  /// Hierarchy filter prefix (empty for all).
  std::string hierarchyFilter;

  /// Base path for source files (for resolving relative paths).
  std::string sourceBasePath;

  /// Maximum number of trend points to display in the chart.
  size_t maxTrendPoints = 50;

  /// Whether to generate standalone HTML (includes CSS/JS inline).
  bool standalone = true;

  /// Custom CSS to inject.
  std::string customCSS;
};

/// Represents a node in the hierarchy tree.
struct HierarchyNode {
  std::string name;
  std::string fullPath;
  std::vector<std::string> coveragePointNames;
  std::map<std::string, std::unique_ptr<HierarchyNode>> children;

  size_t totalPoints = 0;
  size_t coveredPoints = 0;
  double coveragePercent = 0.0;

  /// Calculate coverage statistics recursively.
  void calculateCoverage(const CoverageDatabase &db);
};

/// Represents source file coverage information.
struct SourceFileCoverage {
  std::string filename;
  std::vector<std::pair<uint32_t, const CoveragePoint *>>
      linePoints; // line -> coverage point

  size_t totalLines = 0;
  size_t coveredLines = 0;
  double coveragePercent = 0.0;

  void calculateCoverage();
};

/// Generator for comprehensive HTML coverage reports.
class CoverageReportGenerator {
public:
  explicit CoverageReportGenerator(const CoverageDatabase &db,
                                   HTMLReportOptions options = {});

  /// Generate the complete HTML report.
  llvm::Error generateReport(llvm::raw_ostream &os);

  /// Generate report to a file.
  llvm::Error generateReport(llvm::StringRef path);

  /// Build the hierarchy tree from coverage points.
  void buildHierarchyTree();

  /// Build source file coverage map.
  void buildSourceFileCoverage();

  /// Get the root of the hierarchy tree.
  const HierarchyNode *getHierarchyRoot() const { return hierarchyRoot.get(); }

  /// Get source file coverage map.
  const llvm::StringMap<SourceFileCoverage> &getSourceFileCoverage() const {
    return sourceFileCoverage;
  }

private:
  const CoverageDatabase &db;
  HTMLReportOptions options;

  std::unique_ptr<HierarchyNode> hierarchyRoot;
  llvm::StringMap<SourceFileCoverage> sourceFileCoverage;

  /// Generate HTML header with CSS and meta tags.
  void generateHTMLHeader(llvm::raw_ostream &os);

  /// Generate navigation bar.
  void generateNavigation(llvm::raw_ostream &os);

  /// Generate summary dashboard cards.
  void generateSummaryDashboard(llvm::raw_ostream &os);

  /// Generate coverage by type section.
  void generateCoverageByType(llvm::raw_ostream &os);

  /// Generate hierarchical coverage breakdown with drill-down.
  void generateHierarchyBreakdown(llvm::raw_ostream &os);

  /// Generate hierarchy tree HTML recursively.
  void generateHierarchyNode(llvm::raw_ostream &os, const HierarchyNode *node,
                             int depth);

  /// Generate source file coverage section.
  void generateSourceFileCoverageSection(llvm::raw_ostream &os);

  /// Generate annotated source view for a file.
  void generateAnnotatedSource(llvm::raw_ostream &os,
                               const SourceFileCoverage &file);

  /// Generate trend visualization chart.
  void generateTrendVisualization(llvm::raw_ostream &os);

  /// Generate uncovered items list.
  void generateUncoveredItems(llvm::raw_ostream &os);

  /// Generate coverage groups section.
  void generateCoverageGroups(llvm::raw_ostream &os);

  /// Generate exclusions section.
  void generateExclusions(llvm::raw_ostream &os);

  /// Generate HTML footer with JavaScript.
  void generateHTMLFooter(llvm::raw_ostream &os);

  /// Get CSS color class for coverage percentage.
  static const char *getCoverageColorClass(double percent);

  /// Format percentage for display.
  static std::string formatPercent(double percent);

  /// HTML-escape a string.
  static std::string htmlEscape(llvm::StringRef str);
};

} // namespace circt

#endif // CIRCT_SUPPORT_COVERAGEREPORTGENERATOR_H
