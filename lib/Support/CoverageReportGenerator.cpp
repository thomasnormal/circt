//===- CoverageReportGenerator.cpp - Coverage report generation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CoverageReportGenerator class for generating
// comprehensive HTML coverage reports.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CoverageReportGenerator.h"
#include "circt/Support/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace circt;

//===----------------------------------------------------------------------===//
// HierarchyNode Implementation
//===----------------------------------------------------------------------===//

void HierarchyNode::calculateCoverage(const CoverageDatabase &db) {
  totalPoints = 0;
  coveredPoints = 0;

  // Count coverage points at this node
  for (const auto &pointName : coveragePointNames) {
    const auto *point = db.getCoveragePoint(pointName);
    if (point) {
      ++totalPoints;
      if (point->isCovered() || db.isExcluded(pointName))
        ++coveredPoints;
    }
  }

  // Recursively calculate for children
  for (auto &child : children) {
    child.second->calculateCoverage(db);
    totalPoints += child.second->totalPoints;
    coveredPoints += child.second->coveredPoints;
  }

  // Calculate percentage
  if (totalPoints > 0)
    coveragePercent = (static_cast<double>(coveredPoints) / totalPoints) * 100.0;
  else
    coveragePercent = 100.0;
}

//===----------------------------------------------------------------------===//
// SourceFileCoverage Implementation
//===----------------------------------------------------------------------===//

void SourceFileCoverage::calculateCoverage() {
  coveredLines = 0;
  totalLines = linePoints.size();

  for (const auto &lp : linePoints) {
    if (lp.second && lp.second->isCovered())
      ++coveredLines;
  }

  if (totalLines > 0)
    coveragePercent = (static_cast<double>(coveredLines) / totalLines) * 100.0;
  else
    coveragePercent = 100.0;
}

//===----------------------------------------------------------------------===//
// CoverageReportGenerator Implementation
//===----------------------------------------------------------------------===//

CoverageReportGenerator::CoverageReportGenerator(const CoverageDatabase &db,
                                                 HTMLReportOptions options)
    : db(db), options(std::move(options)) {
  buildHierarchyTree();
  buildSourceFileCoverage();
}

void CoverageReportGenerator::buildHierarchyTree() {
  hierarchyRoot = std::make_unique<HierarchyNode>();
  hierarchyRoot->name = "root";
  hierarchyRoot->fullPath = "";

  for (const auto &kv : db.getCoveragePoints()) {
    const auto &point = kv.second;

    // Skip if filtered
    if (!options.hierarchyFilter.empty() &&
        !llvm::StringRef(point.hierarchy).starts_with(options.hierarchyFilter))
      continue;

    // Parse hierarchy path
    llvm::SmallVector<llvm::StringRef, 8> parts;
    llvm::StringRef(point.hierarchy).split(parts, '.');

    // Navigate/create hierarchy nodes
    HierarchyNode *current = hierarchyRoot.get();
    std::string pathSoFar;

    for (size_t i = 0; i < parts.size(); ++i) {
      if (!pathSoFar.empty())
        pathSoFar += ".";
      pathSoFar += parts[i].str();

      auto it = current->children.find(parts[i].str());
      if (it == current->children.end()) {
        auto newNode = std::make_unique<HierarchyNode>();
        newNode->name = parts[i].str();
        newNode->fullPath = pathSoFar;
        auto *nodePtr = newNode.get();
        current->children[parts[i].str()] = std::move(newNode);
        current = nodePtr;
      } else {
        current = it->second.get();
      }
    }

    // Add coverage point to the leaf node
    current->coveragePointNames.push_back(point.name);
  }

  // Calculate coverage for all nodes
  hierarchyRoot->calculateCoverage(db);
}

void CoverageReportGenerator::buildSourceFileCoverage() {
  for (const auto &kv : db.getCoveragePoints()) {
    const auto &point = kv.second;

    // Skip if no source location
    if (point.location.filename.empty())
      continue;

    // Skip if filtered
    if (!options.hierarchyFilter.empty() &&
        !llvm::StringRef(point.hierarchy).starts_with(options.hierarchyFilter))
      continue;

    auto &fileCov = sourceFileCoverage[point.location.filename];
    if (fileCov.filename.empty())
      fileCov.filename = point.location.filename;

    fileCov.linePoints.emplace_back(point.location.line, &point);
  }

  // Sort line points and calculate coverage for each file
  for (auto &kv : sourceFileCoverage) {
    auto &fileCov = kv.second;
    std::sort(fileCov.linePoints.begin(), fileCov.linePoints.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    fileCov.calculateCoverage();
  }
}

llvm::Error CoverageReportGenerator::generateReport(llvm::raw_ostream &os) {
  generateHTMLHeader(os);
  generateNavigation(os);
  generateSummaryDashboard(os);
  generateCoverageByType(os);
  generateHierarchyBreakdown(os);
  generateSourceFileCoverageSection(os);

  if (options.includeTrends && !db.getTrends().empty())
    generateTrendVisualization(os);

  generateUncoveredItems(os);
  generateCoverageGroups(os);
  generateExclusions(os);
  generateHTMLFooter(os);

  return llvm::Error::success();
}

llvm::Error CoverageReportGenerator::generateReport(llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec);
  if (ec)
    return llvm::createStringError(ec, "Failed to open file for writing: " +
                                            path.str());
  return generateReport(os);
}

const char *CoverageReportGenerator::getCoverageColorClass(double percent) {
  if (percent >= 90.0)
    return "cov-high";
  if (percent >= 70.0)
    return "cov-medium";
  return "cov-low";
}

std::string CoverageReportGenerator::formatPercent(double percent) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << percent << "%";
  return oss.str();
}

std::string CoverageReportGenerator::htmlEscape(llvm::StringRef str) {
  std::string result;
  result.reserve(str.size());
  for (char c : str) {
    switch (c) {
    case '&':
      result += "&amp;";
      break;
    case '<':
      result += "&lt;";
      break;
    case '>':
      result += "&gt;";
      break;
    case '"':
      result += "&quot;";
      break;
    case '\'':
      result += "&#39;";
      break;
    default:
      result += c;
    }
  }
  return result;
}

void CoverageReportGenerator::generateHTMLHeader(llvm::raw_ostream &os) {
  os << R"(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>)" << htmlEscape(options.title)
     << R"(</title>
  <style>
    :root {
      --bg-primary: #f8f9fa;
      --bg-secondary: #ffffff;
      --text-primary: #212529;
      --text-secondary: #6c757d;
      --border-color: #dee2e6;
      --cov-high: #28a745;
      --cov-high-bg: #d4edda;
      --cov-medium: #ffc107;
      --cov-medium-bg: #fff3cd;
      --cov-low: #dc3545;
      --cov-low-bg: #f8d7da;
      --accent: #007bff;
      --shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.6;
    }

    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

    /* Navigation */
    .nav {
      background: #343a40;
      color: white;
      padding: 15px 20px;
      position: sticky;
      top: 0;
      z-index: 1000;
      box-shadow: var(--shadow);
    }
    .nav-content {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .nav-title { font-size: 1.5rem; font-weight: 600; }
    .nav-links a {
      color: rgba(255,255,255,0.8);
      text-decoration: none;
      margin-left: 20px;
      transition: color 0.2s;
    }
    .nav-links a:hover { color: white; }

    /* Cards */
    .card {
      background: var(--bg-secondary);
      border-radius: 8px;
      box-shadow: var(--shadow);
      margin-bottom: 20px;
      overflow: hidden;
    }
    .card-header {
      background: #f1f3f5;
      padding: 15px 20px;
      border-bottom: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card-header h2 {
      font-size: 1.25rem;
      font-weight: 600;
    }
    .card-body { padding: 20px; }

    /* Dashboard Grid */
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-bottom: 20px;
    }
    .metric-card {
      background: var(--bg-secondary);
      border-radius: 8px;
      padding: 25px;
      text-align: center;
      box-shadow: var(--shadow);
      transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value {
      font-size: 2.5rem;
      font-weight: 700;
      line-height: 1.2;
    }
    .metric-label {
      color: var(--text-secondary);
      font-size: 0.9rem;
      margin-top: 5px;
    }
    .metric-bar {
      height: 6px;
      background: #e9ecef;
      border-radius: 3px;
      margin-top: 15px;
      overflow: hidden;
    }
    .metric-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

    /* Coverage Colors */
    .cov-high { color: var(--cov-high); }
    .cov-medium { color: var(--cov-medium); }
    .cov-low { color: var(--cov-low); }
    .bg-cov-high { background: var(--cov-high); }
    .bg-cov-medium { background: var(--cov-medium); }
    .bg-cov-low { background: var(--cov-low); }

    /* Progress Bar */
    .progress-bar {
      height: 20px;
      background: #e9ecef;
      border-radius: 10px;
      overflow: hidden;
      flex: 1;
      margin: 0 15px;
    }
    .progress-fill {
      height: 100%;
      border-radius: 10px;
      transition: width 0.3s;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 0.75rem;
      font-weight: 600;
    }

    /* Coverage Row */
    .coverage-row {
      display: flex;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid var(--border-color);
    }
    .coverage-row:last-child { border-bottom: none; }
    .coverage-name { min-width: 150px; font-weight: 500; }
    .coverage-stats {
      min-width: 100px;
      text-align: right;
      color: var(--text-secondary);
      font-size: 0.875rem;
    }
    .coverage-percent {
      min-width: 70px;
      text-align: right;
      font-weight: 600;
    }

    /* Hierarchy Tree */
    .hierarchy-tree { padding-left: 0; list-style: none; }
    .hierarchy-tree ul { list-style: none; padding-left: 25px; }
    .hierarchy-node {
      padding: 8px 0;
      border-bottom: 1px solid var(--border-color);
    }
    .hierarchy-node:last-child { border-bottom: none; }
    .hierarchy-toggle {
      cursor: pointer;
      user-select: none;
      display: flex;
      align-items: center;
    }
    .hierarchy-toggle:before {
      content: '\25B6';
      display: inline-block;
      width: 20px;
      font-size: 0.7em;
      transition: transform 0.2s;
    }
    .hierarchy-toggle.expanded:before { transform: rotate(90deg); }
    .hierarchy-content { display: none; }
    .hierarchy-content.show { display: block; }
    .hierarchy-leaf:before { content: ''; width: 20px; }
    .hierarchy-name { flex: 1; font-weight: 500; }
    .hierarchy-info {
      display: flex;
      align-items: center;
      margin-left: 15px;
    }
    .hierarchy-mini-bar {
      width: 100px;
      height: 8px;
      background: #e9ecef;
      border-radius: 4px;
      overflow: hidden;
      margin-right: 10px;
    }
    .hierarchy-mini-fill { height: 100%; border-radius: 4px; }
    .hierarchy-percent { min-width: 50px; text-align: right; font-weight: 600; }

    /* Source Files */
    .source-file-list { list-style: none; }
    .source-file-item {
      display: flex;
      align-items: center;
      padding: 12px;
      border-bottom: 1px solid var(--border-color);
      cursor: pointer;
      transition: background 0.2s;
    }
    .source-file-item:hover { background: #f8f9fa; }
    .source-file-name { flex: 1; font-family: monospace; }
    .source-file-stats { color: var(--text-secondary); margin-right: 15px; }

    /* Annotated Source */
    .source-view {
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 0.85rem;
      overflow-x: auto;
    }
    .source-line {
      display: flex;
      white-space: pre;
      min-height: 1.5em;
    }
    .source-line:hover { background: rgba(0,0,0,0.03); }
    .line-number {
      min-width: 50px;
      padding: 0 10px;
      text-align: right;
      color: var(--text-secondary);
      background: #f8f9fa;
      border-right: 1px solid var(--border-color);
      user-select: none;
    }
    .line-hits {
      min-width: 50px;
      padding: 0 10px;
      text-align: right;
      font-weight: 500;
    }
    .line-code { padding-left: 15px; flex: 1; }
    .line-covered { background: var(--cov-high-bg); }
    .line-uncovered { background: var(--cov-low-bg); }
    .line-covered .line-hits { color: var(--cov-high); }
    .line-uncovered .line-hits { color: var(--cov-low); }

    /* Trend Chart */
    .trend-chart {
      height: 200px;
      display: flex;
      align-items: flex-end;
      gap: 2px;
      padding: 20px 0;
      border-bottom: 2px solid var(--border-color);
    }
    .trend-bar {
      flex: 1;
      min-width: 10px;
      max-width: 30px;
      border-radius: 4px 4px 0 0;
      transition: height 0.3s;
      cursor: pointer;
      position: relative;
    }
    .trend-bar:hover { opacity: 0.8; }
    .trend-tooltip {
      display: none;
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      background: #333;
      color: white;
      padding: 5px 10px;
      border-radius: 4px;
      font-size: 0.75rem;
      white-space: nowrap;
      z-index: 100;
    }
    .trend-bar:hover .trend-tooltip { display: block; }

    /* Tables */
    .data-table {
      width: 100%;
      border-collapse: collapse;
    }
    .data-table th, .data-table td {
      text-align: left;
      padding: 12px;
      border-bottom: 1px solid var(--border-color);
    }
    .data-table th {
      background: #f8f9fa;
      font-weight: 600;
      position: sticky;
      top: 60px;
    }
    .data-table tr:hover { background: #f8f9fa; }
    .data-table .status-covered { color: var(--cov-high); }
    .data-table .status-uncovered { color: var(--cov-low); }
    .data-table .status-excluded { color: var(--text-secondary); }

    /* Badges */
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 12px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge-covered { background: var(--cov-high-bg); color: var(--cov-high); }
    .badge-uncovered { background: var(--cov-low-bg); color: var(--cov-low); }
    .badge-excluded { background: #e9ecef; color: var(--text-secondary); }

    /* Footer */
    .footer {
      text-align: center;
      padding: 20px;
      color: var(--text-secondary);
      font-size: 0.875rem;
      border-top: 1px solid var(--border-color);
      margin-top: 20px;
    }

    /* Collapsible */
    .collapsible { cursor: pointer; }
    .collapsible-content { display: none; }
    .collapsible-content.show { display: block; }

    /* Search/Filter */
    .search-box {
      padding: 10px 15px;
      border: 1px solid var(--border-color);
      border-radius: 6px;
      width: 100%;
      font-size: 0.9rem;
    }
    .search-box:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(0,123,255,0.1);
    }

    /* Tab Navigation */
    .tabs { display: flex; border-bottom: 2px solid var(--border-color); }
    .tab {
      padding: 12px 20px;
      cursor: pointer;
      border-bottom: 2px solid transparent;
      margin-bottom: -2px;
      transition: all 0.2s;
    }
    .tab:hover { background: #f8f9fa; }
    .tab.active {
      border-bottom-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }
    .tab-content { display: none; padding: 20px 0; }
    .tab-content.active { display: block; }

    /* Responsive */
    @media (max-width: 768px) {
      .dashboard-grid { grid-template-columns: 1fr 1fr; }
      .coverage-row { flex-wrap: wrap; }
      .progress-bar { order: 3; width: 100%; margin: 10px 0 0 0; }
    }
  </style>
)";

  if (!options.customCSS.empty()) {
    os << "  <style>" << options.customCSS << "</style>\n";
  }

  os << R"(</head>
<body>
)";
}

void CoverageReportGenerator::generateNavigation(llvm::raw_ostream &os) {
  os << R"(  <nav class="nav">
    <div class="nav-content">
      <div class="nav-title">)" << htmlEscape(options.title) << R"(</div>
      <div class="nav-links">
        <a href="#summary">Summary</a>
        <a href="#hierarchy">Hierarchy</a>
        <a href="#sources">Sources</a>
        <a href="#uncovered">Uncovered</a>
)";

  if (options.includeTrends && !db.getTrends().empty())
    os << R"(        <a href="#trends">Trends</a>
)";

  os << R"(      </div>
    </div>
  </nav>
)";
}

void CoverageReportGenerator::generateSummaryDashboard(llvm::raw_ostream &os) {
  double overall = db.getOverallCoverage();
  size_t totalPoints = db.getTotalPointCount();
  size_t coveredPoints = db.getCoveredPointCount();

  os << R"(
  <div class="container">
    <div id="summary" class="dashboard-grid">
      <div class="metric-card">
        <div class="metric-value )" << getCoverageColorClass(overall) << R"(">)"
     << formatPercent(overall) << R"(</div>
        <div class="metric-label">Overall Coverage</div>
        <div class="metric-bar">
          <div class="metric-bar-fill bg-)" << getCoverageColorClass(overall)
     << R"(" style="width: )" << overall << R"(%"></div>
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">)" << totalPoints << R"(</div>
        <div class="metric-label">Total Points</div>
      </div>
      <div class="metric-card">
        <div class="metric-value cov-high">)" << coveredPoints << R"(</div>
        <div class="metric-label">Covered Points</div>
      </div>
      <div class="metric-card">
        <div class="metric-value cov-low">)" << (totalPoints - coveredPoints) << R"(</div>
        <div class="metric-label">Uncovered Points</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">)" << db.getExclusions().size() << R"(</div>
        <div class="metric-label">Exclusions</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">)" << sourceFileCoverage.size() << R"(</div>
        <div class="metric-label">Source Files</div>
      </div>
    </div>
)";
}

void CoverageReportGenerator::generateCoverageByType(llvm::raw_ostream &os) {
  os << R"(
    <div class="card">
      <div class="card-header">
        <h2>Coverage by Type</h2>
      </div>
      <div class="card-body">
)";

  auto generateTypeRow = [&](CoverageType type, const char *name) {
    size_t total = db.getTotalPointCountByType(type);
    if (total == 0)
      return;

    double percent = db.getCoverageByType(type);
    size_t covered = db.getCoveredPointCountByType(type);

    os << R"(        <div class="coverage-row">
          <div class="coverage-name">)" << name << R"(</div>
          <div class="progress-bar">
            <div class="progress-fill bg-)" << getCoverageColorClass(percent)
       << R"(" style="width: )" << percent << R"(%">)"
       << formatPercent(percent) << R"(</div>
          </div>
          <div class="coverage-stats">)" << covered << "/" << total << R"(</div>
          <div class="coverage-percent )" << getCoverageColorClass(percent) << R"(">)"
       << formatPercent(percent) << R"(</div>
        </div>
)";
  };

  generateTypeRow(CoverageType::Line, "Line Coverage");
  generateTypeRow(CoverageType::Toggle, "Toggle Coverage");
  generateTypeRow(CoverageType::Branch, "Branch Coverage");
  generateTypeRow(CoverageType::Condition, "Condition Coverage");
  generateTypeRow(CoverageType::FSM, "FSM Coverage");
  generateTypeRow(CoverageType::Assertion, "Assertion Coverage");
  generateTypeRow(CoverageType::Coverpoint, "Coverpoint Coverage");

  os << R"(      </div>
    </div>
)";
}

void CoverageReportGenerator::generateHierarchyBreakdown(llvm::raw_ostream &os) {
  os << R"HTML(
    <div id="hierarchy" class="card">
      <div class="card-header">
        <h2>Hierarchy Breakdown</h2>
        <input type="text" class="search-box" placeholder="Filter hierarchy..."
               onkeyup="filterHierarchy(this.value)" style="max-width: 300px;">
      </div>
      <div class="card-body">
        <ul class="hierarchy-tree" id="hierarchy-tree">
)HTML";

  // Generate tree starting from root's children
  for (const auto &child : hierarchyRoot->children) {
    generateHierarchyNode(os, child.second.get(), 0);
  }

  os << R"(        </ul>
      </div>
    </div>
)";
}

void CoverageReportGenerator::generateHierarchyNode(llvm::raw_ostream &os,
                                                     const HierarchyNode *node,
                                                     int depth) {
  bool hasChildren = !node->children.empty();
  const char *colorClass = getCoverageColorClass(node->coveragePercent);

  os << R"(          <li class="hierarchy-node" data-path=")"
     << htmlEscape(node->fullPath) << R"(">
            <div class=")" << (hasChildren ? "hierarchy-toggle" : "hierarchy-leaf")
     << R"(">
              <span class="hierarchy-name">)" << htmlEscape(node->name) << R"(</span>
              <div class="hierarchy-info">
                <div class="hierarchy-mini-bar">
                  <div class="hierarchy-mini-fill bg-)" << colorClass
     << R"(" style="width: )" << node->coveragePercent << R"(%"></div>
                </div>
                <span class="hierarchy-percent )" << colorClass << R"(">)"
     << formatPercent(node->coveragePercent) << R"(</span>
                <span class="coverage-stats" style="margin-left: 10px;">)"
     << node->coveredPoints << "/" << node->totalPoints << R"(</span>
              </div>
            </div>
)";

  if (hasChildren) {
    os << R"(            <div class="hierarchy-content">
              <ul>
)";
    for (const auto &child : node->children) {
      generateHierarchyNode(os, child.second.get(), depth + 1);
    }
    os << R"(              </ul>
            </div>
)";
  }

  os << R"(          </li>
)";
}

void CoverageReportGenerator::generateSourceFileCoverageSection(
    llvm::raw_ostream &os) {
  if (sourceFileCoverage.empty())
    return;

  os << R"(
    <div id="sources" class="card">
      <div class="card-header">
        <h2>Source File Coverage</h2>
      </div>
      <div class="card-body">
        <ul class="source-file-list">
)";

  // Sort files by coverage (lowest first to highlight problem areas)
  std::vector<const SourceFileCoverage *> sortedFiles;
  for (const auto &kv : sourceFileCoverage)
    sortedFiles.push_back(&kv.second);

  std::sort(sortedFiles.begin(), sortedFiles.end(),
            [](const auto *a, const auto *b) {
              return a->coveragePercent < b->coveragePercent;
            });

  for (const auto *file : sortedFiles) {
    const char *colorClass = getCoverageColorClass(file->coveragePercent);

    os << "          <li class=\"source-file-item\" onclick=\"toggleSource('"
       << htmlEscape(file->filename) << "')\">\n"
       << "            <span class=\"source-file-name\">" << htmlEscape(file->filename) << "</span>\n"
       << "            <span class=\"source-file-stats\">" << file->coveredLines << "/"
       << file->totalLines << " lines</span>\n"
       << "            <div class=\"hierarchy-mini-bar\" style=\"width: 150px;\">\n"
       << "              <div class=\"hierarchy-mini-fill bg-" << colorClass
       << "\" style=\"width: " << file->coveragePercent << "%\"></div>\n"
       << "            </div>\n"
       << "            <span class=\"hierarchy-percent " << colorClass << "\" style=\"margin-left: 10px;\">"
       << formatPercent(file->coveragePercent) << "</span>\n"
       << "          </li>\n"
       << "          <div id=\"source-" << htmlEscape(file->filename)
       << "\" class=\"source-view collapsible-content\">\n";

    generateAnnotatedSource(os, *file);

    os << "          </div>\n";
  }

  os << R"(        </ul>
      </div>
    </div>
)";
}

void CoverageReportGenerator::generateAnnotatedSource(
    llvm::raw_ostream &os, const SourceFileCoverage &file) {

  // Try to read the source file if source annotations are enabled
  std::vector<std::string> sourceLines;
  bool hasSource = false;

  if (options.includeSourceAnnotations) {
    std::string fullPath = file.filename;
    if (!options.sourceBasePath.empty()) {
      llvm::SmallString<256> path(options.sourceBasePath);
      llvm::sys::path::append(path, file.filename);
      fullPath = std::string(path);
    }

    auto bufferOrErr = llvm::MemoryBuffer::getFile(fullPath);
    if (bufferOrErr) {
      llvm::StringRef content = bufferOrErr.get()->getBuffer();
      llvm::SmallVector<llvm::StringRef, 64> lines;
      content.split(lines, '\n');
      for (const auto &line : lines)
        sourceLines.push_back(line.str());
      hasSource = true;
    }
  }

  // Build a map from line number to coverage point
  std::map<uint32_t, const CoveragePoint *> lineMap;
  for (const auto &lp : file.linePoints)
    lineMap[lp.first] = lp.second;

  if (hasSource) {
    for (size_t i = 0; i < sourceLines.size(); ++i) {
      uint32_t lineNum = i + 1;
      auto it = lineMap.find(lineNum);
      const CoveragePoint *point = (it != lineMap.end()) ? it->second : nullptr;

      std::string lineClass = "";
      std::string hitsStr = "";

      if (point) {
        if (point->isCovered()) {
          lineClass = " line-covered";
          hitsStr = std::to_string(point->hits);
        } else {
          lineClass = " line-uncovered";
          hitsStr = "0";
        }
      }

      os << R"(            <div class="source-line)" << lineClass << R"(">
              <span class="line-number">)" << lineNum << R"(</span>
              <span class="line-hits">)" << hitsStr << R"(</span>
              <span class="line-code">)" << htmlEscape(sourceLines[i]) << R"(</span>
            </div>
)";
    }
  } else {
    // No source available, just show coverage points
    for (const auto &lp : file.linePoints) {
      const char *lineClass =
          lp.second->isCovered() ? "line-covered" : "line-uncovered";
      std::string hitsStr =
          lp.second->isCovered() ? std::to_string(lp.second->hits) : "0";

      os << R"(            <div class="source-line )" << lineClass << R"(">
              <span class="line-number">)" << lp.first << R"(</span>
              <span class="line-hits">)" << hitsStr << R"(</span>
              <span class="line-code">[Coverage point at line )" << lp.first << R"(]</span>
            </div>
)";
    }
  }
}

void CoverageReportGenerator::generateTrendVisualization(llvm::raw_ostream &os) {
  const auto &trends = db.getTrends();
  if (trends.empty())
    return;

  os << R"(
    <div id="trends" class="card">
      <div class="card-header">
        <h2>Coverage Trends</h2>
      </div>
      <div class="card-body">
        <div class="trend-chart">
)";

  // Limit trend points to display
  size_t startIdx = 0;
  if (trends.size() > options.maxTrendPoints)
    startIdx = trends.size() - options.maxTrendPoints;

  for (size_t i = startIdx; i < trends.size(); ++i) {
    const auto &trend = trends[i];
    const char *colorClass = getCoverageColorClass(trend.overallCoverage);

    os << R"(          <div class="trend-bar bg-)" << colorClass
       << R"(" style="height: )" << trend.overallCoverage << R"(%">
            <div class="trend-tooltip">
              <strong>)" << htmlEscape(trend.runId) << R"(</strong><br>
              )" << htmlEscape(trend.timestamp) << R"(<br>
              Coverage: )" << formatPercent(trend.overallCoverage) << R"(
            </div>
          </div>
)";
  }

  os << R"(        </div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Run ID</th>
              <th>Commit</th>
              <th>Line</th>
              <th>Toggle</th>
              <th>Branch</th>
              <th>Overall</th>
            </tr>
          </thead>
          <tbody>
)";

  // Show trends in reverse order (newest first)
  for (auto it = trends.rbegin(); it != trends.rend(); ++it) {
    const auto &trend = *it;
    os << R"(            <tr>
              <td>)" << htmlEscape(trend.timestamp) << R"(</td>
              <td>)" << htmlEscape(trend.runId) << R"(</td>
              <td>)" << (trend.commitHash.empty() ? "-" : htmlEscape(trend.commitHash.substr(0, 8))) << R"(</td>
              <td>)" << formatPercent(trend.lineCoverage) << R"(</td>
              <td>)" << formatPercent(trend.toggleCoverage) << R"(</td>
              <td>)" << formatPercent(trend.branchCoverage) << R"(</td>
              <td class=")" << getCoverageColorClass(trend.overallCoverage) << R"(">)"
       << formatPercent(trend.overallCoverage) << R"(</td>
            </tr>
)";
  }

  os << R"(          </tbody>
        </table>
      </div>
    </div>
)";
}

void CoverageReportGenerator::generateUncoveredItems(llvm::raw_ostream &os) {
  std::vector<const CoveragePoint *> uncoveredPoints;

  for (const auto &kv : db.getCoveragePoints()) {
    const auto &point = kv.second;
    if (!point.isCovered() && !db.isExcluded(point.name)) {
      if (options.hierarchyFilter.empty() ||
          llvm::StringRef(point.hierarchy).starts_with(options.hierarchyFilter)) {
        uncoveredPoints.push_back(&point);
      }
    }
  }

  os << R"(
    <div id="uncovered" class="card">
      <div class="card-header">
        <h2>Uncovered Items</h2>
        <span class="badge badge-uncovered">)" << uncoveredPoints.size() << R"( items</span>
      </div>
      <div class="card-body">
)";

  if (uncoveredPoints.empty()) {
    os << R"(        <p style="text-align: center; color: var(--cov-high);">
          All coverage points are covered or excluded!
        </p>
)";
  } else {
    os << R"(        <table class="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Type</th>
              <th>Location</th>
              <th>Hierarchy</th>
            </tr>
          </thead>
          <tbody>
)";

    for (const auto *point : uncoveredPoints) {
      os << "            <tr>\n"
         << "              <td>" << htmlEscape(point->name) << "</td>\n"
         << "              <td>" << getCoverageTypeName(point->type) << "</td>\n"
         << "              <td>";
      if (!point->location.filename.empty()) {
        os << "<a href=\"#source-" << htmlEscape(point->location.filename)
           << "\" onclick=\"showSource('" << htmlEscape(point->location.filename)
           << "')\">" << htmlEscape(point->location.filename) << ":"
           << point->location.line << "</a>";
      } else {
        os << "-";
      }
      os << "</td>\n"
         << "              <td>" << htmlEscape(point->hierarchy) << "</td>\n"
         << "            </tr>\n";
    }

    os << R"(          </tbody>
        </table>
)";
  }

  os << R"(      </div>
    </div>
)";
}

void CoverageReportGenerator::generateCoverageGroups(llvm::raw_ostream &os) {
  const auto &groups = db.getCoverageGroups();
  if (groups.empty())
    return;

  os << R"(
    <div class="card">
      <div class="card-header">
        <h2>Coverage Groups</h2>
      </div>
      <div class="card-body">
)";

  for (const auto &kv : groups) {
    const auto &group = kv.second;
    double percent = group.getCoveragePercent(db.getCoveragePoints());
    const char *colorClass = getCoverageColorClass(percent);

    os << R"(        <div class="coverage-row">
          <div class="coverage-name">)" << htmlEscape(group.name);
    if (!group.description.empty())
      os << R"( <small style="color: var(--text-secondary);">)"
         << htmlEscape(group.description) << "</small>";
    os << R"(</div>
          <div class="progress-bar">
            <div class="progress-fill bg-)" << colorClass
       << R"(" style="width: )" << percent << R"(%">)"
       << formatPercent(percent) << R"(</div>
          </div>
          <div class="coverage-stats">)" << group.pointNames.size() << R"( points</div>
          <div class="coverage-percent )" << colorClass << R"(">)"
       << formatPercent(percent) << R"(</div>
        </div>
)";
  }

  os << R"(      </div>
    </div>
)";
}

void CoverageReportGenerator::generateExclusions(llvm::raw_ostream &os) {
  const auto &exclusions = db.getExclusions();
  if (exclusions.empty())
    return;

  os << R"(
    <div class="card">
      <div class="card-header">
        <h2>Exclusions</h2>
        <span class="badge badge-excluded">)" << exclusions.size() << R"( items</span>
      </div>
      <div class="card-body">
        <table class="data-table">
          <thead>
            <tr>
              <th>Coverage Point</th>
              <th>Reason</th>
              <th>Author</th>
              <th>Date</th>
              <th>Ticket</th>
            </tr>
          </thead>
          <tbody>
)";

  for (const auto &exclusion : exclusions) {
    os << R"(            <tr>
              <td>)" << htmlEscape(exclusion.pointName) << R"(</td>
              <td>)" << htmlEscape(exclusion.reason) << R"(</td>
              <td>)" << htmlEscape(exclusion.author.empty() ? "-" : exclusion.author)
       << R"(</td>
              <td>)" << htmlEscape(exclusion.date.empty() ? "-" : exclusion.date)
       << R"(</td>
              <td>)" << htmlEscape(exclusion.ticketId.empty() ? "-" : exclusion.ticketId)
       << R"(</td>
            </tr>
)";
  }

  os << R"(          </tbody>
        </table>
      </div>
    </div>
)";
}

void CoverageReportGenerator::generateHTMLFooter(llvm::raw_ostream &os) {
  os << R"(
    <footer class="footer">
      Generated by <strong>circt-cov</strong> ()" << getCirctVersion() << R"()
    </footer>
  </div>

  <script>
    // Hierarchy tree toggle
    document.querySelectorAll('.hierarchy-toggle').forEach(function(toggle) {
      toggle.addEventListener('click', function() {
        this.classList.toggle('expanded');
        var content = this.nextElementSibling;
        if (content && content.classList.contains('hierarchy-content')) {
          content.classList.toggle('show');
        }
      });
    });

    // Filter hierarchy
    function filterHierarchy(query) {
      query = query.toLowerCase();
      document.querySelectorAll('.hierarchy-node').forEach(function(node) {
        var path = node.getAttribute('data-path').toLowerCase();
        var name = node.querySelector('.hierarchy-name').textContent.toLowerCase();
        if (query === '' || path.includes(query) || name.includes(query)) {
          node.style.display = '';
          // Expand parents
          var parent = node.parentElement;
          while (parent && parent.classList.contains('hierarchy-content')) {
            parent.classList.add('show');
            var toggle = parent.previousElementSibling;
            if (toggle) toggle.classList.add('expanded');
            parent = parent.parentElement.parentElement;
          }
        } else {
          node.style.display = 'none';
        }
      });
    }

    // Source file toggle
    function toggleSource(filename) {
      var el = document.getElementById('source-' + filename);
      if (el) {
        el.classList.toggle('show');
      }
    }

    function showSource(filename) {
      var el = document.getElementById('source-' + filename);
      if (el) {
        el.classList.add('show');
        el.scrollIntoView({behavior: 'smooth', block: 'start'});
      }
    }

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-links a').forEach(function(link) {
      link.addEventListener('click', function(e) {
        var href = this.getAttribute('href');
        if (href.startsWith('#')) {
          e.preventDefault();
          var target = document.querySelector(href);
          if (target) {
            target.scrollIntoView({behavior: 'smooth', block: 'start'});
          }
        }
      });
    });

    // Expand all hierarchy nodes by default up to depth 2
    function expandToDepth(depth) {
      document.querySelectorAll('.hierarchy-tree > li > .hierarchy-toggle').forEach(function(toggle) {
        toggle.classList.add('expanded');
        var content = toggle.nextElementSibling;
        if (content) content.classList.add('show');
      });
    }
    expandToDepth(2);
  </script>
</body>
</html>
)";
}
