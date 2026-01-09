//===- circt-cov.cpp - Coverage data manipulation tool --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-cov' tool for manipulating coverage
// databases. It supports merging coverage data from multiple runs,
// generating reports in various formats, and comparing coverage data.
//
// Usage:
//   circt-cov merge run1.cov run2.cov -o merged.cov
//   circt-cov report coverage.cov --format=text|html|json
//   circt-cov diff old.cov new.cov
//   circt-cov exclude coverage.cov --exclusions=exclusions.json
//   circt-cov trend coverage.cov --add-point --run-id=<id>
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CoverageDatabase.h"
#include "circt/Support/Version.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

namespace cl = llvm::cl;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-cov Options");

enum SubCommand { None, Merge, Report, Diff, Exclude, Trend, Convert };

static cl::opt<SubCommand> command(
    cl::desc("Command to execute:"),
    cl::values(clEnumValN(Merge, "merge", "Merge multiple coverage databases"),
               clEnumValN(Report, "report", "Generate coverage report"),
               clEnumValN(Diff, "diff", "Compare two coverage databases"),
               clEnumValN(Exclude, "exclude", "Apply exclusions to coverage"),
               clEnumValN(Trend, "trend", "Track coverage trends over time"),
               clEnumValN(Convert, "convert",
                          "Convert between coverage formats")),
    cl::init(None), cl::cat(mainCategory));

static cl::list<std::string> inputFiles(cl::Positional,
                                        cl::desc("<input files>"),
                                        cl::cat(mainCategory));

static cl::opt<std::string> outputFile("o", cl::desc("Output file"),
                                       cl::value_desc("filename"),
                                       cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string>
    format("format", cl::desc("Output format (text, html, json, binary)"),
           cl::value_desc("format"), cl::init("text"), cl::cat(mainCategory));

static cl::opt<std::string>
    exclusionsFile("exclusions", cl::desc("Exclusions file (JSON format)"),
                   cl::value_desc("filename"), cl::cat(mainCategory));

static cl::opt<std::string> runId("run-id", cl::desc("Run identifier for trend tracking"),
                                  cl::value_desc("id"), cl::cat(mainCategory));

static cl::opt<std::string>
    commitHash("commit", cl::desc("Git commit hash for trend tracking"),
               cl::value_desc("hash"), cl::cat(mainCategory));

static cl::opt<bool> addPoint("add-point",
                              cl::desc("Add current coverage as trend point"),
                              cl::cat(mainCategory));

static cl::opt<bool>
    verbose("v", cl::desc("Verbose output"), cl::cat(mainCategory));

static cl::opt<double>
    coverageThreshold("threshold",
                      cl::desc("Coverage threshold for pass/fail (0-100)"),
                      cl::value_desc("percent"), cl::init(0.0),
                      cl::cat(mainCategory));

static cl::opt<std::string>
    hierarchyFilter("hierarchy",
                    cl::desc("Filter results by hierarchy prefix"),
                    cl::value_desc("prefix"), cl::cat(mainCategory));

static cl::opt<bool> uncoveredOnly("uncovered-only",
                                   cl::desc("Show only uncovered points"),
                                   cl::cat(mainCategory));

static cl::opt<bool> showTrends("show-trends",
                                cl::desc("Include trend data in report"),
                                cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Detect file format from extension or content.
static bool isJSONFormat(llvm::StringRef path) {
  return path.ends_with(".json");
}

/// Load a coverage database from file, auto-detecting format.
static llvm::Expected<CoverageDatabase> loadDatabase(llvm::StringRef path) {
  if (isJSONFormat(path))
    return CoverageDatabase::readFromJSON(path);
  return CoverageDatabase::readFromFile(path);
}

/// Save a coverage database to file based on format option.
static llvm::Error saveDatabase(const CoverageDatabase &db,
                                llvm::StringRef path,
                                llvm::StringRef formatStr) {
  if (formatStr == "json" || isJSONFormat(path))
    return db.writeToJSON(path);
  return db.writeToFile(path);
}

/// Format a percentage with color (for terminal output).
static void printCoveragePercent(llvm::raw_ostream &os, double percent,
                                 bool useColor = true) {
  if (useColor) {
    if (percent >= 90.0)
      os << "\033[32m"; // Green
    else if (percent >= 70.0)
      os << "\033[33m"; // Yellow
    else
      os << "\033[31m"; // Red
  }
  os << llvm::format("%6.2f%%", percent);
  if (useColor)
    os << "\033[0m";
}

/// Generate a progress bar.
static std::string progressBar(double percent, int width = 20) {
  int filled = static_cast<int>(std::round(percent / 100.0 * width));
  std::string bar;
  bar.reserve(width + 2);
  bar += "[";
  for (int i = 0; i < width; ++i) {
    if (i < filled)
      bar += "=";
    else if (i == filled)
      bar += ">";
    else
      bar += " ";
  }
  bar += "]";
  return bar;
}

//===----------------------------------------------------------------------===//
// Merge Command
//===----------------------------------------------------------------------===//

static int runMerge() {
  if (inputFiles.size() < 2) {
    llvm::errs() << "Error: merge command requires at least 2 input files\n";
    return 1;
  }

  // Load the first database
  auto dbOrErr = loadDatabase(inputFiles[0]);
  if (!dbOrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(dbOrErr.takeError()) << "\n";
    return 1;
  }

  CoverageDatabase merged = std::move(*dbOrErr);

  if (verbose)
    llvm::outs() << "Loaded " << inputFiles[0] << " with "
                 << merged.getTotalPointCount() << " coverage points\n";

  // Merge remaining databases
  for (size_t i = 1; i < inputFiles.size(); ++i) {
    auto otherOrErr = loadDatabase(inputFiles[i]);
    if (!otherOrErr) {
      llvm::errs() << "Error loading " << inputFiles[i] << ": "
                   << llvm::toString(otherOrErr.takeError()) << "\n";
      return 1;
    }

    if (verbose)
      llvm::outs() << "Merging " << inputFiles[i] << " with "
                   << otherOrErr->getTotalPointCount() << " coverage points\n";

    merged.merge(*otherOrErr);
  }

  // Save the merged database
  if (auto err = saveDatabase(merged, outputFile, format)) {
    llvm::errs() << "Error writing output: " << llvm::toString(std::move(err))
                 << "\n";
    return 1;
  }

  llvm::outs() << "Merged " << inputFiles.size() << " databases into "
               << outputFile << "\n";
  llvm::outs() << "Total coverage points: " << merged.getTotalPointCount()
               << "\n";
  llvm::outs() << "Overall coverage: ";
  printCoveragePercent(llvm::outs(), merged.getOverallCoverage());
  llvm::outs() << "\n";

  return 0;
}

//===----------------------------------------------------------------------===//
// Report Command - Text Format
//===----------------------------------------------------------------------===//

static void generateTextReport(const CoverageDatabase &db,
                               llvm::raw_ostream &os) {
  os << std::string(80, '=') << "\n";
  os << "                       COVERAGE REPORT\n";
  os << std::string(80, '=') << "\n\n";

  // Summary section
  os << "SUMMARY\n";
  os << std::string(80, '-') << "\n";

  os << llvm::format("%-30s", "Overall Coverage:");
  printCoveragePercent(os, db.getOverallCoverage());
  os << " " << progressBar(db.getOverallCoverage()) << "\n";

  os << llvm::format("%-30s", "Total Points:");
  os << db.getTotalPointCount() << "\n";

  os << llvm::format("%-30s", "Covered Points:");
  os << db.getCoveredPointCount() << "\n\n";

  // Coverage by type
  os << "COVERAGE BY TYPE\n";
  os << std::string(80, '-') << "\n";

  auto printTypeCoverage = [&](CoverageType type, const char *name) {
    size_t total = db.getTotalPointCountByType(type);
    if (total > 0) {
      double percent = db.getCoverageByType(type);
      size_t covered = db.getCoveredPointCountByType(type);
      os << llvm::format("%-20s", name);
      printCoveragePercent(os, percent);
      os << " " << progressBar(percent)
         << llvm::format(" (%zu/%zu)\n", covered, total);
    }
  };

  printTypeCoverage(CoverageType::Line, "Line Coverage:");
  printTypeCoverage(CoverageType::Toggle, "Toggle Coverage:");
  printTypeCoverage(CoverageType::Branch, "Branch Coverage:");
  printTypeCoverage(CoverageType::Condition, "Condition Coverage:");
  printTypeCoverage(CoverageType::FSM, "FSM Coverage:");
  printTypeCoverage(CoverageType::Assertion, "Assertion Coverage:");
  printTypeCoverage(CoverageType::Coverpoint, "Coverpoint Coverage:");
  os << "\n";

  // Groups
  const auto &groups = db.getCoverageGroups();
  if (!groups.empty()) {
    os << "COVERAGE GROUPS\n";
    os << std::string(80, '-') << "\n";

    for (const auto &kv : groups) {
      const auto &group = kv.second;
      double percent = group.getCoveragePercent(db.getCoveragePoints());
      os << llvm::format("%-40s", group.name.c_str());
      printCoveragePercent(os, percent);
      os << " " << progressBar(percent) << "\n";
    }
    os << "\n";
  }

  // Uncovered points
  if (uncoveredOnly || verbose) {
    os << "UNCOVERED POINTS\n";
    os << std::string(80, '-') << "\n";

    int uncoveredCount = 0;
    for (const auto &kv : db.getCoveragePoints()) {
      const auto &point = kv.second;
      if (point.isCovered())
        continue;
      if (db.isExcluded(point.name))
        continue;
      if (!hierarchyFilter.empty() &&
          !llvm::StringRef(point.hierarchy).starts_with(hierarchyFilter))
        continue;

      ++uncoveredCount;
      os << "  " << point.name << "\n";
      os << "    Type: " << getCoverageTypeName(point.type) << "\n";
      if (!point.location.filename.empty()) {
        os << "    Location: " << point.location.filename << ":"
           << point.location.line;
        if (point.location.column > 0)
          os << ":" << point.location.column;
        os << "\n";
      }
      if (!point.hierarchy.empty())
        os << "    Hierarchy: " << point.hierarchy << "\n";
      os << "\n";
    }

    if (uncoveredCount == 0)
      os << "  (none)\n\n";
  }

  // Exclusions
  const auto &exclusions = db.getExclusions();
  if (!exclusions.empty()) {
    os << "EXCLUSIONS\n";
    os << std::string(80, '-') << "\n";

    for (const auto &exclusion : exclusions) {
      os << "  " << exclusion.pointName << "\n";
      os << "    Reason: " << exclusion.reason << "\n";
      if (!exclusion.author.empty())
        os << "    Author: " << exclusion.author << "\n";
      os << "\n";
    }
  }

  // Trends
  if (showTrends && !db.getTrends().empty()) {
    os << "COVERAGE TRENDS\n";
    os << std::string(80, '-') << "\n";
    os << llvm::format("%-24s %-16s %-10s %-10s %-10s\n", "Timestamp", "Run ID",
                       "Line", "Toggle", "Overall");

    for (const auto &trend : db.getTrends()) {
      os << llvm::format("%-24s %-16s %9.2f%% %9.2f%% %9.2f%%\n",
                         trend.timestamp.c_str(), trend.runId.c_str(),
                         trend.lineCoverage, trend.toggleCoverage,
                         trend.overallCoverage);
    }
  }
}

//===----------------------------------------------------------------------===//
// Report Command - HTML Format
//===----------------------------------------------------------------------===//

static void generateHTMLReport(const CoverageDatabase &db,
                               llvm::raw_ostream &os) {
  // HTML header and styles
  os << R"(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CIRCT Coverage Report</title>
  <style>
    :root {
      --bg-color: #f5f5f5;
      --card-bg: white;
      --text-color: #333;
      --border-color: #ddd;
      --green: #28a745;
      --yellow: #ffc107;
      --red: #dc3545;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
      background: var(--bg-color);
      color: var(--text-color);
      margin: 0;
      padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    .card {
      background: var(--card-bg);
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin-bottom: 20px;
      padding: 20px;
    }
    .card h2 {
      margin-top: 0;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border-color);
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
    }
    .metric {
      text-align: center;
      padding: 20px;
    }
    .metric-value {
      font-size: 36px;
      font-weight: bold;
    }
    .metric-label { color: #666; margin-top: 5px; }
    .progress-bar {
      height: 24px;
      background: #e9ecef;
      border-radius: 4px;
      overflow: hidden;
      margin: 10px 0;
    }
    .progress-fill {
      height: 100%;
      transition: width 0.3s ease;
    }
    .green { background: var(--green); color: white; }
    .yellow { background: var(--yellow); color: black; }
    .red { background: var(--red); color: white; }
    .coverage-row {
      display: flex;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid var(--border-color);
    }
    .coverage-row:last-child { border-bottom: none; }
    .coverage-name { flex: 1; }
    .coverage-percent { width: 80px; text-align: right; font-weight: bold; }
    .coverage-bar { width: 200px; margin-left: 20px; }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      padding: 12px;
      border-bottom: 1px solid var(--border-color);
    }
    th { background: #f8f9fa; font-weight: 600; }
    .uncovered { background: #fff3cd; }
    .covered { background: #d4edda; }
    .excluded { background: #e2e3e5; }
    .collapsible {
      cursor: pointer;
      user-select: none;
    }
    .collapsible:before {
      content: '\25B6';
      display: inline-block;
      margin-right: 8px;
      transition: transform 0.2s;
    }
    .collapsible.active:before { transform: rotate(90deg); }
    .content { display: none; padding-left: 20px; }
    .content.show { display: block; }
    .trend-chart {
      height: 200px;
      display: flex;
      align-items: flex-end;
      gap: 4px;
      padding: 20px 0;
    }
    .trend-bar {
      flex: 1;
      background: var(--green);
      min-height: 10px;
      border-radius: 4px 4px 0 0;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>CIRCT Coverage Report</h1>
)";

  // Helper to get color class
  auto getColorClass = [](double percent) -> const char * {
    if (percent >= 90.0)
      return "green";
    if (percent >= 70.0)
      return "yellow";
    return "red";
  };

  // Summary card
  double overall = db.getOverallCoverage();
  os << R"(
    <div class="card">
      <h2>Summary</h2>
      <div class="summary-grid">
        <div class="metric">
          <div class="metric-value )" << getColorClass(overall) << R"(">)"
     << llvm::format("%.1f%%", overall) << R"(</div>
          <div class="metric-label">Overall Coverage</div>
        </div>
        <div class="metric">
          <div class="metric-value">)" << db.getTotalPointCount() << R"(</div>
          <div class="metric-label">Total Points</div>
        </div>
        <div class="metric">
          <div class="metric-value">)" << db.getCoveredPointCount() << R"(</div>
          <div class="metric-label">Covered Points</div>
        </div>
        <div class="metric">
          <div class="metric-value">)" << db.getExclusions().size() << R"(</div>
          <div class="metric-label">Exclusions</div>
        </div>
      </div>
    </div>
)";

  // Coverage by type
  os << R"(
    <div class="card">
      <h2>Coverage by Type</h2>
)";

  auto printTypeRow = [&](CoverageType type, const char *name) {
    size_t total = db.getTotalPointCountByType(type);
    if (total > 0) {
      double percent = db.getCoverageByType(type);
      size_t covered = db.getCoveredPointCountByType(type);
      os << R"(      <div class="coverage-row">
        <div class="coverage-name">)" << name << R"( <small>()"
         << covered << "/" << total << R"()</small></div>
        <div class="coverage-percent )" << getColorClass(percent) << R"(">)"
         << llvm::format("%.1f%%", percent) << R"(</div>
        <div class="coverage-bar">
          <div class="progress-bar">
            <div class="progress-fill )" << getColorClass(percent)
         << R"(" style="width: )" << percent << R"(%"></div>
          </div>
        </div>
      </div>
)";
    }
  };

  printTypeRow(CoverageType::Line, "Line Coverage");
  printTypeRow(CoverageType::Toggle, "Toggle Coverage");
  printTypeRow(CoverageType::Branch, "Branch Coverage");
  printTypeRow(CoverageType::Condition, "Condition Coverage");
  printTypeRow(CoverageType::FSM, "FSM Coverage");
  printTypeRow(CoverageType::Assertion, "Assertion Coverage");
  printTypeRow(CoverageType::Coverpoint, "Coverpoint Coverage");

  os << "    </div>\n";

  // Coverage groups
  const auto &groups = db.getCoverageGroups();
  if (!groups.empty()) {
    os << R"(
    <div class="card">
      <h2>Coverage Groups</h2>
)";
    for (const auto &kv : groups) {
      const auto &group = kv.second;
      double percent = group.getCoveragePercent(db.getCoveragePoints());
      os << R"(      <div class="coverage-row">
        <div class="coverage-name">)" << group.name;
      if (!group.description.empty())
        os << R"( <small>)" << group.description << "</small>";
      os << R"(</div>
        <div class="coverage-percent )" << getColorClass(percent) << R"(">)"
         << llvm::format("%.1f%%", percent) << R"(</div>
        <div class="coverage-bar">
          <div class="progress-bar">
            <div class="progress-fill )" << getColorClass(percent)
         << R"(" style="width: )" << percent << R"(%"></div>
          </div>
        </div>
      </div>
)";
    }
    os << "    </div>\n";
  }

  // Uncovered points
  os << R"(
    <div class="card">
      <h2>Coverage Details</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Location</th>
            <th>Hierarchy</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
)";

  for (const auto &kv : db.getCoveragePoints()) {
    const auto &point = kv.second;
    if (!hierarchyFilter.empty() &&
        !llvm::StringRef(point.hierarchy).starts_with(hierarchyFilter))
      continue;
    if (uncoveredOnly && (point.isCovered() || db.isExcluded(point.name)))
      continue;

    const char *rowClass = "uncovered";
    const char *status = "Uncovered";
    if (db.isExcluded(point.name)) {
      rowClass = "excluded";
      status = "Excluded";
    } else if (point.isCovered()) {
      rowClass = "covered";
      status = "Covered";
    }

    os << R"(          <tr class=")" << rowClass << R"(">
            <td>)" << point.name << R"(</td>
            <td>)" << getCoverageTypeName(point.type) << R"(</td>
            <td>)";
    if (!point.location.filename.empty()) {
      os << point.location.filename << ":" << point.location.line;
    }
    os << R"(</td>
            <td>)" << point.hierarchy << R"(</td>
            <td>)" << status << R"(</td>
          </tr>
)";
  }

  os << R"(        </tbody>
      </table>
    </div>
)";

  // Trend chart
  const auto &trends = db.getTrends();
  if (!trends.empty()) {
    os << R"(
    <div class="card">
      <h2>Coverage Trends</h2>
      <div class="trend-chart">
)";
    for (const auto &trend : trends) {
      os << R"(        <div class="trend-bar" style="height: )"
         << trend.overallCoverage << R"(%" title=")"
         << trend.timestamp << ": " << llvm::format("%.1f%%", trend.overallCoverage)
         << R"("></div>
)";
    }
    os << R"(      </div>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Run ID</th>
            <th>Line</th>
            <th>Toggle</th>
            <th>Branch</th>
            <th>Overall</th>
          </tr>
        </thead>
        <tbody>
)";
    for (const auto &trend : trends) {
      os << R"(          <tr>
            <td>)" << trend.timestamp << R"(</td>
            <td>)" << trend.runId << R"(</td>
            <td>)" << llvm::format("%.1f%%", trend.lineCoverage) << R"(</td>
            <td>)" << llvm::format("%.1f%%", trend.toggleCoverage) << R"(</td>
            <td>)" << llvm::format("%.1f%%", trend.branchCoverage) << R"(</td>
            <td class=")" << getColorClass(trend.overallCoverage) << R"(">)"
         << llvm::format("%.1f%%", trend.overallCoverage) << R"(</td>
          </tr>
)";
    }
    os << R"(        </tbody>
      </table>
    </div>
)";
  }

  // Footer
  os << R"(
    <div class="card" style="text-align: center; color: #666;">
      Generated by circt-cov ()" << getCirctVersion() << R"()
    </div>
  </div>
  <script>
    // Collapsible sections
    document.querySelectorAll('.collapsible').forEach(function(element) {
      element.addEventListener('click', function() {
        this.classList.toggle('active');
        var content = this.nextElementSibling;
        content.classList.toggle('show');
      });
    });
  </script>
</body>
</html>
)";
}

//===----------------------------------------------------------------------===//
// Report Command
//===----------------------------------------------------------------------===//

static int runReport() {
  if (inputFiles.empty()) {
    llvm::errs() << "Error: report command requires an input file\n";
    return 1;
  }

  auto dbOrErr = loadDatabase(inputFiles[0]);
  if (!dbOrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(dbOrErr.takeError()) << "\n";
    return 1;
  }

  CoverageDatabase &db = *dbOrErr;

  // Load exclusions if provided
  if (!exclusionsFile.empty()) {
    if (auto err = db.loadExclusions(exclusionsFile)) {
      llvm::errs() << "Warning: failed to load exclusions: "
                   << llvm::toString(std::move(err)) << "\n";
    }
  }

  // Generate report
  std::error_code ec;
  llvm::raw_fd_ostream fileOs(outputFile, ec);
  if (ec && outputFile != "-") {
    llvm::errs() << "Error opening output file: " << ec.message() << "\n";
    return 1;
  }

  llvm::raw_ostream &os = (outputFile == "-") ? llvm::outs() : fileOs;

  if (format == "html") {
    generateHTMLReport(db, os);
  } else if (format == "json") {
    os << db.toJSON();
  } else {
    // Default to text
    generateTextReport(db, os);
  }

  // Check threshold
  double threshold = coverageThreshold;
  if (threshold > 0.0) {
    double coverage = db.getOverallCoverage();
    if (coverage < threshold) {
      llvm::errs() << "Coverage " << llvm::format("%.2f%%", coverage)
                   << " is below threshold "
                   << llvm::format("%.2f%%", threshold) << "\n";
      return 1;
    }
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Diff Command
//===----------------------------------------------------------------------===//

static int runDiff() {
  if (inputFiles.size() != 2) {
    llvm::errs() << "Error: diff command requires exactly 2 input files\n";
    return 1;
  }

  auto db1OrErr = loadDatabase(inputFiles[0]);
  if (!db1OrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(db1OrErr.takeError()) << "\n";
    return 1;
  }

  auto db2OrErr = loadDatabase(inputFiles[1]);
  if (!db2OrErr) {
    llvm::errs() << "Error loading " << inputFiles[1] << ": "
                 << llvm::toString(db2OrErr.takeError()) << "\n";
    return 1;
  }

  CoverageDatabase &newDb = *db1OrErr;
  CoverageDatabase &oldDb = *db2OrErr;

  auto result = newDb.diff(oldDb);

  llvm::outs() << "Coverage Comparison\n";
  llvm::outs() << "==================\n\n";

  llvm::outs() << "Coverage delta: ";
  if (result.coverageDelta >= 0)
    llvm::outs() << "+";
  llvm::outs() << llvm::format("%.2f%%", result.coverageDelta) << "\n\n";

  llvm::outs() << inputFiles[0] << ": ";
  printCoveragePercent(llvm::outs(), newDb.getOverallCoverage());
  llvm::outs() << " (" << newDb.getCoveredPointCount() << "/"
               << newDb.getTotalPointCount() << ")\n";

  llvm::outs() << inputFiles[1] << ": ";
  printCoveragePercent(llvm::outs(), oldDb.getOverallCoverage());
  llvm::outs() << " (" << oldDb.getCoveredPointCount() << "/"
               << oldDb.getTotalPointCount() << ")\n\n";

  if (!result.newlyCovered.empty()) {
    llvm::outs() << "Newly covered (" << result.newlyCovered.size() << "):\n";
    for (const auto &name : result.newlyCovered) {
      llvm::outs() << "  + " << name << "\n";
    }
    llvm::outs() << "\n";
  }

  if (!result.newlyUncovered.empty()) {
    llvm::outs() << "Newly uncovered (" << result.newlyUncovered.size()
                 << "):\n";
    for (const auto &name : result.newlyUncovered) {
      llvm::outs() << "  - " << name << "\n";
    }
    llvm::outs() << "\n";
  }

  if (verbose) {
    if (!result.onlyInThis.empty()) {
      llvm::outs() << "Only in " << inputFiles[0] << " ("
                   << result.onlyInThis.size() << "):\n";
      for (const auto &name : result.onlyInThis) {
        llvm::outs() << "  " << name << "\n";
      }
      llvm::outs() << "\n";
    }

    if (!result.onlyInOther.empty()) {
      llvm::outs() << "Only in " << inputFiles[1] << " ("
                   << result.onlyInOther.size() << "):\n";
      for (const auto &name : result.onlyInOther) {
        llvm::outs() << "  " << name << "\n";
      }
      llvm::outs() << "\n";
    }
  }

  // Return non-zero if coverage decreased
  if (result.coverageDelta < 0) {
    llvm::errs() << "Warning: coverage decreased by "
                 << llvm::format("%.2f%%", -result.coverageDelta) << "\n";
    return 1;
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Exclude Command
//===----------------------------------------------------------------------===//

static int runExclude() {
  if (inputFiles.empty()) {
    llvm::errs() << "Error: exclude command requires an input file\n";
    return 1;
  }

  if (exclusionsFile.empty()) {
    llvm::errs() << "Error: --exclusions file is required\n";
    return 1;
  }

  auto dbOrErr = loadDatabase(inputFiles[0]);
  if (!dbOrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(dbOrErr.takeError()) << "\n";
    return 1;
  }

  CoverageDatabase &db = *dbOrErr;

  // Load and apply exclusions
  if (auto err = db.loadExclusions(exclusionsFile)) {
    llvm::errs() << "Error loading exclusions: "
                 << llvm::toString(std::move(err)) << "\n";
    return 1;
  }

  llvm::outs() << "Applied " << db.getExclusions().size() << " exclusions\n";
  llvm::outs() << "Coverage after exclusions: ";
  printCoveragePercent(llvm::outs(), db.getOverallCoverage());
  llvm::outs() << "\n";

  // Save updated database
  if (outputFile != "-") {
    if (auto err = saveDatabase(db, outputFile, format)) {
      llvm::errs() << "Error writing output: " << llvm::toString(std::move(err))
                   << "\n";
      return 1;
    }
    llvm::outs() << "Saved to " << outputFile << "\n";
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Trend Command
//===----------------------------------------------------------------------===//

static int runTrend() {
  if (inputFiles.empty()) {
    llvm::errs() << "Error: trend command requires an input file\n";
    return 1;
  }

  auto dbOrErr = loadDatabase(inputFiles[0]);
  if (!dbOrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(dbOrErr.takeError()) << "\n";
    return 1;
  }

  CoverageDatabase &db = *dbOrErr;

  if (addPoint) {
    // Add current coverage as a trend point
    std::string id = runId.empty() ? "run" : runId.getValue();
    auto trend = db.createCurrentTrendPoint(id, commitHash);
    db.addTrendPoint(trend);

    llvm::outs() << "Added trend point:\n";
    llvm::outs() << "  Timestamp: " << trend.timestamp << "\n";
    llvm::outs() << "  Run ID: " << trend.runId << "\n";
    if (!trend.commitHash.empty())
      llvm::outs() << "  Commit: " << trend.commitHash << "\n";
    llvm::outs() << "  Overall: "
                 << llvm::format("%.2f%%", trend.overallCoverage) << "\n";

    // Save updated database
    if (outputFile != "-") {
      if (auto err = saveDatabase(db, outputFile, format)) {
        llvm::errs() << "Error writing output: "
                     << llvm::toString(std::move(err)) << "\n";
        return 1;
      }
    }
  }

  // Show trend history
  const auto &trends = db.getTrends();
  if (trends.empty()) {
    llvm::outs() << "No trend data available\n";
    return 0;
  }

  llvm::outs() << "\nCoverage Trend History:\n";
  llvm::outs() << llvm::format("%-24s %-16s %-10s %-10s %-10s %-10s\n",
                               "Timestamp", "Run ID", "Line", "Toggle",
                               "Branch", "Overall");
  llvm::outs() << std::string(80, '-') << "\n";

  double prevCoverage = 0.0;
  for (size_t i = 0; i < trends.size(); ++i) {
    const auto &trend = trends[i];
    llvm::outs() << llvm::format("%-24s %-16s %9.2f%% %9.2f%% %9.2f%% ",
                                 trend.timestamp.c_str(), trend.runId.c_str(),
                                 trend.lineCoverage, trend.toggleCoverage,
                                 trend.branchCoverage);
    printCoveragePercent(llvm::outs(), trend.overallCoverage);

    // Show delta from previous
    if (i > 0) {
      double delta = trend.overallCoverage - prevCoverage;
      if (delta >= 0)
        llvm::outs() << " (+";
      else
        llvm::outs() << " (";
      llvm::outs() << llvm::format("%.2f%%", delta) << ")";
    }
    llvm::outs() << "\n";
    prevCoverage = trend.overallCoverage;
  }

  return 0;
}

//===----------------------------------------------------------------------===//
// Convert Command
//===----------------------------------------------------------------------===//

static int runConvert() {
  if (inputFiles.empty()) {
    llvm::errs() << "Error: convert command requires an input file\n";
    return 1;
  }

  auto dbOrErr = loadDatabase(inputFiles[0]);
  if (!dbOrErr) {
    llvm::errs() << "Error loading " << inputFiles[0] << ": "
                 << llvm::toString(dbOrErr.takeError()) << "\n";
    return 1;
  }

  if (auto err = saveDatabase(*dbOrErr, outputFile, format)) {
    llvm::errs() << "Error writing output: " << llvm::toString(std::move(err))
                 << "\n";
    return 1;
  }

  llvm::outs() << "Converted " << inputFiles[0] << " to " << outputFile
               << " (" << format << " format)\n";

  return 0;
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

static void printHelp() {
  llvm::outs() << R"(
circt-cov - CIRCT Coverage Database Tool

USAGE: circt-cov <command> [options] <input files>

COMMANDS:
  merge     Merge multiple coverage databases
  report    Generate coverage report (text, html, json)
  diff      Compare two coverage databases
  exclude   Apply exclusions to coverage data
  trend     Track coverage trends over time
  convert   Convert between coverage formats

EXAMPLES:
  circt-cov merge run1.cov run2.cov -o merged.cov
  circt-cov report coverage.cov --format=html -o report.html
  circt-cov diff new.cov old.cov
  circt-cov exclude coverage.cov --exclusions=exclude.json -o updated.cov
  circt-cov trend coverage.cov --add-point --run-id=ci-123
  circt-cov convert coverage.cov -o coverage.json --format=json

OPTIONS:
)";
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::setBugReportMsg(circtBugReportMsg);

  cl::HideUnrelatedOptions(mainCategory);
  cl::ParseCommandLineOptions(argc, argv, "CIRCT Coverage Database Tool\n");

  // If no command specified, show help
  if (command == None) {
    printHelp();
    cl::PrintHelpMessage();
    return 0;
  }

  switch (command) {
  case Merge:
    return runMerge();
  case Report:
    return runReport();
  case Diff:
    return runDiff();
  case Exclude:
    return runExclude();
  case Trend:
    return runTrend();
  case Convert:
    return runConvert();
  default:
    llvm::errs() << "Unknown command\n";
    return 1;
  }
}
