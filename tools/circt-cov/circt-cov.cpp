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
#include "circt/Support/CoverageReportGenerator.h"
#include "circt/Support/ResourceGuard.h"
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

static cl::opt<std::string>
    sourceBasePath("source-path",
                   cl::desc("Base path for resolving source file locations"),
                   cl::value_desc("path"), cl::cat(mainCategory));

static cl::opt<std::string>
    reportTitle("title", cl::desc("Custom title for HTML report"),
                cl::value_desc("title"), cl::init("CIRCT Coverage Report"),
                cl::cat(mainCategory));

static cl::opt<bool>
    noSourceAnnotations("no-source-annotations",
                        cl::desc("Disable source file annotations in HTML"),
                        cl::cat(mainCategory));

static cl::opt<bool>
    incrementalMerge("incremental",
                     cl::desc("Incremental merge (add to first file)"),
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
  // For incremental merge, we only need 1 input file (the new coverage)
  // and the output file should be an existing database to merge into.
  if (incrementalMerge) {
    if (inputFiles.empty()) {
      llvm::errs() << "Error: incremental merge requires at least 1 input file\n";
      return 1;
    }

    // Load existing database if output file exists
    CoverageDatabase merged;
    if (outputFile != "-" && llvm::sys::fs::exists(outputFile)) {
      auto existingOrErr = loadDatabase(outputFile);
      if (!existingOrErr) {
        llvm::errs() << "Error loading existing database " << outputFile << ": "
                     << llvm::toString(existingOrErr.takeError()) << "\n";
        return 1;
      }
      merged = std::move(*existingOrErr);
      if (verbose)
        llvm::outs() << "Loaded existing database " << outputFile << " with "
                     << merged.getTotalPointCount() << " coverage points\n";
    }

    // Merge all input files
    for (size_t i = 0; i < inputFiles.size(); ++i) {
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

    llvm::outs() << "Incrementally merged " << inputFiles.size()
                 << " database(s) into " << outputFile << "\n";
    llvm::outs() << "Total coverage points: " << merged.getTotalPointCount()
                 << "\n";
    llvm::outs() << "Overall coverage: ";
    printCoveragePercent(llvm::outs(), merged.getOverallCoverage());
    llvm::outs() << "\n";

    return 0;
  }

  // Standard merge requires at least 2 input files
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
    // Use the enhanced HTML report generator
    HTMLReportOptions htmlOptions;
    htmlOptions.title = reportTitle;
    htmlOptions.includeSourceAnnotations = !noSourceAnnotations;
    htmlOptions.includeTrends = showTrends || !db.getTrends().empty();
    htmlOptions.uncoveredOnly = uncoveredOnly;
    htmlOptions.hierarchyFilter = hierarchyFilter;
    htmlOptions.sourceBasePath = sourceBasePath;

    CoverageReportGenerator generator(db, htmlOptions);
    if (auto err = generator.generateReport(os)) {
      llvm::errs() << "Error generating HTML report: "
                   << llvm::toString(std::move(err)) << "\n";
      return 1;
    }
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

  cl::HideUnrelatedOptions({&mainCategory, &circt::getResourceGuardCategory()});
  cl::ParseCommandLineOptions(argc, argv, "CIRCT Coverage Database Tool\n");
  circt::installResourceGuard();

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
