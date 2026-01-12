//===- CoverageDatabase.h - Coverage data storage ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CoverageDatabase class for storing and manipulating
// coverage data. This is similar to UCDB (Unified Coverage Database) format
// used in commercial simulators.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_COVERAGEDATABASE_H
#define CIRCT_SUPPORT_COVERAGEDATABASE_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace circt {

/// Source location information for a coverage point.
struct SourceLocation {
  std::string filename;
  uint32_t line = 0;
  uint32_t column = 0;

  bool operator==(const SourceLocation &other) const {
    return filename == other.filename && line == other.line &&
           column == other.column;
  }
};

/// Type of coverage point.
enum class CoverageType {
  Line,      // Line/statement coverage
  Toggle,    // Signal toggle coverage (0->1, 1->0)
  Branch,    // Branch coverage (true/false)
  Condition, // Condition coverage
  FSM,       // FSM state/transition coverage
  Assertion, // Assertion coverage
  Coverpoint // User-defined coverpoint
};

/// Returns the string name for a coverage type.
StringRef getCoverageTypeName(CoverageType type);

/// Parses a coverage type from a string.
std::optional<CoverageType> parseCoverageType(StringRef name);

/// Single coverage point data.
struct CoveragePoint {
  std::string name;         // Unique identifier for this coverage point
  CoverageType type;        // Type of coverage
  uint64_t hits = 0;        // Number of times this point was hit
  uint64_t goal = 1;        // Target hit count (1 for most, more for stress)
  SourceLocation location;  // Source location
  std::string hierarchy;    // Module hierarchy path
  std::string description;  // Optional human-readable description

  // For toggle coverage - which transitions were observed
  bool toggle01 = false; // 0 -> 1 transition observed
  bool toggle10 = false; // 1 -> 0 transition observed

  // For branch coverage
  bool branchTrue = false;
  bool branchFalse = false;

  // Extra metadata as key-value pairs
  llvm::StringMap<std::string> metadata;

  /// Check if this coverage point is covered.
  bool isCovered() const;

  /// Get coverage percentage (0.0 - 100.0).
  double getCoveragePercent() const;

  /// Merge another coverage point into this one.
  void merge(const CoveragePoint &other);
};

/// Exclusion reason for a coverage point.
struct CoverageExclusion {
  std::string pointName;    // Name of excluded coverage point
  std::string reason;       // Reason for exclusion
  std::string author;       // Who approved the exclusion
  std::string date;         // When exclusion was added
  std::string ticketId;     // Optional bug/ticket reference

  bool operator==(const CoverageExclusion &other) const {
    return pointName == other.pointName;
  }
};

/// Coverage group containing related coverage points.
struct CoverageGroup {
  std::string name;
  std::string description;
  std::vector<std::string> pointNames; // References to coverage points

  /// Calculate aggregate coverage for this group.
  double getCoveragePercent(
      const llvm::StringMap<CoveragePoint> &allPoints) const;
};

/// Trend data point for tracking coverage over time.
struct CoverageTrendPoint {
  std::string timestamp;          // ISO 8601 timestamp
  std::string runId;              // Unique run identifier
  std::string commitHash;         // Git commit hash if available
  double lineCoverage = 0.0;      // Line coverage percentage
  double toggleCoverage = 0.0;    // Toggle coverage percentage
  double branchCoverage = 0.0;    // Branch coverage percentage
  double overallCoverage = 0.0;   // Overall coverage percentage
  uint64_t totalPoints = 0;       // Total coverage points
  uint64_t coveredPoints = 0;     // Covered coverage points
};

/// Main coverage database class.
/// This class stores coverage data in a format similar to UCDB and provides
/// operations for merging, querying, and exporting coverage data.
class CoverageDatabase {
public:
  /// Magic number for binary format.
  static constexpr uint32_t MAGIC = 0x43435644; // "CCVD"

  /// Current database version.
  static constexpr uint32_t VERSION = 1;

  CoverageDatabase() = default;

  //===--------------------------------------------------------------------===//
  // Coverage Point Operations
  //===--------------------------------------------------------------------===//

  /// Add or update a coverage point.
  void addCoveragePoint(const CoveragePoint &point);

  /// Record a hit for a coverage point.
  void recordHit(StringRef name, uint64_t count = 1);

  /// Get a coverage point by name.
  const CoveragePoint *getCoveragePoint(StringRef name) const;

  /// Check if a coverage point exists.
  bool hasCoveragePoint(StringRef name) const;

  /// Get all coverage points.
  const llvm::StringMap<CoveragePoint> &getCoveragePoints() const {
    return coveragePoints;
  }

  /// Get all coverage points of a specific type.
  std::vector<const CoveragePoint *> getCoveragePointsByType(
      CoverageType type) const;

  /// Get all coverage points in a hierarchy.
  std::vector<const CoveragePoint *> getCoveragePointsByHierarchy(
      StringRef hierarchyPrefix) const;

  //===--------------------------------------------------------------------===//
  // Coverage Group Operations
  //===--------------------------------------------------------------------===//

  /// Add a coverage group.
  void addCoverageGroup(const CoverageGroup &group);

  /// Get a coverage group by name.
  const CoverageGroup *getCoverageGroup(StringRef name) const;

  /// Get all coverage groups.
  const llvm::StringMap<CoverageGroup> &getCoverageGroups() const {
    return coverageGroups;
  }

  //===--------------------------------------------------------------------===//
  // Exclusion Operations
  //===--------------------------------------------------------------------===//

  /// Add an exclusion.
  void addExclusion(const CoverageExclusion &exclusion);

  /// Check if a coverage point is excluded.
  bool isExcluded(StringRef pointName) const;

  /// Get exclusion reason for a point.
  const CoverageExclusion *getExclusion(StringRef pointName) const;

  /// Get all exclusions.
  const std::vector<CoverageExclusion> &getExclusions() const {
    return exclusions;
  }

  /// Load exclusions from a file.
  llvm::Error loadExclusions(StringRef path);

  /// Save exclusions to a file.
  llvm::Error saveExclusions(StringRef path) const;

  //===--------------------------------------------------------------------===//
  // Trend Operations
  //===--------------------------------------------------------------------===//

  /// Add a trend data point.
  void addTrendPoint(const CoverageTrendPoint &trend);

  /// Get all trend data.
  const std::vector<CoverageTrendPoint> &getTrends() const { return trends; }

  /// Create a trend point from current coverage state.
  CoverageTrendPoint createCurrentTrendPoint(StringRef runId,
                                             StringRef commitHash = "") const;

  //===--------------------------------------------------------------------===//
  // Metrics and Statistics
  //===--------------------------------------------------------------------===//

  /// Get overall coverage percentage.
  double getOverallCoverage() const;

  /// Get coverage percentage by type.
  double getCoverageByType(CoverageType type) const;

  /// Get total number of coverage points.
  size_t getTotalPointCount() const { return coveragePoints.size(); }

  /// Get number of covered points.
  size_t getCoveredPointCount() const;

  /// Get number of covered points by type.
  size_t getCoveredPointCountByType(CoverageType type) const;

  /// Get total number of points by type.
  size_t getTotalPointCountByType(CoverageType type) const;

  //===--------------------------------------------------------------------===//
  // Merge Operations
  //===--------------------------------------------------------------------===//

  /// Merge another database into this one.
  /// Coverage hits are accumulated, and metadata is merged.
  void merge(const CoverageDatabase &other);

  /// Compare with another database and return difference statistics.
  struct DiffResult {
    std::vector<std::string> newlyCovered;     // Points covered in this but not other
    std::vector<std::string> newlyUncovered;   // Points covered in other but not this
    std::vector<std::string> onlyInThis;       // Points only in this database
    std::vector<std::string> onlyInOther;      // Points only in other database
    double coverageDelta = 0.0;                // Overall coverage change
  };

  DiffResult diff(const CoverageDatabase &other) const;

  //===--------------------------------------------------------------------===//
  // Persistence
  //===--------------------------------------------------------------------===//

  /// Write database to a file (binary format).
  llvm::Error writeToFile(StringRef path) const;

  /// Read database from a file (binary format).
  static llvm::Expected<CoverageDatabase> readFromFile(StringRef path);

  /// Write database to JSON format.
  llvm::Error writeToJSON(StringRef path) const;

  /// Read database from JSON format.
  static llvm::Expected<CoverageDatabase> readFromJSON(StringRef path);

  /// Convert to JSON value.
  llvm::json::Value toJSON() const;

  /// Parse from JSON value.
  static llvm::Expected<CoverageDatabase> fromJSON(const llvm::json::Value &json);

  //===--------------------------------------------------------------------===//
  // Metadata
  //===--------------------------------------------------------------------===//

  /// Set database metadata.
  void setMetadata(StringRef key, StringRef value) {
    metadata[key] = value.str();
  }

  /// Get database metadata.
  std::optional<StringRef> getMetadata(StringRef key) const {
    auto it = metadata.find(key);
    if (it != metadata.end())
      return StringRef(it->second);
    return std::nullopt;
  }

  /// Get all metadata.
  const llvm::StringMap<std::string> &getAllMetadata() const {
    return metadata;
  }

  /// Clear all data in the database.
  void clear();

private:
  /// All coverage points indexed by name.
  llvm::StringMap<CoveragePoint> coveragePoints;

  /// Coverage groups.
  llvm::StringMap<CoverageGroup> coverageGroups;

  /// Exclusions.
  std::vector<CoverageExclusion> exclusions;

  /// Trend data.
  std::vector<CoverageTrendPoint> trends;

  /// Database-level metadata.
  llvm::StringMap<std::string> metadata;

  /// Index of exclusions by point name for fast lookup.
  llvm::StringMap<size_t> exclusionIndex;
};

} // namespace circt

#endif // CIRCT_SUPPORT_COVERAGEDATABASE_H
