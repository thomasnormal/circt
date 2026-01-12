//===- IncrementalCompiler.h - Incremental compilation support ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the incremental compilation infrastructure for simulation.
// It provides hash-based change detection, artifact caching, and incremental
// relinking for efficient recompilation of modified designs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_INCREMENTALCOMPILER_H
#define CIRCT_DIALECT_SIM_INCREMENTALCOMPILER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// ContentHash - 128-bit hash for content identification
//===----------------------------------------------------------------------===//

/// A 128-bit content hash for identifying module/signal content.
struct ContentHash {
  uint64_t high;
  uint64_t low;

  ContentHash() : high(0), low(0) {}
  ContentHash(uint64_t h, uint64_t l) : high(h), low(l) {}

  bool operator==(const ContentHash &other) const {
    return high == other.high && low == other.low;
  }

  bool operator!=(const ContentHash &other) const { return !(*this == other); }

  bool operator<(const ContentHash &other) const {
    return high < other.high || (high == other.high && low < other.low);
  }

  bool isZero() const { return high == 0 && low == 0; }

  /// Create a hash from a string using xxHash.
  static ContentHash fromString(llvm::StringRef str);

  /// Create a hash from binary data.
  static ContentHash fromData(const void *data, size_t size);

  /// Combine two hashes.
  ContentHash combine(const ContentHash &other) const;

  /// Convert to a hex string for debugging/storage.
  std::string toHexString() const;

  /// Parse from a hex string.
  static ContentHash fromHexString(llvm::StringRef hex);
};

/// Hash function for use in unordered containers.
struct ContentHashHasher {
  size_t operator()(const ContentHash &hash) const {
    return hash.high ^ hash.low;
  }
};

//===----------------------------------------------------------------------===//
// ModuleId - Unique identifier for a compilation unit
//===----------------------------------------------------------------------===//

/// Unique identifier for a module/compilation unit.
using ModuleId = std::string;

//===----------------------------------------------------------------------===//
// CompilationUnit - A single unit of compilation
//===----------------------------------------------------------------------===//

/// Represents a single unit of compilation (typically a module).
struct CompilationUnit {
  /// Unique identifier for this compilation unit.
  ModuleId id;

  /// Human-readable name.
  std::string name;

  /// Content hash of the source.
  ContentHash sourceHash;

  /// Content hash of the compiled artifact.
  ContentHash artifactHash;

  /// Dependencies on other compilation units.
  llvm::SmallVector<ModuleId, 8> dependencies;

  /// Modules that depend on this unit.
  llvm::SmallVector<ModuleId, 8> dependents;

  /// Timestamp of last compilation.
  std::chrono::system_clock::time_point lastCompiled;

  /// Whether this unit needs recompilation.
  bool dirty = true;

  /// Whether compilation succeeded.
  bool valid = false;

  CompilationUnit() = default;
  CompilationUnit(ModuleId id, std::string name)
      : id(std::move(id)), name(std::move(name)) {}
};

//===----------------------------------------------------------------------===//
// ArtifactCache - Cache for compiled artifacts
//===----------------------------------------------------------------------===//

/// Represents a cached compiled artifact.
struct CachedArtifact {
  /// Hash of the artifact content.
  ContentHash hash;

  /// Source hash this artifact was compiled from.
  ContentHash sourceHash;

  /// Path to the cached artifact file (if disk-based).
  std::string diskPath;

  /// In-memory artifact data (if memory-based).
  std::vector<uint8_t> memoryData;

  /// Timestamp of creation.
  std::chrono::system_clock::time_point created;

  /// Last access timestamp (for LRU eviction).
  std::chrono::system_clock::time_point lastAccessed;

  /// Size in bytes.
  size_t size = 0;

  /// Whether this artifact is stored on disk or in memory.
  bool onDisk = false;
};

/// Cache for compiled artifacts with LRU eviction.
class ArtifactCache {
public:
  /// Configuration for the artifact cache.
  struct Config {
    /// Maximum total cache size in bytes.
    size_t maxCacheSize;

    /// Maximum number of artifacts to cache.
    size_t maxArtifacts;

    /// Path to disk cache directory.
    std::string diskCachePath;

    /// Whether to use disk caching.
    bool useDiskCache;

    /// Whether to use memory caching.
    bool useMemoryCache;

    Config()
        : maxCacheSize(1024 * 1024 * 1024), // 1 GB
          maxArtifacts(10000), diskCachePath(""), useDiskCache(false),
          useMemoryCache(true) {}
  };

  ArtifactCache(Config config = Config());
  ~ArtifactCache();

  /// Store an artifact in the cache.
  bool store(const ContentHash &sourceHash, const std::vector<uint8_t> &data);

  /// Retrieve an artifact from the cache.
  bool retrieve(const ContentHash &sourceHash, std::vector<uint8_t> &data);

  /// Check if an artifact exists in the cache.
  bool contains(const ContentHash &sourceHash) const;

  /// Remove an artifact from the cache.
  void remove(const ContentHash &sourceHash);

  /// Clear the entire cache.
  void clear();

  /// Get cache statistics.
  struct Statistics {
    size_t hits = 0;
    size_t misses = 0;
    size_t evictions = 0;
    size_t currentSize = 0;
    size_t currentCount = 0;
  };

  Statistics getStatistics() const { return stats; }

  /// Evict artifacts to make room for new data.
  void evict(size_t bytesNeeded);

private:
  /// Load artifact from disk.
  bool loadFromDisk(const ContentHash &hash, std::vector<uint8_t> &data);

  /// Store artifact to disk.
  bool storeToDisk(const ContentHash &hash, const std::vector<uint8_t> &data);

  Config config;
  std::unordered_map<ContentHash, CachedArtifact, ContentHashHasher> cache;
  mutable std::mutex cacheMutex;
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// DependencyTracker - Track module dependencies
//===----------------------------------------------------------------------===//

/// Tracks dependencies between compilation units.
class DependencyTracker {
public:
  /// Add a compilation unit.
  void addUnit(const CompilationUnit &unit);

  /// Add a dependency from dependent to dependency.
  void addDependency(const ModuleId &dependent, const ModuleId &dependency);

  /// Remove a compilation unit.
  void removeUnit(const ModuleId &id);

  /// Get a compilation unit.
  CompilationUnit *getUnit(const ModuleId &id);
  const CompilationUnit *getUnit(const ModuleId &id) const;

  /// Get all units that depend on the given unit.
  llvm::SmallVector<ModuleId, 8> getDependents(const ModuleId &id) const;

  /// Get all units that the given unit depends on.
  llvm::SmallVector<ModuleId, 8> getDependencies(const ModuleId &id) const;

  /// Get all units affected by changes to the given unit.
  /// Uses transitive closure of dependents.
  llvm::SmallVector<ModuleId, 16>
  getAffectedUnits(const ModuleId &changedUnit) const;

  /// Get a topological ordering of all units.
  /// Returns empty vector if there's a cycle.
  std::vector<ModuleId> getTopologicalOrder() const;

  /// Check for dependency cycles.
  bool hasCycle() const;

  /// Get all compilation units.
  const std::unordered_map<ModuleId, CompilationUnit> &getUnits() const {
    return units;
  }

private:
  std::unordered_map<ModuleId, CompilationUnit> units;
};

//===----------------------------------------------------------------------===//
// ChangeDetector - Detect changes in source files
//===----------------------------------------------------------------------===//

/// Detects changes in source files using hashing.
class ChangeDetector {
public:
  /// Configuration for change detection.
  struct Config {
    /// Whether to use file modification time as a quick check.
    bool useModTime;

    /// Whether to always recompute hashes.
    bool alwaysHash;

    Config() : useModTime(true), alwaysHash(false) {}
  };

  ChangeDetector(Config config = Config()) : config(config) {}

  /// Check if a file has changed since last check.
  bool hasChanged(const std::string &path, ContentHash &newHash);

  /// Update the stored hash for a file.
  void updateHash(const std::string &path, const ContentHash &hash);

  /// Get the stored hash for a file.
  ContentHash getStoredHash(const std::string &path) const;

  /// Clear all stored hashes.
  void clear();

  /// Compute hash of a file.
  static ContentHash computeFileHash(const std::string &path);

  /// Compute hash of in-memory content.
  static ContentHash computeContentHash(llvm::StringRef content);

private:
  struct FileRecord {
    ContentHash hash;
    std::chrono::system_clock::time_point modTime;
    size_t fileSize;
  };

  Config config;
  std::unordered_map<std::string, FileRecord> fileRecords;
};

//===----------------------------------------------------------------------===//
// IncrementalCompiler - Main incremental compilation interface
//===----------------------------------------------------------------------===//

/// The main incremental compiler that coordinates change detection,
/// dependency tracking, and artifact caching.
class IncrementalCompiler {
public:
  /// Configuration for the incremental compiler.
  struct Config {
    /// Cache configuration.
    ArtifactCache::Config cacheConfig;

    /// Change detector configuration.
    ChangeDetector::Config changeConfig;

    /// Whether to enable parallel compilation.
    bool parallelCompilation;

    /// Number of parallel compilation threads.
    size_t parallelThreads;

    /// Whether to save/load state across sessions.
    bool persistState;

    /// Path for persistent state storage.
    std::string statePath;

    Config()
        : parallelCompilation(true), parallelThreads(0), persistState(false),
          statePath("") {}
  };

  /// Callback type for compiling a single unit.
  using CompileCallback =
      std::function<bool(const CompilationUnit &unit,
                         std::vector<uint8_t> &artifact)>;

  /// Callback type for linking multiple artifacts.
  using LinkCallback =
      std::function<bool(const std::vector<std::vector<uint8_t>> &artifacts,
                         std::vector<uint8_t> &linked)>;

  IncrementalCompiler(Config config = Config());
  ~IncrementalCompiler();

  //===------------------------------------------------------------------===//
  // Compilation Unit Management
  //===------------------------------------------------------------------===//

  /// Register a compilation unit.
  void registerUnit(const ModuleId &id, const std::string &name,
                    const ContentHash &sourceHash);

  /// Register a dependency between units.
  void registerDependency(const ModuleId &dependent, const ModuleId &dependency);

  /// Update the source hash for a unit (marks it dirty if changed).
  void updateSourceHash(const ModuleId &id, const ContentHash &newHash);

  /// Mark a unit as dirty (needs recompilation).
  void markDirty(const ModuleId &id);

  /// Get units that need recompilation.
  std::vector<ModuleId> getDirtyUnits() const;

  //===------------------------------------------------------------------===//
  // Compilation
  //===------------------------------------------------------------------===//

  /// Compile all dirty units.
  /// Returns true if all compilations succeeded.
  bool compileAll(CompileCallback compile);

  /// Compile a single unit.
  bool compileUnit(const ModuleId &id, CompileCallback compile);

  /// Link all compiled units.
  bool link(LinkCallback link, std::vector<uint8_t> &result);

  /// Perform incremental build (compile dirty + relink).
  bool incrementalBuild(CompileCallback compile, LinkCallback link,
                        std::vector<uint8_t> &result);

  //===------------------------------------------------------------------===//
  // Cache Management
  //===------------------------------------------------------------------===//

  /// Get the artifact cache.
  ArtifactCache &getCache() { return cache; }

  /// Clear all cached artifacts.
  void clearCache();

  //===------------------------------------------------------------------===//
  // State Persistence
  //===------------------------------------------------------------------===//

  /// Save compiler state to disk.
  bool saveState(const std::string &path);

  /// Load compiler state from disk.
  bool loadState(const std::string &path);

  //===------------------------------------------------------------------===//
  // Statistics
  //===------------------------------------------------------------------===//

  struct Statistics {
    size_t unitsRegistered = 0;
    size_t unitsCompiled = 0;
    size_t unitsCached = 0;
    size_t compilationTime = 0;
    size_t linkTime = 0;
    size_t totalTime = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Print incremental compilation report.
  void printReport(llvm::raw_ostream &os) const;

private:
  /// Compile units in parallel.
  bool compileParallel(const std::vector<ModuleId> &units,
                       CompileCallback compile);

  /// Compile units sequentially.
  bool compileSequential(const std::vector<ModuleId> &units,
                         CompileCallback compile);

  Config config;
  DependencyTracker dependencies;
  ChangeDetector changeDetector;
  ArtifactCache cache;
  Statistics stats;

  /// Artifacts for each unit (after compilation).
  std::unordered_map<ModuleId, std::vector<uint8_t>> artifacts;
};

//===----------------------------------------------------------------------===//
// ModuleHasher - Compute hashes for MLIR modules
//===----------------------------------------------------------------------===//

/// Computes content hashes for MLIR modules and operations.
class ModuleHasher {
public:
  /// Hash an operation and its body.
  static ContentHash hashOperation(void *op);

  /// Hash a region.
  static ContentHash hashRegion(void *region);

  /// Hash a module by name.
  static ContentHash hashModule(const std::string &moduleName, void *module);

  /// Hash multiple operations and combine.
  static ContentHash hashOperations(const std::vector<void *> &ops);
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_INCREMENTALCOMPILER_H
