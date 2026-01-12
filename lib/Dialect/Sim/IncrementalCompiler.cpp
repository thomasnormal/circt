//===- IncrementalCompiler.cpp - Incremental compilation support ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the incremental compilation infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/IncrementalCompiler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <queue>
#include <thread>

#define DEBUG_TYPE "sim-incremental-compiler"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// ContentHash Implementation
//===----------------------------------------------------------------------===//

// Simple xxHash-like implementation for portability
static uint64_t xxhash64(const void *data, size_t len, uint64_t seed) {
  const uint64_t PRIME1 = 11400714785074694791ULL;
  const uint64_t PRIME2 = 14029467366897019727ULL;
  const uint64_t PRIME3 = 1609587929392839161ULL;
  const uint64_t PRIME4 = 9650029242287828579ULL;
  const uint64_t PRIME5 = 2870177450012600261ULL;

  const uint8_t *p = static_cast<const uint8_t *>(data);
  const uint8_t *end = p + len;
  uint64_t h64;

  if (len >= 32) {
    uint64_t v1 = seed + PRIME1 + PRIME2;
    uint64_t v2 = seed + PRIME2;
    uint64_t v3 = seed;
    uint64_t v4 = seed - PRIME1;

    while (p + 32 <= end) {
      uint64_t k1, k2, k3, k4;
      memcpy(&k1, p, 8);
      memcpy(&k2, p + 8, 8);
      memcpy(&k3, p + 16, 8);
      memcpy(&k4, p + 24, 8);

      v1 += k1 * PRIME2;
      v1 = (v1 << 31) | (v1 >> 33);
      v1 *= PRIME1;

      v2 += k2 * PRIME2;
      v2 = (v2 << 31) | (v2 >> 33);
      v2 *= PRIME1;

      v3 += k3 * PRIME2;
      v3 = (v3 << 31) | (v3 >> 33);
      v3 *= PRIME1;

      v4 += k4 * PRIME2;
      v4 = (v4 << 31) | (v4 >> 33);
      v4 *= PRIME1;

      p += 32;
    }

    h64 = ((v1 << 1) | (v1 >> 63)) + ((v2 << 7) | (v2 >> 57)) +
          ((v3 << 12) | (v3 >> 52)) + ((v4 << 18) | (v4 >> 46));

    h64 ^= ((v1 * PRIME2) << 31 | (v1 * PRIME2) >> 33) * PRIME1;
    h64 = h64 * PRIME1 + PRIME4;

    h64 ^= ((v2 * PRIME2) << 31 | (v2 * PRIME2) >> 33) * PRIME1;
    h64 = h64 * PRIME1 + PRIME4;

    h64 ^= ((v3 * PRIME2) << 31 | (v3 * PRIME2) >> 33) * PRIME1;
    h64 = h64 * PRIME1 + PRIME4;

    h64 ^= ((v4 * PRIME2) << 31 | (v4 * PRIME2) >> 33) * PRIME1;
    h64 = h64 * PRIME1 + PRIME4;
  } else {
    h64 = seed + PRIME5;
  }

  h64 += len;

  while (p + 8 <= end) {
    uint64_t k;
    memcpy(&k, p, 8);
    k *= PRIME2;
    k = (k << 31) | (k >> 33);
    k *= PRIME1;
    h64 ^= k;
    h64 = ((h64 << 27) | (h64 >> 37)) * PRIME1 + PRIME4;
    p += 8;
  }

  while (p + 4 <= end) {
    uint32_t k;
    memcpy(&k, p, 4);
    h64 ^= k * PRIME1;
    h64 = ((h64 << 23) | (h64 >> 41)) * PRIME2 + PRIME3;
    p += 4;
  }

  while (p < end) {
    h64 ^= (*p) * PRIME5;
    h64 = ((h64 << 11) | (h64 >> 53)) * PRIME1;
    p++;
  }

  h64 ^= h64 >> 33;
  h64 *= PRIME2;
  h64 ^= h64 >> 29;
  h64 *= PRIME3;
  h64 ^= h64 >> 32;

  return h64;
}

ContentHash ContentHash::fromString(llvm::StringRef str) {
  return fromData(str.data(), str.size());
}

ContentHash ContentHash::fromData(const void *data, size_t size) {
  uint64_t h1 = xxhash64(data, size, 0);
  uint64_t h2 = xxhash64(data, size, h1);
  return ContentHash(h1, h2);
}

ContentHash ContentHash::combine(const ContentHash &other) const {
  uint64_t data[4] = {high, low, other.high, other.low};
  return fromData(data, sizeof(data));
}

std::string ContentHash::toHexString() const {
  char buffer[33];
  snprintf(buffer, sizeof(buffer), "%016llx%016llx",
           static_cast<unsigned long long>(high),
           static_cast<unsigned long long>(low));
  return std::string(buffer);
}

ContentHash ContentHash::fromHexString(llvm::StringRef hex) {
  if (hex.size() != 32)
    return ContentHash();

  uint64_t h = 0, l = 0;
  for (int i = 0; i < 16; ++i) {
    h = (h << 4) | llvm::hexDigitValue(hex[i]);
  }
  for (int i = 16; i < 32; ++i) {
    l = (l << 4) | llvm::hexDigitValue(hex[i]);
  }
  return ContentHash(h, l);
}

//===----------------------------------------------------------------------===//
// ArtifactCache Implementation
//===----------------------------------------------------------------------===//

ArtifactCache::ArtifactCache(Config config) : config(config) {}

ArtifactCache::~ArtifactCache() = default;

bool ArtifactCache::store(const ContentHash &sourceHash,
                          const std::vector<uint8_t> &data) {
  std::lock_guard<std::mutex> lock(cacheMutex);

  // Check if we need to evict
  if (stats.currentSize + data.size() > config.maxCacheSize) {
    evict(data.size());
  }

  if (cache.size() >= config.maxArtifacts) {
    evict(0); // Evict at least one item
  }

  CachedArtifact artifact;
  artifact.hash = ContentHash::fromData(data.data(), data.size());
  artifact.sourceHash = sourceHash;
  artifact.created = std::chrono::system_clock::now();
  artifact.lastAccessed = artifact.created;
  artifact.size = data.size();

  if (config.useMemoryCache) {
    artifact.memoryData = data;
    artifact.onDisk = false;
  } else if (config.useDiskCache) {
    if (!storeToDisk(sourceHash, data)) {
      return false;
    }
    artifact.onDisk = true;
    artifact.diskPath = config.diskCachePath + "/" + sourceHash.toHexString();
  }

  cache[sourceHash] = std::move(artifact);
  stats.currentSize += data.size();
  stats.currentCount++;

  LLVM_DEBUG(llvm::dbgs() << "Cached artifact for " << sourceHash.toHexString()
                          << " (" << data.size() << " bytes)\n");
  return true;
}

bool ArtifactCache::retrieve(const ContentHash &sourceHash,
                             std::vector<uint8_t> &data) {
  std::lock_guard<std::mutex> lock(cacheMutex);

  auto it = cache.find(sourceHash);
  if (it == cache.end()) {
    stats.misses++;
    return false;
  }

  CachedArtifact &artifact = it->second;
  artifact.lastAccessed = std::chrono::system_clock::now();

  if (artifact.onDisk) {
    if (!loadFromDisk(sourceHash, data)) {
      stats.misses++;
      return false;
    }
  } else {
    data = artifact.memoryData;
  }

  stats.hits++;
  LLVM_DEBUG(llvm::dbgs() << "Cache hit for " << sourceHash.toHexString()
                          << "\n");
  return true;
}

bool ArtifactCache::contains(const ContentHash &sourceHash) const {
  std::lock_guard<std::mutex> lock(cacheMutex);
  return cache.find(sourceHash) != cache.end();
}

void ArtifactCache::remove(const ContentHash &sourceHash) {
  std::lock_guard<std::mutex> lock(cacheMutex);
  auto it = cache.find(sourceHash);
  if (it != cache.end()) {
    stats.currentSize -= it->second.size;
    stats.currentCount--;
    cache.erase(it);
  }
}

void ArtifactCache::clear() {
  std::lock_guard<std::mutex> lock(cacheMutex);
  cache.clear();
  stats.currentSize = 0;
  stats.currentCount = 0;
}

void ArtifactCache::evict(size_t bytesNeeded) {
  // LRU eviction
  std::vector<std::pair<std::chrono::system_clock::time_point, ContentHash>>
      accessTimes;
  for (const auto &kv : cache) {
    accessTimes.push_back({kv.second.lastAccessed, kv.first});
  }

  // Sort by access time (oldest first)
  std::sort(accessTimes.begin(), accessTimes.end());

  size_t evicted = 0;
  for (const auto &[time, hash] : accessTimes) {
    if (evicted >= bytesNeeded && cache.size() < config.maxArtifacts)
      break;

    auto it = cache.find(hash);
    if (it != cache.end()) {
      evicted += it->second.size;
      stats.currentSize -= it->second.size;
      stats.currentCount--;
      stats.evictions++;
      cache.erase(it);
    }
  }
}

bool ArtifactCache::loadFromDisk(const ContentHash &hash,
                                 std::vector<uint8_t> &data) {
  std::string path = config.diskCachePath + "/" + hash.toHexString();
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return false;

  auto &buffer = *bufferOrErr;
  data.assign(buffer->getBufferStart(), buffer->getBufferEnd());
  return true;
}

bool ArtifactCache::storeToDisk(const ContentHash &hash,
                                const std::vector<uint8_t> &data) {
  std::string path = config.diskCachePath + "/" + hash.toHexString();

  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec);
  if (ec)
    return false;

  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  return !file.has_error();
}

//===----------------------------------------------------------------------===//
// DependencyTracker Implementation
//===----------------------------------------------------------------------===//

void DependencyTracker::addUnit(const CompilationUnit &unit) {
  units[unit.id] = unit;
}

void DependencyTracker::addDependency(const ModuleId &dependent,
                                      const ModuleId &dependency) {
  auto depIt = units.find(dependent);
  auto depsIt = units.find(dependency);

  if (depIt != units.end()) {
    auto &deps = depIt->second.dependencies;
    if (std::find(deps.begin(), deps.end(), dependency) == deps.end()) {
      deps.push_back(dependency);
    }
  }

  if (depsIt != units.end()) {
    auto &dependents = depsIt->second.dependents;
    if (std::find(dependents.begin(), dependents.end(), dependent) ==
        dependents.end()) {
      dependents.push_back(dependent);
    }
  }
}

void DependencyTracker::removeUnit(const ModuleId &id) {
  // Remove from dependents' dependency lists
  auto it = units.find(id);
  if (it != units.end()) {
    for (const auto &dep : it->second.dependencies) {
      auto depIt = units.find(dep);
      if (depIt != units.end()) {
        auto &dependents = depIt->second.dependents;
        dependents.erase(
            std::remove(dependents.begin(), dependents.end(), id),
            dependents.end());
      }
    }
    for (const auto &dependent : it->second.dependents) {
      auto depIt = units.find(dependent);
      if (depIt != units.end()) {
        auto &deps = depIt->second.dependencies;
        deps.erase(std::remove(deps.begin(), deps.end(), id), deps.end());
      }
    }
  }

  units.erase(id);
}

CompilationUnit *DependencyTracker::getUnit(const ModuleId &id) {
  auto it = units.find(id);
  return it != units.end() ? &it->second : nullptr;
}

const CompilationUnit *DependencyTracker::getUnit(const ModuleId &id) const {
  auto it = units.find(id);
  return it != units.end() ? &it->second : nullptr;
}

llvm::SmallVector<ModuleId, 8>
DependencyTracker::getDependents(const ModuleId &id) const {
  const auto *unit = getUnit(id);
  return unit ? unit->dependents : llvm::SmallVector<ModuleId, 8>();
}

llvm::SmallVector<ModuleId, 8>
DependencyTracker::getDependencies(const ModuleId &id) const {
  const auto *unit = getUnit(id);
  return unit ? unit->dependencies : llvm::SmallVector<ModuleId, 8>();
}

llvm::SmallVector<ModuleId, 16>
DependencyTracker::getAffectedUnits(const ModuleId &changedUnit) const {
  llvm::SmallVector<ModuleId, 16> affected;
  std::unordered_map<ModuleId, bool> visited;

  std::queue<ModuleId> queue;
  queue.push(changedUnit);

  while (!queue.empty()) {
    ModuleId current = queue.front();
    queue.pop();

    if (visited[current])
      continue;
    visited[current] = true;

    if (current != changedUnit)
      affected.push_back(current);

    for (const auto &dependent : getDependents(current)) {
      if (!visited[dependent]) {
        queue.push(dependent);
      }
    }
  }

  return affected;
}

std::vector<ModuleId> DependencyTracker::getTopologicalOrder() const {
  std::vector<ModuleId> order;
  std::unordered_map<ModuleId, int> inDegree;
  std::queue<ModuleId> ready;

  // Calculate in-degrees
  for (const auto &kv : units) {
    inDegree[kv.first] = kv.second.dependencies.size();
    if (inDegree[kv.first] == 0)
      ready.push(kv.first);
  }

  while (!ready.empty()) {
    ModuleId current = ready.front();
    ready.pop();
    order.push_back(current);

    for (const auto &dependent : getDependents(current)) {
      if (--inDegree[dependent] == 0) {
        ready.push(dependent);
      }
    }
  }

  // If order size != units size, there's a cycle
  if (order.size() != units.size())
    return {};

  return order;
}

bool DependencyTracker::hasCycle() const {
  return getTopologicalOrder().empty() && !units.empty();
}

//===----------------------------------------------------------------------===//
// ChangeDetector Implementation
//===----------------------------------------------------------------------===//

bool ChangeDetector::hasChanged(const std::string &path, ContentHash &newHash) {
  newHash = computeFileHash(path);

  auto it = fileRecords.find(path);
  if (it == fileRecords.end()) {
    // New file
    return true;
  }

  if (newHash != it->second.hash) {
    return true;
  }

  return false;
}

void ChangeDetector::updateHash(const std::string &path,
                                const ContentHash &hash) {
  FileRecord record;
  record.hash = hash;
  record.modTime = std::chrono::system_clock::now();
  fileRecords[path] = record;
}

ContentHash ChangeDetector::getStoredHash(const std::string &path) const {
  auto it = fileRecords.find(path);
  return it != fileRecords.end() ? it->second.hash : ContentHash();
}

void ChangeDetector::clear() { fileRecords.clear(); }

ContentHash ChangeDetector::computeFileHash(const std::string &path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return ContentHash();

  auto &buffer = *bufferOrErr;
  return ContentHash::fromData(buffer->getBufferStart(),
                               buffer->getBufferSize());
}

ContentHash ChangeDetector::computeContentHash(llvm::StringRef content) {
  return ContentHash::fromString(content);
}

//===----------------------------------------------------------------------===//
// IncrementalCompiler Implementation
//===----------------------------------------------------------------------===//

IncrementalCompiler::IncrementalCompiler(Config config)
    : config(config), changeDetector(config.changeConfig),
      cache(config.cacheConfig) {
  if (config.parallelThreads == 0) {
    config.parallelThreads = std::thread::hardware_concurrency();
    if (config.parallelThreads == 0)
      config.parallelThreads = 4;
  }
}

IncrementalCompiler::~IncrementalCompiler() = default;

void IncrementalCompiler::registerUnit(const ModuleId &id,
                                       const std::string &name,
                                       const ContentHash &sourceHash) {
  CompilationUnit unit(id, name);
  unit.sourceHash = sourceHash;
  unit.dirty = true;
  dependencies.addUnit(unit);
  stats.unitsRegistered++;
}

void IncrementalCompiler::registerDependency(const ModuleId &dependent,
                                             const ModuleId &dependency) {
  dependencies.addDependency(dependent, dependency);
}

void IncrementalCompiler::updateSourceHash(const ModuleId &id,
                                           const ContentHash &newHash) {
  CompilationUnit *unit = dependencies.getUnit(id);
  if (!unit)
    return;

  if (unit->sourceHash != newHash) {
    unit->sourceHash = newHash;
    unit->dirty = true;

    // Mark all dependents as dirty
    for (const auto &dependent : dependencies.getAffectedUnits(id)) {
      CompilationUnit *depUnit = dependencies.getUnit(dependent);
      if (depUnit) {
        depUnit->dirty = true;
      }
    }
  }
}

void IncrementalCompiler::markDirty(const ModuleId &id) {
  CompilationUnit *unit = dependencies.getUnit(id);
  if (unit) {
    unit->dirty = true;
  }
}

std::vector<ModuleId> IncrementalCompiler::getDirtyUnits() const {
  std::vector<ModuleId> dirty;
  for (const auto &kv : dependencies.getUnits()) {
    if (kv.second.dirty) {
      dirty.push_back(kv.first);
    }
  }
  return dirty;
}

bool IncrementalCompiler::compileAll(CompileCallback compile) {
  std::vector<ModuleId> dirtyUnits = getDirtyUnits();

  if (dirtyUnits.empty())
    return true;

  LLVM_DEBUG(llvm::dbgs() << "Compiling " << dirtyUnits.size()
                          << " dirty units\n");

  auto startTime = std::chrono::high_resolution_clock::now();

  bool success;
  if (config.parallelCompilation && dirtyUnits.size() > 1) {
    success = compileParallel(dirtyUnits, compile);
  } else {
    success = compileSequential(dirtyUnits, compile);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  stats.compilationTime += duration.count();

  return success;
}

bool IncrementalCompiler::compileUnit(const ModuleId &id,
                                      CompileCallback compile) {
  CompilationUnit *unit = dependencies.getUnit(id);
  if (!unit)
    return false;

  // Check cache first
  std::vector<uint8_t> artifact;
  if (cache.retrieve(unit->sourceHash, artifact)) {
    artifacts[id] = std::move(artifact);
    unit->dirty = false;
    unit->valid = true;
    stats.unitsCached++;
    LLVM_DEBUG(llvm::dbgs() << "Using cached artifact for " << id << "\n");
    return true;
  }

  // Compile
  LLVM_DEBUG(llvm::dbgs() << "Compiling unit " << id << "\n");
  if (!compile(*unit, artifact)) {
    unit->valid = false;
    return false;
  }

  // Store in cache
  cache.store(unit->sourceHash, artifact);

  // Update unit state
  artifacts[id] = std::move(artifact);
  unit->artifactHash = ContentHash::fromData(artifacts[id].data(),
                                             artifacts[id].size());
  unit->lastCompiled = std::chrono::system_clock::now();
  unit->dirty = false;
  unit->valid = true;
  stats.unitsCompiled++;

  return true;
}

bool IncrementalCompiler::compileSequential(const std::vector<ModuleId> &units,
                                            CompileCallback compile) {
  // Get topological order
  std::vector<ModuleId> order = dependencies.getTopologicalOrder();
  if (order.empty() && !dependencies.getUnits().empty()) {
    // Cycle detected
    LLVM_DEBUG(llvm::dbgs() << "Dependency cycle detected!\n");
    return false;
  }

  // Filter to only dirty units in topological order
  std::vector<ModuleId> filtered;
  for (const auto &id : order) {
    if (std::find(units.begin(), units.end(), id) != units.end()) {
      filtered.push_back(id);
    }
  }

  for (const auto &id : filtered) {
    if (!compileUnit(id, compile)) {
      return false;
    }
  }

  return true;
}

bool IncrementalCompiler::compileParallel(const std::vector<ModuleId> &units,
                                          CompileCallback compile) {
  // Get topological order for level assignment
  std::unordered_map<ModuleId, int> levels;
  for (const auto &id : units) {
    levels[id] = 0;
  }

  // Calculate levels based on dependencies
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &id : units) {
      int maxDepLevel = -1;
      for (const auto &dep : dependencies.getDependencies(id)) {
        auto it = levels.find(dep);
        if (it != levels.end()) {
          maxDepLevel = std::max(maxDepLevel, it->second);
        }
      }
      int newLevel = maxDepLevel + 1;
      if (newLevel > levels[id]) {
        levels[id] = newLevel;
        changed = true;
      }
    }
  }

  // Group by level
  std::map<int, std::vector<ModuleId>> levelGroups;
  for (const auto &id : units) {
    levelGroups[levels[id]].push_back(id);
  }

  // Compile level by level
  for (const auto &[level, groupUnits] : levelGroups) {
    std::vector<std::thread> threads;
    std::atomic<bool> success{true};

    for (const auto &unitId : groupUnits) {
      if (threads.size() >= config.parallelThreads) {
        // Wait for a thread to finish
        for (auto &t : threads) {
          if (t.joinable())
            t.join();
        }
        threads.clear();
      }

      threads.emplace_back([this, unitId, &compile, &success]() {
        if (!compileUnit(unitId, compile)) {
          success.store(false);
        }
      });
    }

    // Wait for remaining threads
    for (auto &t : threads) {
      if (t.joinable())
        t.join();
    }

    if (!success.load())
      return false;
  }

  return true;
}

bool IncrementalCompiler::link(LinkCallback linkCallback,
                               std::vector<uint8_t> &result) {
  auto startTime = std::chrono::high_resolution_clock::now();

  // Gather all artifacts in topological order
  std::vector<std::vector<uint8_t>> allArtifacts;
  std::vector<ModuleId> order = dependencies.getTopologicalOrder();

  for (const auto &id : order) {
    auto it = artifacts.find(id);
    if (it != artifacts.end()) {
      allArtifacts.push_back(it->second);
    }
  }

  bool success = linkCallback(allArtifacts, result);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  stats.linkTime += duration.count();

  return success;
}

bool IncrementalCompiler::incrementalBuild(CompileCallback compile,
                                           LinkCallback linkCallback,
                                           std::vector<uint8_t> &result) {
  auto startTime = std::chrono::high_resolution_clock::now();

  if (!compileAll(compile)) {
    return false;
  }

  if (!link(linkCallback, result)) {
    return false;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  stats.totalTime = duration.count();

  return true;
}

void IncrementalCompiler::clearCache() { cache.clear(); }

bool IncrementalCompiler::saveState(const std::string &path) {
  std::error_code ec;
  llvm::raw_fd_ostream file(path, ec);
  if (ec)
    return false;

  // Simple format: one unit per line
  // id|name|sourceHash|artifactHash|dirty|valid
  for (const auto &kv : dependencies.getUnits()) {
    const auto &unit = kv.second;
    file << unit.id << "|" << unit.name << "|"
         << unit.sourceHash.toHexString() << "|"
         << unit.artifactHash.toHexString() << "|" << (unit.dirty ? "1" : "0")
         << "|" << (unit.valid ? "1" : "0") << "\n";
  }

  return !file.has_error();
}

bool IncrementalCompiler::loadState(const std::string &path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr)
    return false;

  llvm::StringRef content = (*bufferOrErr)->getBuffer();

  // Parse line by line
  llvm::SmallVector<llvm::StringRef, 0> lines;
  content.split(lines, '\n');

  for (const auto &line : lines) {
    if (line.empty())
      continue;

    llvm::SmallVector<llvm::StringRef, 6> parts;
    line.split(parts, '|');
    if (parts.size() < 6)
      continue;

    CompilationUnit unit;
    unit.id = parts[0].str();
    unit.name = parts[1].str();
    unit.sourceHash = ContentHash::fromHexString(parts[2]);
    unit.artifactHash = ContentHash::fromHexString(parts[3]);
    unit.dirty = parts[4] == "1";
    unit.valid = parts[5] == "1";

    dependencies.addUnit(unit);
  }

  return true;
}

void IncrementalCompiler::printReport(llvm::raw_ostream &os) const {
  os << "=== Incremental Compilation Report ===\n";
  os << "Units registered: " << stats.unitsRegistered << "\n";
  os << "Units compiled: " << stats.unitsCompiled << "\n";
  os << "Units from cache: " << stats.unitsCached << "\n";
  os << "Compilation time: " << stats.compilationTime << " ms\n";
  os << "Link time: " << stats.linkTime << " ms\n";
  os << "Total time: " << stats.totalTime << " ms\n";
  os << "\n";

  auto cacheStats = cache.getStatistics();
  os << "Cache statistics:\n";
  os << "  Hits: " << cacheStats.hits << "\n";
  os << "  Misses: " << cacheStats.misses << "\n";
  os << "  Evictions: " << cacheStats.evictions << "\n";
  os << "  Current size: " << cacheStats.currentSize << " bytes\n";
  os << "  Current count: " << cacheStats.currentCount << "\n";
}

//===----------------------------------------------------------------------===//
// ModuleHasher Implementation
//===----------------------------------------------------------------------===//

ContentHash ModuleHasher::hashOperation(void *op) {
  // Placeholder - actual implementation would use MLIR operation attributes
  return ContentHash::fromData(&op, sizeof(op));
}

ContentHash ModuleHasher::hashRegion(void *region) {
  return ContentHash::fromData(&region, sizeof(region));
}

ContentHash ModuleHasher::hashModule(const std::string &moduleName,
                                     void *module) {
  ContentHash nameHash = ContentHash::fromString(moduleName);
  ContentHash moduleHash = ContentHash::fromData(&module, sizeof(module));
  return nameHash.combine(moduleHash);
}

ContentHash ModuleHasher::hashOperations(const std::vector<void *> &ops) {
  ContentHash combined;
  for (void *op : ops) {
    combined = combined.combine(hashOperation(op));
  }
  return combined;
}
