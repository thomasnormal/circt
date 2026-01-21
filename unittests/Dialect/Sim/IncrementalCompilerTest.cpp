//===- IncrementalCompilerTest.cpp - IncrementalCompiler unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/IncrementalCompiler.h"
#include "gtest/gtest.h"

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// ContentHash Tests
//===----------------------------------------------------------------------===//

TEST(ContentHashTest, DefaultConstruction) {
  ContentHash hash;
  EXPECT_TRUE(hash.isZero());
  EXPECT_EQ(hash.high, 0u);
  EXPECT_EQ(hash.low, 0u);
}

TEST(ContentHashTest, FromString) {
  ContentHash hash1 = ContentHash::fromString("hello");
  ContentHash hash2 = ContentHash::fromString("hello");
  ContentHash hash3 = ContentHash::fromString("world");

  EXPECT_FALSE(hash1.isZero());
  EXPECT_EQ(hash1, hash2);
  EXPECT_NE(hash1, hash3);
}

TEST(ContentHashTest, FromData) {
  uint8_t data1[] = {1, 2, 3, 4};
  uint8_t data2[] = {1, 2, 3, 4};
  uint8_t data3[] = {5, 6, 7, 8};

  ContentHash hash1 = ContentHash::fromData(data1, sizeof(data1));
  ContentHash hash2 = ContentHash::fromData(data2, sizeof(data2));
  ContentHash hash3 = ContentHash::fromData(data3, sizeof(data3));

  EXPECT_EQ(hash1, hash2);
  EXPECT_NE(hash1, hash3);
}

TEST(ContentHashTest, Combine) {
  ContentHash hash1 = ContentHash::fromString("hello");
  ContentHash hash2 = ContentHash::fromString("world");
  ContentHash combined = hash1.combine(hash2);

  EXPECT_FALSE(combined.isZero());
  EXPECT_NE(combined, hash1);
  EXPECT_NE(combined, hash2);

  // Combining should be deterministic
  ContentHash combined2 = hash1.combine(hash2);
  EXPECT_EQ(combined, combined2);
}

TEST(ContentHashTest, HexStringConversion) {
  ContentHash original = ContentHash::fromString("test");
  std::string hexStr = original.toHexString();

  EXPECT_EQ(hexStr.length(), 32u);

  ContentHash parsed = ContentHash::fromHexString(hexStr);
  EXPECT_EQ(original, parsed);
}

//===----------------------------------------------------------------------===//
// ArtifactCache Tests
//===----------------------------------------------------------------------===//

TEST(ArtifactCacheTest, StoreAndRetrieve) {
  ArtifactCache::Config config;
  config.useMemoryCache = true;
  ArtifactCache cache(config);

  ContentHash hash = ContentHash::fromString("test");
  std::vector<uint8_t> data = {1, 2, 3, 4, 5};

  EXPECT_TRUE(cache.store(hash, data));
  EXPECT_TRUE(cache.contains(hash));

  std::vector<uint8_t> retrieved;
  EXPECT_TRUE(cache.retrieve(hash, retrieved));
  EXPECT_EQ(data, retrieved);
}

TEST(ArtifactCacheTest, Missing) {
  ArtifactCache cache;

  ContentHash hash = ContentHash::fromString("nonexistent");
  EXPECT_FALSE(cache.contains(hash));

  std::vector<uint8_t> data;
  EXPECT_FALSE(cache.retrieve(hash, data));
}

TEST(ArtifactCacheTest, Remove) {
  ArtifactCache cache;

  ContentHash hash = ContentHash::fromString("test");
  std::vector<uint8_t> data = {1, 2, 3};

  cache.store(hash, data);
  EXPECT_TRUE(cache.contains(hash));

  cache.remove(hash);
  EXPECT_FALSE(cache.contains(hash));
}

TEST(ArtifactCacheTest, Clear) {
  ArtifactCache cache;

  ContentHash hash1 = ContentHash::fromString("test1");
  ContentHash hash2 = ContentHash::fromString("test2");
  std::vector<uint8_t> data = {1, 2, 3};

  cache.store(hash1, data);
  cache.store(hash2, data);

  cache.clear();

  EXPECT_FALSE(cache.contains(hash1));
  EXPECT_FALSE(cache.contains(hash2));
}

TEST(ArtifactCacheTest, Statistics) {
  ArtifactCache cache;

  ContentHash hash = ContentHash::fromString("test");
  std::vector<uint8_t> data = {1, 2, 3};

  cache.store(hash, data);

  std::vector<uint8_t> retrieved;
  cache.retrieve(hash, retrieved);
  cache.retrieve(hash, retrieved);

  ContentHash missing = ContentHash::fromString("missing");
  cache.retrieve(missing, retrieved);

  auto stats = cache.getStatistics();
  EXPECT_EQ(stats.hits, 2u);
  EXPECT_EQ(stats.misses, 1u);
  EXPECT_EQ(stats.currentCount, 1u);
}

//===----------------------------------------------------------------------===//
// DependencyTracker Tests
//===----------------------------------------------------------------------===//

TEST(DependencyTrackerTest, AddUnit) {
  DependencyTracker tracker;

  CompilationUnit unit("mod1", "Module 1");
  tracker.addUnit(unit);

  CompilationUnit *retrieved = tracker.getUnit("mod1");
  EXPECT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved->name, "Module 1");
}

TEST(DependencyTrackerTest, AddDependency) {
  DependencyTracker tracker;

  CompilationUnit unit1("mod1", "Module 1");
  CompilationUnit unit2("mod2", "Module 2");
  tracker.addUnit(unit1);
  tracker.addUnit(unit2);

  tracker.addDependency("mod2", "mod1"); // mod2 depends on mod1

  auto deps = tracker.getDependencies("mod2");
  EXPECT_EQ(deps.size(), 1u);
  EXPECT_EQ(deps[0], "mod1");

  auto dependents = tracker.getDependents("mod1");
  EXPECT_EQ(dependents.size(), 1u);
  EXPECT_EQ(dependents[0], "mod2");
}

TEST(DependencyTrackerTest, GetAffectedUnits) {
  DependencyTracker tracker;

  // Create a chain: mod1 <- mod2 <- mod3
  CompilationUnit unit1("mod1", "Module 1");
  CompilationUnit unit2("mod2", "Module 2");
  CompilationUnit unit3("mod3", "Module 3");
  tracker.addUnit(unit1);
  tracker.addUnit(unit2);
  tracker.addUnit(unit3);

  tracker.addDependency("mod2", "mod1");
  tracker.addDependency("mod3", "mod2");

  auto affected = tracker.getAffectedUnits("mod1");
  EXPECT_EQ(affected.size(), 2u);
  // Both mod2 and mod3 should be affected
}

TEST(DependencyTrackerTest, TopologicalOrder) {
  DependencyTracker tracker;

  CompilationUnit unit1("mod1", "Module 1");
  CompilationUnit unit2("mod2", "Module 2");
  CompilationUnit unit3("mod3", "Module 3");
  tracker.addUnit(unit1);
  tracker.addUnit(unit2);
  tracker.addUnit(unit3);

  tracker.addDependency("mod2", "mod1");
  tracker.addDependency("mod3", "mod2");

  auto order = tracker.getTopologicalOrder();
  EXPECT_EQ(order.size(), 3u);

  // mod1 should come before mod2, mod2 before mod3
  auto it1 = std::find(order.begin(), order.end(), "mod1");
  auto it2 = std::find(order.begin(), order.end(), "mod2");
  auto it3 = std::find(order.begin(), order.end(), "mod3");

  EXPECT_LT(it1, it2);
  EXPECT_LT(it2, it3);
}

TEST(DependencyTrackerTest, RemoveUnit) {
  DependencyTracker tracker;

  CompilationUnit unit1("mod1", "Module 1");
  CompilationUnit unit2("mod2", "Module 2");
  tracker.addUnit(unit1);
  tracker.addUnit(unit2);

  tracker.addDependency("mod2", "mod1");

  tracker.removeUnit("mod1");

  EXPECT_EQ(tracker.getUnit("mod1"), nullptr);
  EXPECT_EQ(tracker.getDependencies("mod2").size(), 0u);
}

//===----------------------------------------------------------------------===//
// ChangeDetector Tests
//===----------------------------------------------------------------------===//

TEST(ChangeDetectorTest, ComputeContentHash) {
  ContentHash hash1 = ChangeDetector::computeContentHash("hello world");
  ContentHash hash2 = ChangeDetector::computeContentHash("hello world");
  ContentHash hash3 = ChangeDetector::computeContentHash("different");

  EXPECT_EQ(hash1, hash2);
  EXPECT_NE(hash1, hash3);
}

TEST(ChangeDetectorTest, UpdateAndGetHash) {
  ChangeDetector detector;

  ContentHash hash1 = ContentHash::fromString("content1");
  detector.updateHash("/path/to/file", hash1);

  ContentHash stored = detector.getStoredHash("/path/to/file");
  EXPECT_EQ(stored, hash1);
}

TEST(ChangeDetectorTest, Clear) {
  ChangeDetector detector;

  ContentHash hash = ContentHash::fromString("content");
  detector.updateHash("/path/to/file", hash);
  detector.clear();

  ContentHash stored = detector.getStoredHash("/path/to/file");
  EXPECT_TRUE(stored.isZero());
}

//===----------------------------------------------------------------------===//
// IncrementalCompiler Tests
//===----------------------------------------------------------------------===//

TEST(IncrementalCompilerTest, RegisterUnit) {
  IncrementalCompiler compiler;

  ContentHash hash = ContentHash::fromString("source code");
  compiler.registerUnit("mod1", "Module 1", hash);

  auto dirtyUnits = compiler.getDirtyUnits();
  EXPECT_EQ(dirtyUnits.size(), 1u);
  EXPECT_EQ(dirtyUnits[0], "mod1");
}

TEST(IncrementalCompilerTest, RegisterDependency) {
  IncrementalCompiler compiler;

  ContentHash hash1 = ContentHash::fromString("source1");
  ContentHash hash2 = ContentHash::fromString("source2");

  compiler.registerUnit("mod1", "Module 1", hash1);
  compiler.registerUnit("mod2", "Module 2", hash2);
  compiler.registerDependency("mod2", "mod1");

  auto dirtyUnits = compiler.getDirtyUnits();
  EXPECT_EQ(dirtyUnits.size(), 2u);
}

TEST(IncrementalCompilerTest, UpdateSourceHash) {
  IncrementalCompiler compiler;

  ContentHash hash1 = ContentHash::fromString("original");
  compiler.registerUnit("mod1", "Module 1", hash1);

  // Compile to mark as clean
  compiler.compileAll([](const CompilationUnit &, std::vector<uint8_t> &artifact) {
    artifact = {1, 2, 3};
    return true;
  });

  EXPECT_EQ(compiler.getDirtyUnits().size(), 0u);

  // Update hash should mark as dirty
  ContentHash hash2 = ContentHash::fromString("modified");
  compiler.updateSourceHash("mod1", hash2);

  EXPECT_EQ(compiler.getDirtyUnits().size(), 1u);
}

TEST(IncrementalCompilerTest, CompileUnit) {
  IncrementalCompiler compiler;

  ContentHash hash = ContentHash::fromString("source");
  compiler.registerUnit("mod1", "Module 1", hash);

  bool compiled = false;
  compiler.compileAll([&compiled](const CompilationUnit &unit,
                                   std::vector<uint8_t> &artifact) {
    compiled = true;
    artifact = {0x00, 0x01, 0x02};
    return true;
  });

  EXPECT_TRUE(compiled);
  EXPECT_EQ(compiler.getDirtyUnits().size(), 0u);

  // Second compile should hit cache
  compiled = false;
  compiler.compileAll([&compiled](const CompilationUnit &unit,
                                   std::vector<uint8_t> &artifact) {
    compiled = true;
    artifact = {0x00, 0x01, 0x02};
    return true;
  });

  EXPECT_FALSE(compiled); // Should not recompile
}

TEST(IncrementalCompilerTest, CompileFailure) {
  IncrementalCompiler compiler;

  ContentHash hash = ContentHash::fromString("source");
  compiler.registerUnit("mod1", "Module 1", hash);

  bool result = compiler.compileAll([](const CompilationUnit &,
                                        std::vector<uint8_t> &) {
    return false; // Compilation fails
  });

  EXPECT_FALSE(result);
}

TEST(IncrementalCompilerTest, IncrementalBuild) {
  IncrementalCompiler compiler;

  ContentHash hash1 = ContentHash::fromString("source1");
  ContentHash hash2 = ContentHash::fromString("source2");

  compiler.registerUnit("mod1", "Module 1", hash1);
  compiler.registerUnit("mod2", "Module 2", hash2);
  compiler.registerDependency("mod2", "mod1");

  std::vector<uint8_t> result;
  bool success = compiler.incrementalBuild(
      [](const CompilationUnit &, std::vector<uint8_t> &artifact) {
        artifact = {0xAB, 0xCD};
        return true;
      },
      [](const std::vector<std::vector<uint8_t>> &artifacts,
         std::vector<uint8_t> &linked) {
        linked = {0xEF};
        for (const auto &a : artifacts) {
          linked.insert(linked.end(), a.begin(), a.end());
        }
        return true;
      },
      result);

  EXPECT_TRUE(success);
  EXPECT_FALSE(result.empty());
}

TEST(IncrementalCompilerTest, Statistics) {
  IncrementalCompiler compiler;

  ContentHash hash = ContentHash::fromString("source");
  compiler.registerUnit("mod1", "Module 1", hash);

  compiler.compileAll([](const CompilationUnit &, std::vector<uint8_t> &artifact) {
    artifact = {1, 2, 3};
    return true;
  });

  const auto &stats = compiler.getStatistics();
  EXPECT_EQ(stats.unitsRegistered, 1u);
  EXPECT_EQ(stats.unitsCompiled, 1u);
}
