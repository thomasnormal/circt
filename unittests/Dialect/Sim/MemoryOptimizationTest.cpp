//===- MemoryOptimizationTest.cpp - MemoryOptimization unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/MemoryOptimization.h"
#include "gtest/gtest.h"

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// MemoryRegion Tests
//===----------------------------------------------------------------------===//

TEST(MemoryRegionTest, Allocation) {
  MemoryRegion region(0, 1024, false);

  EXPECT_EQ(region.getId(), 0u);
  EXPECT_EQ(region.getSize(), 1024u);
  EXPECT_TRUE(region.isAllocated());
  EXPECT_FALSE(region.isMapped());
}

TEST(MemoryRegionTest, ReadWrite) {
  MemoryRegion region(0, 256, false);

  uint8_t writeData[] = {1, 2, 3, 4, 5};
  region.write(0, writeData, sizeof(writeData));

  uint8_t readData[5] = {0};
  region.read(0, readData, sizeof(readData));

  EXPECT_EQ(memcmp(writeData, readData, sizeof(writeData)), 0);
}

TEST(MemoryRegionTest, OffsetReadWrite) {
  MemoryRegion region(0, 256, false);

  uint32_t value = 0x12345678;
  region.write(100, &value, sizeof(value));

  uint32_t readValue = 0;
  region.read(100, &readValue, sizeof(readValue));

  EXPECT_EQ(value, readValue);
}

TEST(MemoryRegionTest, Zero) {
  MemoryRegion region(0, 16, false);

  uint8_t data[] = {0xFF, 0xFF, 0xFF, 0xFF};
  region.write(0, data, sizeof(data));

  region.zero();

  uint8_t readData[4] = {1, 1, 1, 1};
  region.read(0, readData, sizeof(readData));

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(readData[i], 0);
  }
}

TEST(MemoryRegionTest, Statistics) {
  MemoryRegion region(0, 100, false);

  uint8_t data = 42;
  region.write(0, &data, 1);
  region.write(10, &data, 1);
  region.read(0, &data, 1);

  const auto &stats = region.getStatistics();
  EXPECT_EQ(stats.writes, 2u);
  EXPECT_EQ(stats.reads, 1u);
  EXPECT_EQ(stats.totalSize, 100u);
}

//===----------------------------------------------------------------------===//
// SparseArray Tests
//===----------------------------------------------------------------------===//

TEST(SparseArrayTest, BasicOperations) {
  SparseArray<int, 0> array(1000);

  EXPECT_EQ(array.size(), 1000u);
  EXPECT_EQ(array.storedCount(), 0u);
  EXPECT_EQ(array.get(0), 0);
}

TEST(SparseArrayTest, SetAndGet) {
  SparseArray<int, 0> array(100);

  array.set(10, 42);
  array.set(50, 123);

  EXPECT_EQ(array.get(10), 42);
  EXPECT_EQ(array.get(50), 123);
  EXPECT_EQ(array.get(0), 0);  // Default value
  EXPECT_EQ(array.get(99), 0); // Default value
  EXPECT_EQ(array.storedCount(), 2u);
}

TEST(SparseArrayTest, DefaultValue) {
  SparseArray<int, -1> array(100);

  EXPECT_EQ(array.get(0), -1);

  array.set(5, 42);
  EXPECT_EQ(array.get(5), 42);

  // Setting to default value should remove
  array.set(5, -1);
  EXPECT_EQ(array.storedCount(), 0u);
}

TEST(SparseArrayTest, Has) {
  SparseArray<int> array(100);

  EXPECT_FALSE(array.has(10));

  array.set(10, 42);
  EXPECT_TRUE(array.has(10));

  array.set(10, 0); // Set to default
  EXPECT_FALSE(array.has(10));
}

TEST(SparseArrayTest, Clear) {
  SparseArray<int> array(100);

  array.set(10, 42);
  array.set(20, 123);

  EXPECT_EQ(array.storedCount(), 2u);

  array.clear();

  EXPECT_EQ(array.storedCount(), 0u);
  EXPECT_EQ(array.get(10), 0);
}

TEST(SparseArrayTest, CompressionRatio) {
  SparseArray<int> array(10000);

  // Only store a few values
  array.set(0, 1);
  array.set(9999, 2);

  double ratio = array.compressionRatio();
  EXPECT_GT(ratio, 1000.0); // Should have high compression
}

TEST(SparseArrayTest, Iteration) {
  SparseArray<int> array(100);

  array.set(5, 50);
  array.set(10, 100);
  array.set(20, 200);

  int sum = 0;
  for (const auto &kv : array) {
    sum += kv.second;
  }

  EXPECT_EQ(sum, 350);
}

//===----------------------------------------------------------------------===//
// CompressedBitVector Tests
//===----------------------------------------------------------------------===//

TEST(CompressedBitVectorTest, BasicOperations) {
  CompressedBitVector bv(100);

  EXPECT_EQ(bv.size(), 100u);
  EXPECT_EQ(bv.popcount(), 0u);
}

TEST(CompressedBitVectorTest, SetAndGet) {
  CompressedBitVector bv(100);

  bv.set(10, true);
  bv.set(20, true);
  bv.set(30, true);

  EXPECT_TRUE(bv.get(10));
  EXPECT_TRUE(bv.get(20));
  EXPECT_TRUE(bv.get(30));
  EXPECT_FALSE(bv.get(0));
  EXPECT_FALSE(bv.get(15));
  EXPECT_EQ(bv.popcount(), 3u);
}

TEST(CompressedBitVectorTest, ClearBit) {
  CompressedBitVector bv(100);

  bv.set(10, true);
  EXPECT_TRUE(bv.get(10));

  bv.set(10, false);
  EXPECT_FALSE(bv.get(10));
  EXPECT_EQ(bv.popcount(), 0u);
}

TEST(CompressedBitVectorTest, SetRange) {
  CompressedBitVector bv(100);

  bv.setRange(10, 5, true); // Set bits 10-14

  EXPECT_FALSE(bv.get(9));
  EXPECT_TRUE(bv.get(10));
  EXPECT_TRUE(bv.get(11));
  EXPECT_TRUE(bv.get(12));
  EXPECT_TRUE(bv.get(13));
  EXPECT_TRUE(bv.get(14));
  EXPECT_FALSE(bv.get(15));
  EXPECT_EQ(bv.popcount(), 5u);
}

TEST(CompressedBitVectorTest, Clear) {
  CompressedBitVector bv(100);

  bv.setRange(0, 50, true);
  EXPECT_EQ(bv.popcount(), 50u);

  bv.clear();
  EXPECT_EQ(bv.popcount(), 0u);
}

TEST(CompressedBitVectorTest, Resize) {
  CompressedBitVector bv(100);

  bv.set(50, true);
  bv.set(90, true);

  bv.resize(60);
  EXPECT_EQ(bv.size(), 60u);
  EXPECT_TRUE(bv.get(50));
  EXPECT_EQ(bv.popcount(), 1u); // Bit 90 should be gone

  bv.resize(200);
  EXPECT_EQ(bv.size(), 200u);
  EXPECT_TRUE(bv.get(50));
}

//===----------------------------------------------------------------------===//
// MemoryPool Tests
//===----------------------------------------------------------------------===//

struct TestObject {
  int value;
  char data[28];

  TestObject() : value(0) { memset(data, 0, sizeof(data)); }
  TestObject(int v) : value(v) { memset(data, 0, sizeof(data)); }
};

TEST(MemoryPoolTest, AllocateDeallocate) {
  MemoryPool<TestObject> pool(16);

  TestObject *obj1 = pool.allocate();
  TestObject *obj2 = pool.allocate();

  EXPECT_NE(obj1, nullptr);
  EXPECT_NE(obj2, nullptr);
  EXPECT_NE(obj1, obj2);

  pool.deallocate(obj1);
  pool.deallocate(obj2);
}

TEST(MemoryPoolTest, ConstructDestroy) {
  MemoryPool<TestObject> pool(16);

  TestObject *obj = pool.construct(42);
  EXPECT_EQ(obj->value, 42);

  pool.destroy(obj);
}

TEST(MemoryPoolTest, ReuseMemory) {
  MemoryPool<TestObject> pool(16);

  TestObject *obj1 = pool.allocate();
  pool.deallocate(obj1);

  TestObject *obj2 = pool.allocate();
  EXPECT_EQ(obj1, obj2); // Should reuse the same memory
}

TEST(MemoryPoolTest, Statistics) {
  MemoryPool<TestObject> pool(16);

  pool.allocate();
  pool.allocate();
  TestObject *obj3 = pool.allocate();
  pool.deallocate(obj3);

  auto stats = pool.getStatistics();
  EXPECT_EQ(stats.objectsAllocated, 3u);
  EXPECT_EQ(stats.objectsFreed, 1u);
  EXPECT_EQ(stats.blocksAllocated, 1u);
}

//===----------------------------------------------------------------------===//
// OnDemandElaborator Tests
//===----------------------------------------------------------------------===//

TEST(OnDemandElaboratorTest, RegisterAndElaborate) {
  OnDemandElaborator::Config config;
  config.maxElaboratedInstances = 10;
  OnDemandElaborator elaborator(config);

  bool elaborated = false;
  elaborator.registerInstance("top.mod1", [&elaborated](const std::string &) {
    elaborated = true;
  });

  EXPECT_FALSE(elaborator.isElaborated("top.mod1"));

  elaborator.ensureElaborated("top.mod1");

  EXPECT_TRUE(elaborated);
  EXPECT_TRUE(elaborator.isElaborated("top.mod1"));
}

TEST(OnDemandElaboratorTest, CacheHit) {
  OnDemandElaborator elaborator;

  int callCount = 0;
  elaborator.registerInstance("top.mod1", [&callCount](const std::string &) {
    callCount++;
  });

  elaborator.ensureElaborated("top.mod1");
  elaborator.ensureElaborated("top.mod1");
  elaborator.ensureElaborated("top.mod1");

  EXPECT_EQ(callCount, 1); // Should only elaborate once

  const auto &stats = elaborator.getStatistics();
  EXPECT_EQ(stats.elaborations, 1u);
  EXPECT_EQ(stats.cacheHits, 2u);
}

TEST(OnDemandElaboratorTest, LRUEviction) {
  OnDemandElaborator::Config config;
  config.maxElaboratedInstances = 2;
  OnDemandElaborator elaborator(config);

  elaborator.registerInstance("mod1", [](const std::string &) {});
  elaborator.registerInstance("mod2", [](const std::string &) {});
  elaborator.registerInstance("mod3", [](const std::string &) {});

  elaborator.ensureElaborated("mod1");
  elaborator.ensureElaborated("mod2");

  EXPECT_EQ(elaborator.getElaboratedCount(), 2u);

  // This should evict mod1 (LRU)
  elaborator.ensureElaborated("mod3");

  EXPECT_EQ(elaborator.getElaboratedCount(), 2u);
  EXPECT_FALSE(elaborator.isElaborated("mod1"));
  EXPECT_TRUE(elaborator.isElaborated("mod2"));
  EXPECT_TRUE(elaborator.isElaborated("mod3"));
}

//===----------------------------------------------------------------------===//
// MemoryManager Tests
//===----------------------------------------------------------------------===//

TEST(MemoryManagerTest, AllocateRegion) {
  MemoryManager manager;

  RegionId id = manager.allocateRegion(1024, "test_region");
  EXPECT_NE(id, InvalidRegionId);

  MemoryRegion *region = manager.getRegion(id);
  EXPECT_NE(region, nullptr);
  EXPECT_EQ(region->getSize(), 1024u);
}

TEST(MemoryManagerTest, FreeRegion) {
  MemoryManager manager;

  RegionId id = manager.allocateRegion(1024);
  EXPECT_NE(manager.getRegion(id), nullptr);

  manager.freeRegion(id);
  EXPECT_EQ(manager.getRegion(id), nullptr);
}

TEST(MemoryManagerTest, Statistics) {
  MemoryManager manager;

  manager.allocateRegion(1024);
  manager.allocateRegion(2048);

  const auto &stats = manager.getStatistics();
  EXPECT_EQ(stats.regionsAllocated, 2u);
  EXPECT_EQ(stats.totalBytesAllocated, 3072u);
  EXPECT_EQ(stats.currentBytesUsed, 3072u);
}

//===----------------------------------------------------------------------===//
// SignalStorage Tests
//===----------------------------------------------------------------------===//

TEST(SignalStorageTest, AllocateSignal) {
  MemoryManager manager;
  SignalStorage storage(manager);

  size_t id = storage.allocateSignal(8, "sig1");
  EXPECT_EQ(id, 0u);

  size_t id2 = storage.allocateSignal(32, "sig2");
  EXPECT_EQ(id2, 1u);

  EXPECT_EQ(storage.getSignalCount(), 2u);
}

TEST(SignalStorageTest, ReadWriteSignal) {
  MemoryManager manager;
  SignalStorage storage(manager);

  size_t id = storage.allocateSignal(32);

  uint32_t writeValue = 0xDEADBEEF;
  storage.writeSignal(id, &writeValue);

  uint32_t readValue = 0;
  storage.readSignal(id, &readValue);

  EXPECT_EQ(writeValue, readValue);
}

//===----------------------------------------------------------------------===//
// WaveformBuffer Tests
//===----------------------------------------------------------------------===//

TEST(WaveformBufferTest, RecordChange) {
  WaveformBuffer buffer;

  uint32_t value1 = 0x12345678;
  buffer.recordChange(0, 100, &value1, sizeof(value1));

  uint32_t value2 = 0xABCDEF00;
  buffer.recordChange(0, 200, &value2, sizeof(value2));

  auto changes = buffer.getChanges(0, 0, 1000);
  EXPECT_EQ(changes.size(), 2u);
}

TEST(WaveformBufferTest, GetValue) {
  WaveformBuffer::Config config;
  config.deltaCompression = false;
  config.deduplication = false;
  WaveformBuffer buffer(config);

  uint32_t value1 = 100;
  uint32_t value2 = 200;
  uint32_t value3 = 300;

  buffer.recordChange(0, 10, &value1, sizeof(value1));
  buffer.recordChange(0, 20, &value2, sizeof(value2));
  buffer.recordChange(0, 30, &value3, sizeof(value3));

  uint32_t result = 0;

  // Get value at time 15 (should be value1)
  EXPECT_TRUE(buffer.getValue(0, 15, &result, sizeof(result)));
  EXPECT_EQ(result, 100u);

  // Get value at time 25 (should be value2)
  EXPECT_TRUE(buffer.getValue(0, 25, &result, sizeof(result)));
  EXPECT_EQ(result, 200u);
}

TEST(WaveformBufferTest, Deduplication) {
  WaveformBuffer::Config config;
  config.deduplication = true;
  WaveformBuffer buffer(config);

  uint32_t value = 42;
  buffer.recordChange(0, 100, &value, sizeof(value));
  buffer.recordChange(0, 200, &value, sizeof(value)); // Same value, should skip
  buffer.recordChange(0, 300, &value, sizeof(value)); // Same value, should skip

  auto changes = buffer.getChanges(0, 0, 1000);
  EXPECT_EQ(changes.size(), 1u); // Only one change recorded
}

TEST(WaveformBufferTest, Clear) {
  WaveformBuffer buffer;

  uint32_t value = 42;
  buffer.recordChange(0, 100, &value, sizeof(value));

  buffer.clear();

  auto changes = buffer.getChanges(0, 0, 1000);
  EXPECT_TRUE(changes.empty());
}
