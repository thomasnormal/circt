//===- MemoryOptimization.h - Memory optimization for simulation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory optimization infrastructure for large design
// simulation. It provides memory-mapped state, compressed sparse storage,
// and on-demand elaboration to reduce memory footprint.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_MEMORYOPTIMIZATION_H
#define CIRCT_DIALECT_SIM_MEMORYOPTIMIZATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
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
// MemoryRegion - A region of simulation memory
//===----------------------------------------------------------------------===//

/// Unique identifier for a memory region.
using RegionId = uint32_t;

/// Invalid region ID constant.
constexpr RegionId InvalidRegionId = UINT32_MAX;

/// Represents a contiguous region of simulation memory.
class MemoryRegion {
public:
  MemoryRegion(RegionId id, size_t size, bool mmap = false);
  ~MemoryRegion();

  // Non-copyable but movable
  MemoryRegion(const MemoryRegion &) = delete;
  MemoryRegion &operator=(const MemoryRegion &) = delete;
  MemoryRegion(MemoryRegion &&other) noexcept;
  MemoryRegion &operator=(MemoryRegion &&other) noexcept;

  /// Get the region ID.
  RegionId getId() const { return id; }

  /// Get the size in bytes.
  size_t getSize() const { return size; }

  /// Check if this region is memory-mapped.
  bool isMapped() const { return mapped; }

  /// Get a pointer to the region data.
  uint8_t *getData() { return data; }
  const uint8_t *getData() const { return data; }

  /// Read data from the region.
  void read(size_t offset, void *dest, size_t bytes) const;

  /// Write data to the region.
  void write(size_t offset, const void *src, size_t bytes);

  /// Zero the entire region.
  void zero();

  /// Check if the region is allocated.
  bool isAllocated() const { return data != nullptr; }

  /// Get memory usage statistics.
  struct Statistics {
    size_t totalSize = 0;
    size_t usedSize = 0;
    size_t reads = 0;
    size_t writes = 0;
  };

  const Statistics &getStatistics() const { return stats; }

private:
  RegionId id;
  size_t size;
  uint8_t *data;
  bool mapped;
  bool owned;
  mutable Statistics stats;

#ifdef _WIN32
  void *mappingHandle = nullptr;
#else
  int fd = -1;
#endif
};

//===----------------------------------------------------------------------===//
// SparseArray - Compressed sparse array storage
//===----------------------------------------------------------------------===//

/// A sparse array that only stores non-zero values.
/// Efficient for large arrays with many zero/default values.
template <typename T, T DefaultValue = T()>
class SparseArray {
public:
  SparseArray(size_t logicalSize = 0) : logicalSize(logicalSize) {}

  /// Get the logical size of the array.
  size_t size() const { return logicalSize; }

  /// Resize the array (logical size).
  void resize(size_t newSize) { logicalSize = newSize; }

  /// Get a value at an index (returns default if not stored).
  T get(size_t index) const {
    auto it = values.find(index);
    return it != values.end() ? it->second : DefaultValue;
  }

  /// Set a value at an index.
  void set(size_t index, const T &value) {
    if (value == DefaultValue) {
      values.erase(index);
    } else {
      values[index] = value;
    }
  }

  /// Check if a value is stored at an index.
  bool has(size_t index) const { return values.find(index) != values.end(); }

  /// Clear all stored values.
  void clear() { values.clear(); }

  /// Get the number of stored (non-default) values.
  size_t storedCount() const { return values.size(); }

  /// Get the compression ratio (logical size / stored size).
  double compressionRatio() const {
    if (values.empty())
      return logicalSize > 0 ? static_cast<double>(logicalSize) : 1.0;
    return static_cast<double>(logicalSize) / values.size();
  }

  /// Iterate over stored values.
  using Iterator = typename std::unordered_map<size_t, T>::const_iterator;
  Iterator begin() const { return values.begin(); }
  Iterator end() const { return values.end(); }

  /// Get memory usage in bytes.
  size_t memoryUsage() const {
    return values.size() * (sizeof(size_t) + sizeof(T)) +
           sizeof(std::unordered_map<size_t, T>) + sizeof(size_t);
  }

private:
  size_t logicalSize;
  std::unordered_map<size_t, T> values;
};

//===----------------------------------------------------------------------===//
// CompressedBitVector - Space-efficient bit vector
//===----------------------------------------------------------------------===//

/// A compressed bit vector using run-length encoding.
class CompressedBitVector {
public:
  CompressedBitVector(size_t size = 0);

  /// Get the logical size in bits.
  size_t size() const { return logicalSize; }

  /// Resize the bit vector.
  void resize(size_t newSize);

  /// Get a bit value.
  bool get(size_t index) const;

  /// Set a bit value.
  void set(size_t index, bool value);

  /// Set a range of bits.
  void setRange(size_t start, size_t count, bool value);

  /// Clear all bits to zero.
  void clear();

  /// Count the number of set bits.
  size_t popcount() const;

  /// Get memory usage in bytes.
  size_t memoryUsage() const;

  /// Get compression ratio.
  double compressionRatio() const;

private:
  /// Decompress if beneficial, compress if beneficial.
  void optimize();

  size_t logicalSize;

  /// Run-length encoded representation: pairs of (start, length) for set bits.
  std::vector<std::pair<size_t, size_t>> runs;

  /// Threshold for switching between RLE and dense representation.
  static constexpr double DenseThreshold = 0.3;
};

//===----------------------------------------------------------------------===//
// MemoryPool - Pool allocator for simulation objects
//===----------------------------------------------------------------------===//

/// A memory pool for efficient allocation of fixed-size objects.
template <typename T>
class MemoryPool {
public:
  MemoryPool(size_t blockSize = 1024)
      : blockSize(blockSize), freeList(nullptr), allocated(0), freed(0) {}

  ~MemoryPool() {
    for (auto *block : blocks) {
      delete[] reinterpret_cast<char *>(block);
    }
  }

  /// Allocate an object.
  T *allocate() {
    if (freeList) {
      T *obj = freeList;
      freeList = *reinterpret_cast<T **>(obj);
      allocated++;
      return obj;
    }

    // Allocate new block
    allocateBlock();
    return allocate();
  }

  /// Deallocate an object.
  void deallocate(T *obj) {
    *reinterpret_cast<T **>(obj) = freeList;
    freeList = obj;
    freed++;
  }

  /// Construct an object in place.
  template <typename... Args>
  T *construct(Args &&...args) {
    T *obj = allocate();
    new (obj) T(std::forward<Args>(args)...);
    return obj;
  }

  /// Destroy and deallocate an object.
  void destroy(T *obj) {
    obj->~T();
    deallocate(obj);
  }

  /// Get statistics.
  struct Statistics {
    size_t blocksAllocated = 0;
    size_t objectsAllocated = 0;
    size_t objectsFreed = 0;
    size_t memoryUsed = 0;
  };

  Statistics getStatistics() const {
    Statistics stats;
    stats.blocksAllocated = blocks.size();
    stats.objectsAllocated = allocated;
    stats.objectsFreed = freed;
    stats.memoryUsed = blocks.size() * blockSize * sizeof(T);
    return stats;
  }

private:
  void allocateBlock() {
    char *block = new char[blockSize * sizeof(T)];
    blocks.push_back(reinterpret_cast<T *>(block));

    // Initialize free list within block
    T *first = reinterpret_cast<T *>(block);
    for (size_t i = 0; i < blockSize - 1; ++i) {
      *reinterpret_cast<T **>(&first[i]) = &first[i + 1];
    }
    *reinterpret_cast<T **>(&first[blockSize - 1]) = freeList;
    freeList = first;
  }

  size_t blockSize;
  std::vector<T *> blocks;
  T *freeList;
  size_t allocated;
  size_t freed;
};

//===----------------------------------------------------------------------===//
// OnDemandElaborator - Lazy elaboration of design hierarchy
//===----------------------------------------------------------------------===//

/// Callback for elaborating a module instance.
using ElaborateCallback = std::function<void(const std::string &instancePath)>;

/// Manages on-demand elaboration of the design hierarchy.
class OnDemandElaborator {
public:
  /// Configuration for on-demand elaboration.
  struct Config {
    /// Maximum number of instances to keep elaborated.
    size_t maxElaboratedInstances;

    /// Whether to elaborate in background thread.
    bool backgroundElaboration;

    /// Prefetch depth (how many levels ahead to elaborate).
    size_t prefetchDepth;

    Config()
        : maxElaboratedInstances(1000), backgroundElaboration(false),
          prefetchDepth(2) {}
  };

  OnDemandElaborator(Config config = Config());
  ~OnDemandElaborator();

  /// Register an instance for potential elaboration.
  void registerInstance(const std::string &path, ElaborateCallback elaborate);

  /// Check if an instance is elaborated.
  bool isElaborated(const std::string &path) const;

  /// Ensure an instance is elaborated.
  void ensureElaborated(const std::string &path);

  /// Mark an instance as accessed (for LRU tracking).
  void markAccessed(const std::string &path);

  /// Unelaborate instances to free memory.
  void unelaborate(const std::string &path);

  /// Unelaborate least recently used instances.
  void evictLRU(size_t count = 1);

  /// Get the number of elaborated instances.
  size_t getElaboratedCount() const { return elaboratedInstances.size(); }

  /// Statistics.
  struct Statistics {
    size_t elaborations = 0;
    size_t unelaborations = 0;
    size_t cacheHits = 0;
    size_t cacheMisses = 0;
  };

  const Statistics &getStatistics() const { return stats; }

private:
  struct InstanceInfo {
    ElaborateCallback elaborate;
    uint64_t lastAccess;
    bool isElaborated;
  };

  Config config;
  std::unordered_map<std::string, InstanceInfo> instances;
  std::unordered_map<std::string, bool> elaboratedInstances;
  uint64_t accessCounter;
  mutable std::mutex mutex;
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// MemoryManager - Central memory management for simulation
//===----------------------------------------------------------------------===//

/// Central manager for all simulation memory.
class MemoryManager {
public:
  /// Configuration for the memory manager.
  struct Config {
    /// Maximum total memory usage in bytes.
    size_t maxMemory;

    /// Whether to use memory mapping for large regions.
    bool useMmap;

    /// Threshold for memory mapping (regions larger than this are mapped).
    size_t mmapThreshold;

    /// Enable sparse array optimization.
    bool useSparseArrays;

    /// Enable on-demand elaboration.
    bool useOnDemandElaboration;

    Config()
        : maxMemory(4ULL * 1024 * 1024 * 1024), // 4 GB
          useMmap(true), mmapThreshold(1024 * 1024), // 1 MB
          useSparseArrays(true), useOnDemandElaboration(true) {}
  };

  MemoryManager(Config config = Config());
  ~MemoryManager();

  //===------------------------------------------------------------------===//
  // Region Management
  //===------------------------------------------------------------------===//

  /// Allocate a memory region.
  RegionId allocateRegion(size_t size, const std::string &name = "");

  /// Free a memory region.
  void freeRegion(RegionId id);

  /// Get a memory region.
  MemoryRegion *getRegion(RegionId id);
  const MemoryRegion *getRegion(RegionId id) const;

  //===------------------------------------------------------------------===//
  // Sparse Array Management
  //===------------------------------------------------------------------===//

  /// Create a sparse array for the given type.
  template <typename T>
  SparseArray<T> *createSparseArray(size_t size, const std::string &name = "") {
    auto *array = new SparseArray<T>(size);
    sparseArrays[name.empty() ? std::to_string(sparseArrayCounter++)
                              : name] = array;
    return array;
  }

  //===------------------------------------------------------------------===//
  // On-Demand Elaboration
  //===------------------------------------------------------------------===//

  /// Get the on-demand elaborator.
  OnDemandElaborator &getElaborator() { return elaborator; }

  //===------------------------------------------------------------------===//
  // Memory Pressure Handling
  //===------------------------------------------------------------------===//

  /// Get current memory usage.
  size_t getCurrentUsage() const;

  /// Get maximum memory allowed.
  size_t getMaxMemory() const { return config.maxMemory; }

  /// Check if under memory pressure.
  bool isUnderPressure() const {
    return getCurrentUsage() > config.maxMemory * 0.9;
  }

  /// Free memory to reduce pressure.
  void reducePressure();

  //===------------------------------------------------------------------===//
  // Statistics
  //===------------------------------------------------------------------===//

  struct Statistics {
    size_t regionsAllocated = 0;
    size_t regionsFreed = 0;
    size_t totalBytesAllocated = 0;
    size_t currentBytesUsed = 0;
    size_t peakBytesUsed = 0;
    size_t mmappedRegions = 0;
    size_t sparseArrays = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Print memory report.
  void printReport(llvm::raw_ostream &os) const;

private:
  Config config;
  llvm::DenseMap<RegionId, std::unique_ptr<MemoryRegion>> regions;
  std::unordered_map<std::string, void *> sparseArrays;
  OnDemandElaborator elaborator;
  RegionId nextRegionId = 0;
  size_t sparseArrayCounter = 0;
  Statistics stats;
  mutable std::mutex mutex;
};

//===----------------------------------------------------------------------===//
// SignalStorage - Optimized storage for simulation signals
//===----------------------------------------------------------------------===//

/// Storage class for simulation signal values with compression.
class SignalStorage {
public:
  /// Configuration for signal storage.
  struct Config {
    /// Use compression for large signals.
    bool useCompression;

    /// Threshold for compression (signals larger than this are compressed).
    size_t compressionThreshold;

    /// Use delta encoding for waveform storage.
    bool useDeltaEncoding;

    Config()
        : useCompression(true), compressionThreshold(1024),
          useDeltaEncoding(true) {}
  };

  SignalStorage(MemoryManager &manager, Config config = Config());

  /// Allocate storage for a signal.
  size_t allocateSignal(size_t bitWidth, const std::string &name = "");

  /// Free storage for a signal.
  void freeSignal(size_t signalId);

  /// Read a signal value.
  void readSignal(size_t signalId, void *dest) const;

  /// Write a signal value.
  void writeSignal(size_t signalId, const void *src);

  /// Get the number of allocated signals.
  size_t getSignalCount() const { return signals.size(); }

  /// Get memory usage.
  size_t getMemoryUsage() const;

private:
  struct SignalInfo {
    size_t bitWidth;
    size_t byteSize;
    size_t offset; // Offset in region
    RegionId region;
    std::string name;
  };

  MemoryManager &manager;
  Config config;
  std::vector<SignalInfo> signals;
  RegionId currentRegion;
  size_t currentOffset;
  size_t nextSignalId = 0;
};

//===----------------------------------------------------------------------===//
// WaveformBuffer - Efficient waveform storage with compression
//===----------------------------------------------------------------------===//

/// Stores waveform data with delta compression.
class WaveformBuffer {
public:
  /// Configuration for waveform storage.
  struct Config {
    /// Maximum buffer size before flushing to disk.
    size_t maxBufferSize;

    /// Enable delta compression.
    bool deltaCompression;

    /// Enable value change deduplication.
    bool deduplication;

    Config()
        : maxBufferSize(64 * 1024 * 1024), // 64 MB
          deltaCompression(true), deduplication(true) {}
  };

  WaveformBuffer(Config config = Config());

  /// Record a value change.
  void recordChange(size_t signalId, uint64_t time, const void *value,
                    size_t bytes);

  /// Get value at a specific time.
  bool getValue(size_t signalId, uint64_t time, void *value,
                size_t bytes) const;

  /// Get all changes for a signal in a time range.
  std::vector<std::pair<uint64_t, std::vector<uint8_t>>>
  getChanges(size_t signalId, uint64_t startTime, uint64_t endTime) const;

  /// Flush buffer to storage.
  void flush();

  /// Clear all data.
  void clear();

  /// Get memory usage.
  size_t getMemoryUsage() const;

  /// Get compression ratio.
  double getCompressionRatio() const;

private:
  struct ValueChange {
    uint64_t time;
    std::vector<uint8_t> value;
  };

  Config config;
  std::unordered_map<size_t, std::vector<ValueChange>> changes;
  size_t currentSize;
  size_t uncompressedSize;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_MEMORYOPTIMIZATION_H
