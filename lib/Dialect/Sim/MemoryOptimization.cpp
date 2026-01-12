//===- MemoryOptimization.cpp - Memory optimization for simulation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements memory optimization infrastructure for large designs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/MemoryOptimization.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#define DEBUG_TYPE "sim-memory"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// MemoryRegion Implementation
//===----------------------------------------------------------------------===//

MemoryRegion::MemoryRegion(RegionId id, size_t size, bool mmap)
    : id(id), size(size), data(nullptr), mapped(false), owned(true) {
  if (mmap && size > 0) {
#ifdef _WIN32
    // Windows memory mapping
    mappingHandle = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr,
                                      PAGE_READWRITE, (size >> 32) & 0xFFFFFFFF,
                                      size & 0xFFFFFFFF, nullptr);
    if (mappingHandle) {
      data = static_cast<uint8_t *>(
          MapViewOfFile(mappingHandle, FILE_MAP_ALL_ACCESS, 0, 0, size));
      if (data) {
        mapped = true;
      } else {
        CloseHandle(mappingHandle);
        mappingHandle = nullptr;
      }
    }
#else
    // POSIX memory mapping
    data = static_cast<uint8_t *>(
        ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (data == MAP_FAILED) {
      data = nullptr;
    } else {
      mapped = true;
    }
#endif
  }

  // Fall back to regular allocation if mmap failed or wasn't requested
  if (!data && size > 0) {
    data = new uint8_t[size];
    mapped = false;
  }

  if (data) {
    stats.totalSize = size;
    zero();
  }
}

MemoryRegion::~MemoryRegion() {
  if (data) {
    if (mapped) {
#ifdef _WIN32
      UnmapViewOfFile(data);
      if (mappingHandle) {
        CloseHandle(mappingHandle);
      }
#else
      munmap(data, size);
#endif
    } else if (owned) {
      delete[] data;
    }
  }
}

MemoryRegion::MemoryRegion(MemoryRegion &&other) noexcept
    : id(other.id), size(other.size), data(other.data), mapped(other.mapped),
      owned(other.owned), stats(other.stats) {
#ifdef _WIN32
  mappingHandle = other.mappingHandle;
  other.mappingHandle = nullptr;
#else
  fd = other.fd;
  other.fd = -1;
#endif
  other.data = nullptr;
  other.size = 0;
  other.owned = false;
}

MemoryRegion &MemoryRegion::operator=(MemoryRegion &&other) noexcept {
  if (this != &other) {
    // Clean up existing resources
    if (data) {
      if (mapped) {
#ifdef _WIN32
        UnmapViewOfFile(data);
        if (mappingHandle) {
          CloseHandle(mappingHandle);
        }
#else
        munmap(data, size);
#endif
      } else if (owned) {
        delete[] data;
      }
    }

    // Move from other
    id = other.id;
    size = other.size;
    data = other.data;
    mapped = other.mapped;
    owned = other.owned;
    stats = other.stats;

#ifdef _WIN32
    mappingHandle = other.mappingHandle;
    other.mappingHandle = nullptr;
#else
    fd = other.fd;
    other.fd = -1;
#endif

    other.data = nullptr;
    other.size = 0;
    other.owned = false;
  }
  return *this;
}

void MemoryRegion::read(size_t offset, void *dest, size_t bytes) const {
  if (data && offset + bytes <= size) {
    std::memcpy(dest, data + offset, bytes);
    stats.reads++;
  }
}

void MemoryRegion::write(size_t offset, const void *src, size_t bytes) {
  if (data && offset + bytes <= size) {
    std::memcpy(data + offset, src, bytes);
    stats.writes++;
    stats.usedSize = std::max(stats.usedSize, offset + bytes);
  }
}

void MemoryRegion::zero() {
  if (data) {
    std::memset(data, 0, size);
  }
}

//===----------------------------------------------------------------------===//
// CompressedBitVector Implementation
//===----------------------------------------------------------------------===//

CompressedBitVector::CompressedBitVector(size_t size) : logicalSize(size) {}

void CompressedBitVector::resize(size_t newSize) {
  if (newSize < logicalSize) {
    // Truncate runs
    for (auto it = runs.begin(); it != runs.end();) {
      if (it->first >= newSize) {
        it = runs.erase(it);
      } else if (it->first + it->second > newSize) {
        it->second = newSize - it->first;
        ++it;
      } else {
        ++it;
      }
    }
  }
  logicalSize = newSize;
}

bool CompressedBitVector::get(size_t index) const {
  if (index >= logicalSize)
    return false;

  for (const auto &run : runs) {
    if (index >= run.first && index < run.first + run.second) {
      return true;
    }
  }
  return false;
}

void CompressedBitVector::set(size_t index, bool value) {
  if (index >= logicalSize)
    return;

  if (value) {
    // Try to extend an existing run
    for (auto &run : runs) {
      if (index == run.first - 1) {
        run.first--;
        run.second++;
        return;
      }
      if (index == run.first + run.second) {
        run.second++;
        return;
      }
      if (index >= run.first && index < run.first + run.second) {
        return; // Already set
      }
    }
    // Create new run
    runs.push_back({index, 1});
  } else {
    // Remove from runs
    for (auto it = runs.begin(); it != runs.end(); ++it) {
      if (index >= it->first && index < it->first + it->second) {
        if (index == it->first) {
          it->first++;
          it->second--;
          if (it->second == 0) {
            runs.erase(it);
          }
        } else if (index == it->first + it->second - 1) {
          it->second--;
        } else {
          // Split run
          size_t newStart = index + 1;
          size_t newLength = it->first + it->second - newStart;
          it->second = index - it->first;
          runs.push_back({newStart, newLength});
        }
        return;
      }
    }
  }
}

void CompressedBitVector::setRange(size_t start, size_t count, bool value) {
  for (size_t i = 0; i < count; ++i) {
    set(start + i, value);
  }
  optimize();
}

void CompressedBitVector::clear() { runs.clear(); }

size_t CompressedBitVector::popcount() const {
  size_t count = 0;
  for (const auto &run : runs) {
    count += run.second;
  }
  return count;
}

size_t CompressedBitVector::memoryUsage() const {
  return runs.size() * sizeof(std::pair<size_t, size_t>) +
         sizeof(std::vector<std::pair<size_t, size_t>>) + sizeof(size_t);
}

double CompressedBitVector::compressionRatio() const {
  size_t denseSize = (logicalSize + 7) / 8;
  size_t compressedSize = memoryUsage();
  return compressedSize > 0 ? static_cast<double>(denseSize) / compressedSize
                            : 1.0;
}

void CompressedBitVector::optimize() {
  // Sort runs
  std::sort(runs.begin(), runs.end());

  // Merge adjacent runs
  std::vector<std::pair<size_t, size_t>> merged;
  for (const auto &run : runs) {
    if (merged.empty() || merged.back().first + merged.back().second < run.first) {
      merged.push_back(run);
    } else {
      merged.back().second =
          std::max(merged.back().first + merged.back().second,
                   run.first + run.second) -
          merged.back().first;
    }
  }
  runs = std::move(merged);
}

//===----------------------------------------------------------------------===//
// OnDemandElaborator Implementation
//===----------------------------------------------------------------------===//

OnDemandElaborator::OnDemandElaborator(Config config)
    : config(config), accessCounter(0) {}

OnDemandElaborator::~OnDemandElaborator() = default;

void OnDemandElaborator::registerInstance(const std::string &path,
                                          ElaborateCallback elaborate) {
  std::lock_guard<std::mutex> lock(mutex);
  InstanceInfo info;
  info.elaborate = std::move(elaborate);
  info.lastAccess = 0;
  info.isElaborated = false;
  instances[path] = std::move(info);
}

bool OnDemandElaborator::isElaborated(const std::string &path) const {
  std::lock_guard<std::mutex> lock(mutex);
  return elaboratedInstances.find(path) != elaboratedInstances.end();
}

void OnDemandElaborator::ensureElaborated(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex);

  auto it = instances.find(path);
  if (it == instances.end())
    return;

  if (it->second.isElaborated) {
    stats.cacheHits++;
    it->second.lastAccess = ++accessCounter;
    return;
  }

  stats.cacheMisses++;

  // Evict if necessary
  while (elaboratedInstances.size() >= config.maxElaboratedInstances) {
    // Find LRU instance
    std::string lruPath;
    uint64_t lruAccess = UINT64_MAX;
    for (const auto &kv : instances) {
      if (kv.second.isElaborated && kv.second.lastAccess < lruAccess) {
        lruAccess = kv.second.lastAccess;
        lruPath = kv.first;
      }
    }
    if (!lruPath.empty()) {
      unelaborate(lruPath);
    } else {
      break;
    }
  }

  // Elaborate
  if (it->second.elaborate) {
    it->second.elaborate(path);
  }
  it->second.isElaborated = true;
  it->second.lastAccess = ++accessCounter;
  elaboratedInstances[path] = true;
  stats.elaborations++;

  LLVM_DEBUG(llvm::dbgs() << "Elaborated instance: " << path << "\n");
}

void OnDemandElaborator::markAccessed(const std::string &path) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = instances.find(path);
  if (it != instances.end()) {
    it->second.lastAccess = ++accessCounter;
  }
}

void OnDemandElaborator::unelaborate(const std::string &path) {
  // Note: This is called with mutex held
  auto it = instances.find(path);
  if (it != instances.end() && it->second.isElaborated) {
    it->second.isElaborated = false;
    elaboratedInstances.erase(path);
    stats.unelaborations++;
    LLVM_DEBUG(llvm::dbgs() << "Unelaborated instance: " << path << "\n");
  }
}

void OnDemandElaborator::evictLRU(size_t count) {
  std::lock_guard<std::mutex> lock(mutex);

  for (size_t i = 0; i < count && !elaboratedInstances.empty(); ++i) {
    std::string lruPath;
    uint64_t lruAccess = UINT64_MAX;
    for (const auto &kv : instances) {
      if (kv.second.isElaborated && kv.second.lastAccess < lruAccess) {
        lruAccess = kv.second.lastAccess;
        lruPath = kv.first;
      }
    }
    if (!lruPath.empty()) {
      unelaborate(lruPath);
    }
  }
}

//===----------------------------------------------------------------------===//
// MemoryManager Implementation
//===----------------------------------------------------------------------===//

MemoryManager::MemoryManager(Config config)
    : config(config),
      elaborator(OnDemandElaborator::Config{
          1000, config.useOnDemandElaboration, 2}) {}

MemoryManager::~MemoryManager() {
  // Clean up sparse arrays
  for (auto &kv : sparseArrays) {
    // Note: We don't know the actual type, so we can't properly delete
    // In production, this would need type erasure
  }
}

RegionId MemoryManager::allocateRegion(size_t size, const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex);

  bool useMmap = config.useMmap && size >= config.mmapThreshold;
  auto region = std::make_unique<MemoryRegion>(nextRegionId, size, useMmap);

  if (!region->isAllocated()) {
    return InvalidRegionId;
  }

  RegionId id = nextRegionId++;
  stats.regionsAllocated++;
  stats.totalBytesAllocated += size;
  stats.currentBytesUsed += size;
  stats.peakBytesUsed = std::max(stats.peakBytesUsed, stats.currentBytesUsed);
  if (useMmap) {
    stats.mmappedRegions++;
  }

  regions[id] = std::move(region);

  LLVM_DEBUG(llvm::dbgs() << "Allocated region " << id << " (" << size
                          << " bytes, mmap=" << useMmap << ")\n");
  return id;
}

void MemoryManager::freeRegion(RegionId id) {
  std::lock_guard<std::mutex> lock(mutex);

  auto it = regions.find(id);
  if (it != regions.end()) {
    stats.currentBytesUsed -= it->second->getSize();
    stats.regionsFreed++;
    if (it->second->isMapped()) {
      stats.mmappedRegions--;
    }
    regions.erase(it);

    LLVM_DEBUG(llvm::dbgs() << "Freed region " << id << "\n");
  }
}

MemoryRegion *MemoryManager::getRegion(RegionId id) {
  auto it = regions.find(id);
  return it != regions.end() ? it->second.get() : nullptr;
}

const MemoryRegion *MemoryManager::getRegion(RegionId id) const {
  auto it = regions.find(id);
  return it != regions.end() ? it->second.get() : nullptr;
}

size_t MemoryManager::getCurrentUsage() const {
  std::lock_guard<std::mutex> lock(mutex);
  return stats.currentBytesUsed;
}

void MemoryManager::reducePressure() {
  std::lock_guard<std::mutex> lock(mutex);

  // Evict elaborated instances
  elaborator.evictLRU(10);

  // Could also compress sparse arrays, evict cached data, etc.
  LLVM_DEBUG(llvm::dbgs() << "Reduced memory pressure\n");
}

void MemoryManager::printReport(llvm::raw_ostream &os) const {
  std::lock_guard<std::mutex> lock(mutex);

  os << "=== Memory Manager Report ===\n";
  os << "Regions allocated: " << stats.regionsAllocated << "\n";
  os << "Regions freed: " << stats.regionsFreed << "\n";
  os << "Current regions: " << regions.size() << "\n";
  os << "Memory-mapped regions: " << stats.mmappedRegions << "\n";
  os << "\n";
  os << "Memory usage:\n";
  os << "  Total allocated: " << stats.totalBytesAllocated << " bytes\n";
  os << "  Current usage: " << stats.currentBytesUsed << " bytes\n";
  os << "  Peak usage: " << stats.peakBytesUsed << " bytes\n";
  os << "  Max allowed: " << config.maxMemory << " bytes\n";
  os << "\n";

  auto elaboratorStats = elaborator.getStatistics();
  os << "On-demand elaboration:\n";
  os << "  Elaborations: " << elaboratorStats.elaborations << "\n";
  os << "  Unelaborations: " << elaboratorStats.unelaborations << "\n";
  os << "  Cache hits: " << elaboratorStats.cacheHits << "\n";
  os << "  Cache misses: " << elaboratorStats.cacheMisses << "\n";
}

//===----------------------------------------------------------------------===//
// SignalStorage Implementation
//===----------------------------------------------------------------------===//

SignalStorage::SignalStorage(MemoryManager &manager, Config config)
    : manager(manager), config(config), currentRegion(InvalidRegionId),
      currentOffset(0) {}

size_t SignalStorage::allocateSignal(size_t bitWidth, const std::string &name) {
  size_t byteSize = (bitWidth + 7) / 8;

  // Allocate a region if needed
  if (currentRegion == InvalidRegionId || currentOffset + byteSize > 1024 * 1024) {
    currentRegion = manager.allocateRegion(1024 * 1024, "signals");
    currentOffset = 0;
  }

  SignalInfo info;
  info.bitWidth = bitWidth;
  info.byteSize = byteSize;
  info.offset = currentOffset;
  info.region = currentRegion;
  info.name = name;

  size_t signalId = nextSignalId++;
  signals.push_back(info);
  currentOffset += byteSize;

  return signalId;
}

void SignalStorage::freeSignal(size_t signalId) {
  // Note: Simple implementation doesn't actually free individual signals
  // A more sophisticated implementation would use a free list
}

void SignalStorage::readSignal(size_t signalId, void *dest) const {
  if (signalId >= signals.size())
    return;

  const SignalInfo &info = signals[signalId];
  const MemoryRegion *region = manager.getRegion(info.region);
  if (region) {
    region->read(info.offset, dest, info.byteSize);
  }
}

void SignalStorage::writeSignal(size_t signalId, const void *src) {
  if (signalId >= signals.size())
    return;

  const SignalInfo &info = signals[signalId];
  MemoryRegion *region = manager.getRegion(info.region);
  if (region) {
    region->write(info.offset, src, info.byteSize);
  }
}

size_t SignalStorage::getMemoryUsage() const {
  size_t total = 0;
  for (const auto &info : signals) {
    total += info.byteSize;
  }
  return total;
}

//===----------------------------------------------------------------------===//
// WaveformBuffer Implementation
//===----------------------------------------------------------------------===//

WaveformBuffer::WaveformBuffer(Config config)
    : config(config), currentSize(0), uncompressedSize(0) {}

void WaveformBuffer::recordChange(size_t signalId, uint64_t time,
                                  const void *value, size_t bytes) {
  ValueChange change;
  change.time = time;
  change.value.assign(static_cast<const uint8_t *>(value),
                      static_cast<const uint8_t *>(value) + bytes);

  auto &signalChanges = changes[signalId];

  // Deduplication: skip if value hasn't changed
  if (config.deduplication && !signalChanges.empty()) {
    if (signalChanges.back().value == change.value) {
      return;
    }
  }

  // Delta compression: store XOR with previous value
  if (config.deltaCompression && !signalChanges.empty()) {
    const auto &prev = signalChanges.back().value;
    if (prev.size() == change.value.size()) {
      std::vector<uint8_t> delta(bytes);
      for (size_t i = 0; i < bytes; ++i) {
        delta[i] = change.value[i] ^ prev[i];
      }
      change.value = std::move(delta);
    }
  }

  signalChanges.push_back(std::move(change));
  currentSize += bytes + sizeof(uint64_t);
  uncompressedSize += bytes + sizeof(uint64_t);

  // Flush if buffer is full
  if (currentSize >= config.maxBufferSize) {
    flush();
  }
}

bool WaveformBuffer::getValue(size_t signalId, uint64_t time, void *value,
                              size_t bytes) const {
  auto it = changes.find(signalId);
  if (it == changes.end() || it->second.empty())
    return false;

  const auto &signalChanges = it->second;

  // Binary search for the time
  auto changeIt = std::lower_bound(
      signalChanges.begin(), signalChanges.end(), time,
      [](const ValueChange &c, uint64_t t) { return c.time <= t; });

  if (changeIt == signalChanges.begin())
    return false;

  --changeIt;

  // Reconstruct value if delta compressed
  if (config.deltaCompression) {
    std::vector<uint8_t> reconstructed(bytes, 0);
    for (auto vit = signalChanges.begin(); vit <= changeIt; ++vit) {
      if (vit->value.size() == bytes) {
        for (size_t i = 0; i < bytes; ++i) {
          reconstructed[i] ^= vit->value[i];
        }
      }
    }
    std::memcpy(value, reconstructed.data(), bytes);
  } else {
    if (changeIt->value.size() >= bytes) {
      std::memcpy(value, changeIt->value.data(), bytes);
    } else {
      return false;
    }
  }

  return true;
}

std::vector<std::pair<uint64_t, std::vector<uint8_t>>>
WaveformBuffer::getChanges(size_t signalId, uint64_t startTime,
                           uint64_t endTime) const {
  std::vector<std::pair<uint64_t, std::vector<uint8_t>>> result;

  auto it = changes.find(signalId);
  if (it == changes.end())
    return result;

  for (const auto &change : it->second) {
    if (change.time >= startTime && change.time <= endTime) {
      result.push_back({change.time, change.value});
    }
  }

  return result;
}

void WaveformBuffer::flush() {
  // In a full implementation, this would write to disk
  LLVM_DEBUG(llvm::dbgs() << "Flushing waveform buffer (" << currentSize
                          << " bytes)\n");
}

void WaveformBuffer::clear() {
  changes.clear();
  currentSize = 0;
  uncompressedSize = 0;
}

size_t WaveformBuffer::getMemoryUsage() const { return currentSize; }

double WaveformBuffer::getCompressionRatio() const {
  return currentSize > 0
             ? static_cast<double>(uncompressedSize) / currentSize
             : 1.0;
}
