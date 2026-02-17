//===- ParallelScheduler.h - Multi-core parallel simulation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the parallel scheduler infrastructure for multi-core
// simulation. It provides partition-based parallelism with thread-safe
// synchronization for boundary signals and barrier-based delta cycle sync.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_PARALLELSCHEDULER_H
#define CIRCT_DIALECT_SIM_PARALLELSCHEDULER_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// PartitionId - Unique identifier for a design partition
//===----------------------------------------------------------------------===//

using PartitionId = uint32_t;

/// Invalid partition ID constant.
constexpr PartitionId InvalidPartitionId = UINT32_MAX;

//===----------------------------------------------------------------------===//
// BoundarySignal - A signal that crosses partition boundaries
//===----------------------------------------------------------------------===//

/// Represents a signal that is shared between partitions.
/// These require special handling for thread-safe updates.
struct BoundarySignal {
  /// The signal ID.
  SignalId signalId;

  /// The source partition (writer).
  PartitionId sourcePartition;

  /// The destination partitions (readers).
  llvm::SmallVector<PartitionId, 4> destPartitions;

  /// Buffered value for synchronization.
  SignalValue bufferedValue;

  /// Lock for thread-safe access.
  mutable std::mutex valueMutex;

  BoundarySignal(SignalId id, PartitionId source)
      : signalId(id), sourcePartition(source) {}

  /// Thread-safe read of the buffered value.
  SignalValue read() const {
    std::lock_guard<std::mutex> lock(valueMutex);
    return bufferedValue;
  }

  /// Thread-safe write to the buffered value.
  void write(const SignalValue &value) {
    std::lock_guard<std::mutex> lock(valueMutex);
    bufferedValue = value;
  }
};

//===----------------------------------------------------------------------===//
// Partition - A unit of parallel execution
//===----------------------------------------------------------------------===//

/// Represents a partition of the design that can be simulated in parallel.
/// Each partition has its own set of processes and internal signals.
class Partition {
public:
  Partition(PartitionId id, const std::string &name)
      : id(id), name(name), active(false) {}

  /// Get the partition ID.
  PartitionId getId() const { return id; }

  /// Get the partition name.
  const std::string &getName() const { return name; }

  /// Add a process to this partition.
  void addProcess(ProcessId processId) { processes.push_back(processId); }

  /// Add an internal signal to this partition.
  void addInternalSignal(SignalId signalId) {
    internalSignals.push_back(signalId);
  }

  /// Add an input boundary signal (from another partition).
  void addInputBoundary(SignalId signalId) {
    inputBoundarySignals.push_back(signalId);
  }

  /// Add an output boundary signal (to other partitions).
  void addOutputBoundary(SignalId signalId) {
    outputBoundarySignals.push_back(signalId);
  }

  /// Get all processes in this partition.
  const llvm::SmallVector<ProcessId, 16> &getProcesses() const {
    return processes;
  }

  /// Get all internal signals.
  const llvm::SmallVector<SignalId, 32> &getInternalSignals() const {
    return internalSignals;
  }

  /// Get input boundary signals.
  const llvm::SmallVector<SignalId, 8> &getInputBoundarySignals() const {
    return inputBoundarySignals;
  }

  /// Get output boundary signals.
  const llvm::SmallVector<SignalId, 8> &getOutputBoundarySignals() const {
    return outputBoundarySignals;
  }

  /// Get the process count.
  size_t getProcessCount() const { return processes.size(); }

  /// Get the signal count (internal only).
  size_t getSignalCount() const { return internalSignals.size(); }

  /// Check if this partition is active (has work to do).
  bool isActive() const { return active.load(); }

  /// Set the active state.
  void setActive(bool state) { active.store(state); }

  /// Get the load metric for this partition (for balancing).
  double getLoad() const {
    // Simple load metric: weighted sum of processes and signals
    return static_cast<double>(processes.size()) * 2.0 +
           static_cast<double>(internalSignals.size()) * 0.5;
  }

  //===------------------------------------------------------------------===//
  // Partition Statistics
  //===------------------------------------------------------------------===//

  struct Statistics {
    std::atomic<size_t> eventsProcessed{0};
    std::atomic<size_t> deltaCycles{0};
    std::atomic<size_t> boundaryReads{0};
    std::atomic<size_t> boundaryWrites{0};
    std::atomic<uint64_t> executionTimeNs{0};
  };

  Statistics &getStatistics() { return stats; }
  const Statistics &getStatistics() const { return stats; }

private:
  PartitionId id;
  std::string name;
  llvm::SmallVector<ProcessId, 16> processes;
  llvm::SmallVector<SignalId, 32> internalSignals;
  llvm::SmallVector<SignalId, 8> inputBoundarySignals;
  llvm::SmallVector<SignalId, 8> outputBoundarySignals;
  std::atomic<bool> active;
  Statistics stats;
};

//===----------------------------------------------------------------------===//
// PartitionGraph - Graph of partition dependencies
//===----------------------------------------------------------------------===//

/// Represents the dependency graph between partitions.
/// Used for ordering and synchronization decisions.
class PartitionGraph {
public:
  /// Add an edge from source to destination partition.
  void addEdge(PartitionId src, PartitionId dst, SignalId signal) {
    edges[src].push_back({dst, signal});
    reverseEdges[dst].push_back({src, signal});
  }

  /// Get outgoing edges from a partition.
  const llvm::SmallVector<std::pair<PartitionId, SignalId>, 8> &
  getOutEdges(PartitionId id) const {
    static const llvm::SmallVector<std::pair<PartitionId, SignalId>, 8> empty;
    auto it = edges.find(id);
    return it != edges.end() ? it->second : empty;
  }

  /// Get incoming edges to a partition.
  const llvm::SmallVector<std::pair<PartitionId, SignalId>, 8> &
  getInEdges(PartitionId id) const {
    static const llvm::SmallVector<std::pair<PartitionId, SignalId>, 8> empty;
    auto it = reverseEdges.find(id);
    return it != reverseEdges.end() ? it->second : empty;
  }

  /// Get the number of boundary signals (cut edges).
  size_t getCutSize() const {
    size_t count = 0;
    for (const auto &kv : edges) {
      count += kv.second.size();
    }
    return count;
  }

  /// Clear the graph.
  void clear() {
    edges.clear();
    reverseEdges.clear();
  }

private:
  llvm::DenseMap<PartitionId,
                 llvm::SmallVector<std::pair<PartitionId, SignalId>, 8>>
      edges;
  llvm::DenseMap<PartitionId,
                 llvm::SmallVector<std::pair<PartitionId, SignalId>, 8>>
      reverseEdges;
};

//===----------------------------------------------------------------------===//
// ThreadBarrier - Synchronization barrier for thread coordination
//===----------------------------------------------------------------------===//

/// A reusable thread barrier for synchronizing worker threads.
class ThreadBarrier {
public:
  explicit ThreadBarrier(size_t count) : threshold(count), count(count),
                                         generation(0) {}

  /// Wait at the barrier until all threads arrive.
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    auto gen = generation;
    if (--count == 0) {
      generation++;
      count = threshold;
      cv.notify_all();
    } else {
      cv.wait(lock, [this, gen] { return gen != generation; });
    }
  }

  /// Reset the barrier for a new count.
  void reset(size_t newCount) {
    std::unique_lock<std::mutex> lock(mutex);
    threshold = newCount;
    count = newCount;
    generation++;
  }

private:
  std::mutex mutex;
  std::condition_variable cv;
  size_t threshold;
  size_t count;
  size_t generation;
};

//===----------------------------------------------------------------------===//
// WorkStealingQueue - Lock-free work stealing queue
//===----------------------------------------------------------------------===//

/// A work stealing queue for dynamic load balancing.
/// Uses atomic operations for lock-free push/pop from the owning thread.
template <typename T>
class WorkStealingQueue {
public:
  WorkStealingQueue(size_t capacity = 1024)
      : buffer(capacity), mask(capacity - 1), top(0), bottom(0) {
    // Ensure capacity is a power of 2
    assert((capacity & (capacity - 1)) == 0 && "Capacity must be power of 2");
  }

  /// Push a work item (called by the owning thread).
  bool push(T item) {
    auto b = bottom.load(std::memory_order_relaxed);
    auto t = top.load(std::memory_order_acquire);

    if (b - t >= buffer.size()) {
      // Queue is full
      return false;
    }

    buffer[b & mask] = std::move(item);
    bottom.store(b + 1, std::memory_order_release);
    return true;
  }

  /// Pop a work item (called by the owning thread).
  bool pop(T &item) {
    auto b = bottom.load(std::memory_order_relaxed) - 1;
    bottom.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto t = top.load(std::memory_order_relaxed);

    if (t <= b) {
      item = std::move(buffer[b & mask]);
      if (t == b) {
        // Last item, try to claim it
        if (!top.compare_exchange_strong(t, t + 1,
                                         std::memory_order_seq_cst,
                                         std::memory_order_relaxed)) {
          // Lost race with steal
          bottom.store(b + 1, std::memory_order_relaxed);
          return false;
        }
        bottom.store(b + 1, std::memory_order_relaxed);
      }
      return true;
    } else {
      bottom.store(b + 1, std::memory_order_relaxed);
      return false;
    }
  }

  /// Steal a work item (called by other threads).
  bool steal(T &item) {
    auto t = top.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto b = bottom.load(std::memory_order_acquire);

    if (t < b) {
      item = buffer[t & mask];
      if (!top.compare_exchange_strong(t, t + 1,
                                       std::memory_order_seq_cst,
                                       std::memory_order_relaxed)) {
        // Lost race with another steal or pop
        return false;
      }
      return true;
    }
    return false;
  }

  /// Check if the queue is empty.
  bool empty() const {
    auto b = bottom.load(std::memory_order_relaxed);
    auto t = top.load(std::memory_order_relaxed);
    return b <= t;
  }

  /// Get approximate size.
  size_t size() const {
    auto b = bottom.load(std::memory_order_relaxed);
    auto t = top.load(std::memory_order_relaxed);
    return b > t ? b - t : 0;
  }

private:
  std::vector<T> buffer;
  size_t mask;
  std::atomic<size_t> top;
  std::atomic<size_t> bottom;
};

//===----------------------------------------------------------------------===//
// ParallelScheduler - Multi-threaded simulation scheduler
//===----------------------------------------------------------------------===//

/// The main parallel scheduler that coordinates multi-threaded simulation.
/// Uses partition-based parallelism with barrier synchronization.
class ParallelScheduler {
public:
  /// Configuration for the parallel scheduler.
  struct Config {
    /// Number of worker threads (0 = auto-detect based on hardware).
    size_t numThreads;

    /// Minimum number of processes per partition.
    size_t minProcessesPerPartition;

    /// Maximum number of partitions.
    size_t maxPartitions;

    /// Enable work stealing between threads.
    bool enableWorkStealing;

    /// Enable dynamic load balancing.
    bool enableDynamicBalancing;

    /// Debug output level (0 = none, 1 = basic, 2 = verbose).
    int debugLevel;

    Config()
        : numThreads(0), minProcessesPerPartition(10), maxPartitions(64),
          enableWorkStealing(true), enableDynamicBalancing(true),
          debugLevel(0) {}
  };

  ParallelScheduler(ProcessScheduler &baseScheduler, Config config = Config());
  ~ParallelScheduler();

  //===------------------------------------------------------------------===//
  // Partition Management
  //===------------------------------------------------------------------===//

  /// Create a new partition.
  PartitionId createPartition(const std::string &name);

  /// Get a partition by ID.
  Partition *getPartition(PartitionId id);
  const Partition *getPartition(PartitionId id) const;

  /// Assign a process to a partition.
  void assignProcess(ProcessId processId, PartitionId partitionId);

  /// Assign a signal to a partition (as internal).
  void assignSignal(SignalId signalId, PartitionId partitionId);

  /// Declare a signal as a boundary signal between partitions.
  void declareBoundarySignal(SignalId signalId, PartitionId source,
                             const llvm::SmallVector<PartitionId, 4> &dests);

  /// Get the partition for a process.
  PartitionId getPartitionForProcess(ProcessId id) const;

  /// Get the partition for a signal.
  PartitionId getPartitionForSignal(SignalId id) const;

  /// Auto-partition the design based on signal dependencies.
  void autoPartition();

  /// Get the number of partitions.
  size_t getPartitionCount() const { return partitions.size(); }

  /// Get all partitions.
  const std::vector<std::unique_ptr<Partition>> &getPartitions() const {
    return partitions;
  }

  //===------------------------------------------------------------------===//
  // Parallel Execution
  //===------------------------------------------------------------------===//

  /// Start the worker threads.
  void startWorkers();

  /// Stop the worker threads.
  void stopWorkers();

  /// Check if workers are running.
  bool isRunning() const { return running.load(); }

  /// Execute one parallel delta cycle.
  /// Returns true if any work was done.
  bool executeParallelDeltaCycle();

  /// Execute all delta cycles at the current time in parallel.
  /// Returns the number of delta cycles executed.
  size_t executeCurrentTimeParallel();

  /// Synchronize boundary signals between partitions.
  void synchronizeBoundaries();

  /// Run the parallel simulation until completion or time limit.
  SimTime runParallel(uint64_t maxTimeFemtoseconds);

  //===------------------------------------------------------------------===//
  // Work Distribution
  //===------------------------------------------------------------------===//

  /// Distribute work items to worker threads.
  void distributeWork();

  /// Steal work from another thread (for load balancing).
  bool stealWork(size_t thiefThread);

  //===------------------------------------------------------------------===//
  // Statistics
  //===------------------------------------------------------------------===//

  struct Statistics {
    std::atomic<size_t> totalDeltaCycles{0};
    std::atomic<size_t> parallelDeltaCycles{0};
    std::atomic<size_t> sequentialDeltaCycles{0};
    std::atomic<size_t> boundarySync{0};
    std::atomic<size_t> workSteals{0};
    std::atomic<size_t> barrierWaits{0};
    std::atomic<uint64_t> totalExecutionTimeNs{0};
    std::atomic<uint64_t> syncOverheadNs{0};
  };

  const Statistics &getStatistics() const { return stats; }

  /// Get the partition graph for analysis.
  const PartitionGraph &getPartitionGraph() const { return partitionGraph; }

  /// Get the actual number of threads being used.
  size_t getNumThreads() const { return numThreads; }

  /// Print partition statistics to the given stream.
  void printStatistics(llvm::raw_ostream &os) const;

private:
  //===------------------------------------------------------------------===//
  // Worker Thread Implementation
  //===------------------------------------------------------------------===//

  /// Worker thread main function.
  void workerMain(size_t threadId);

  /// Execute work items for a specific partition.
  void executePartition(PartitionId id);

  /// Execute ready processes in a partition.
  size_t executePartitionProcesses(Partition &partition);

  ProcessScheduler &baseScheduler;
  Config config;
  size_t numThreads;

  // Partition management
  std::vector<std::unique_ptr<Partition>> partitions;
  llvm::DenseMap<ProcessId, PartitionId> processToPartition;
  llvm::DenseMap<SignalId, PartitionId> signalToPartition;
  llvm::DenseMap<SignalId, std::unique_ptr<BoundarySignal>> boundarySignals;
  PartitionGraph partitionGraph;

  // Thread management
  std::vector<std::thread> workers;
  std::vector<std::unique_ptr<WorkStealingQueue<PartitionId>>> workQueues;
  std::unique_ptr<ThreadBarrier> barrier;
  std::atomic<bool> running;
  /// Monotonic work-dispatch generation. Workers run at most once per epoch.
  std::atomic<uint64_t> workEpoch;
  std::atomic<size_t> activeWorkers;

  // Synchronization
  std::mutex distributionMutex;
  std::condition_variable workCondition;

  // Statistics
  Statistics stats;

  // Thread-local state (indexed by thread ID)
  struct ThreadLocalState {
    size_t eventsProcessed = 0;
    uint64_t lastExecutionNs = 0;
  };
  std::vector<ThreadLocalState> threadStates;
};

//===----------------------------------------------------------------------===//
// PartitionBalancer - Utilities for balancing partition loads
//===----------------------------------------------------------------------===//

/// Utilities for analyzing and balancing partition loads.
class PartitionBalancer {
public:
  /// Calculate load imbalance ratio (max/avg load).
  static double calculateImbalance(
      const std::vector<std::unique_ptr<Partition>> &partitions);

  /// Suggest process moves to balance loads.
  /// Returns pairs of (processId, targetPartition).
  static std::vector<std::pair<ProcessId, PartitionId>>
  suggestMoves(const std::vector<std::unique_ptr<Partition>> &partitions,
               const PartitionGraph &graph, double targetImbalance = 1.2);

  /// Check if rebalancing is needed.
  static bool needsRebalancing(
      const std::vector<std::unique_ptr<Partition>> &partitions,
      double threshold = 1.5);
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_PARALLELSCHEDULER_H
