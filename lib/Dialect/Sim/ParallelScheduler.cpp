//===- ParallelScheduler.cpp - Multi-core parallel simulation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parallel scheduler infrastructure for multi-core
// simulation with partition-based parallelism.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ParallelScheduler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <chrono>
#include <random>

#define DEBUG_TYPE "sim-parallel-scheduler"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// ParallelScheduler Implementation
//===----------------------------------------------------------------------===//

ParallelScheduler::ParallelScheduler(ProcessScheduler &baseScheduler,
                                     Config config)
    : baseScheduler(baseScheduler), config(config), running(false),
      workAvailable(false), activeWorkers(0) {
  // Determine number of threads
  if (config.numThreads == 0) {
    numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0)
      numThreads = 4; // Fallback if detection fails
  } else {
    numThreads = config.numThreads;
  }

  // Create work queues for each thread
  workQueues.resize(numThreads);
  for (size_t i = 0; i < numThreads; ++i) {
    workQueues[i] = std::make_unique<WorkStealingQueue<PartitionId>>(1024);
  }

  // Create barrier for thread synchronization
  barrier = std::make_unique<ThreadBarrier>(numThreads);

  // Initialize thread-local state
  threadStates.resize(numThreads);

  LLVM_DEBUG(llvm::dbgs() << "ParallelScheduler created with " << numThreads
                          << " threads\n");
}

ParallelScheduler::~ParallelScheduler() { stopWorkers(); }

//===----------------------------------------------------------------------===//
// Partition Management
//===----------------------------------------------------------------------===//

PartitionId ParallelScheduler::createPartition(const std::string &name) {
  if (partitions.size() >= config.maxPartitions) {
    return InvalidPartitionId;
  }

  PartitionId id = static_cast<PartitionId>(partitions.size());
  partitions.push_back(std::make_unique<Partition>(id, name));

  LLVM_DEBUG(llvm::dbgs() << "Created partition " << id << ": " << name
                          << "\n");
  return id;
}

Partition *ParallelScheduler::getPartition(PartitionId id) {
  if (id >= partitions.size())
    return nullptr;
  return partitions[id].get();
}

const Partition *ParallelScheduler::getPartition(PartitionId id) const {
  if (id >= partitions.size())
    return nullptr;
  return partitions[id].get();
}

void ParallelScheduler::assignProcess(ProcessId processId,
                                      PartitionId partitionId) {
  if (partitionId >= partitions.size())
    return;

  // Remove from old partition if already assigned
  auto it = processToPartition.find(processId);
  if (it != processToPartition.end()) {
    auto &oldProcs = partitions[it->second]->getProcesses();
    // Note: This is a simplified removal; actual implementation would need
    // mutable access
  }

  processToPartition[processId] = partitionId;
  partitions[partitionId]->addProcess(processId);

  LLVM_DEBUG(llvm::dbgs() << "Assigned process " << processId << " to partition "
                          << partitionId << "\n");
}

void ParallelScheduler::assignSignal(SignalId signalId,
                                     PartitionId partitionId) {
  if (partitionId >= partitions.size())
    return;

  signalToPartition[signalId] = partitionId;
  partitions[partitionId]->addInternalSignal(signalId);
}

void ParallelScheduler::declareBoundarySignal(
    SignalId signalId, PartitionId source,
    const llvm::SmallVector<PartitionId, 4> &dests) {
  auto boundary = std::make_unique<BoundarySignal>(signalId, source);
  boundary->destPartitions = dests;

  // Add to partition boundary lists
  if (source < partitions.size()) {
    partitions[source]->addOutputBoundary(signalId);
  }
  for (PartitionId dest : dests) {
    if (dest < partitions.size()) {
      partitions[dest]->addInputBoundary(signalId);
      partitionGraph.addEdge(source, dest, signalId);
    }
  }

  boundarySignals[signalId] = std::move(boundary);

  LLVM_DEBUG(llvm::dbgs() << "Declared boundary signal " << signalId
                          << " from partition " << source << "\n");
}

PartitionId ParallelScheduler::getPartitionForProcess(ProcessId id) const {
  auto it = processToPartition.find(id);
  return it != processToPartition.end() ? it->second : InvalidPartitionId;
}

PartitionId ParallelScheduler::getPartitionForSignal(SignalId id) const {
  auto it = signalToPartition.find(id);
  return it != signalToPartition.end() ? it->second : InvalidPartitionId;
}

void ParallelScheduler::autoPartition() {
  // Get all processes from the base scheduler
  const auto &processes = baseScheduler.getProcesses();
  if (processes.empty())
    return;

  // Calculate target partition size
  size_t totalProcesses = processes.size();
  size_t targetPartitions =
      std::min(numThreads, totalProcesses / config.minProcessesPerPartition);
  targetPartitions = std::max(targetPartitions, size_t(1));
  targetPartitions = std::min(targetPartitions, config.maxPartitions);

  // Clear existing partitions
  partitions.clear();
  processToPartition.clear();
  partitionGraph.clear();

  // Create partitions
  for (size_t i = 0; i < targetPartitions; ++i) {
    createPartition("partition_" + std::to_string(i));
  }

  // Simple round-robin assignment (a more sophisticated algorithm would
  // analyze signal dependencies and minimize cut edges)
  size_t idx = 0;
  for (const auto &kv : processes) {
    ProcessId pid = kv.first;
    PartitionId partId = static_cast<PartitionId>(idx % targetPartitions);
    assignProcess(pid, partId);
    ++idx;
  }

  LLVM_DEBUG(llvm::dbgs() << "Auto-partitioned " << totalProcesses
                          << " processes into " << targetPartitions
                          << " partitions\n");
}

//===----------------------------------------------------------------------===//
// Parallel Execution
//===----------------------------------------------------------------------===//

void ParallelScheduler::startWorkers() {
  if (running.load())
    return;

  running.store(true);
  workers.clear();
  workers.reserve(numThreads);

  for (size_t i = 0; i < numThreads; ++i) {
    workers.emplace_back(&ParallelScheduler::workerMain, this, i);
  }

  LLVM_DEBUG(llvm::dbgs() << "Started " << numThreads << " worker threads\n");
}

void ParallelScheduler::stopWorkers() {
  if (!running.load())
    return;

  running.store(false);
  workCondition.notify_all();

  for (auto &worker : workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers.clear();

  LLVM_DEBUG(llvm::dbgs() << "Stopped worker threads\n");
}

void ParallelScheduler::workerMain(size_t threadId) {
  LLVM_DEBUG(llvm::dbgs() << "Worker " << threadId << " started\n");

  while (running.load()) {
    // Wait for work to be available
    {
      std::unique_lock<std::mutex> lock(distributionMutex);
      workCondition.wait(lock, [this] {
        return !running.load() || workAvailable.load();
      });
    }

    if (!running.load())
      break;

    activeWorkers.fetch_add(1);

    auto startTime = std::chrono::high_resolution_clock::now();

    // Process work from own queue
    PartitionId partId;
    while (workQueues[threadId]->pop(partId)) {
      executePartition(partId);
      threadStates[threadId].eventsProcessed++;
    }

    // Try to steal work if work stealing is enabled
    if (config.enableWorkStealing) {
      if (stealWork(threadId)) {
        stats.workSteals.fetch_add(1);
      }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        endTime - startTime);
    threadStates[threadId].lastExecutionNs = duration.count();

    activeWorkers.fetch_sub(1);

    // Wait at barrier for synchronization
    barrier->wait();
    stats.barrierWaits.fetch_add(1);
  }

  LLVM_DEBUG(llvm::dbgs() << "Worker " << threadId << " exiting\n");
}

void ParallelScheduler::executePartition(PartitionId id) {
  Partition *partition = getPartition(id);
  if (!partition || !partition->isActive())
    return;

  auto startTime = std::chrono::high_resolution_clock::now();

  // Read boundary inputs
  for (SignalId sigId : partition->getInputBoundarySignals()) {
    auto it = boundarySignals.find(sigId);
    if (it != boundarySignals.end()) {
      SignalValue value = it->second->read();
      baseScheduler.updateSignal(sigId, value);
      partition->getStatistics().boundaryReads.fetch_add(1);
    }
  }

  // Execute processes in this partition
  size_t eventsProcessed = executePartitionProcesses(*partition);

  // Write boundary outputs
  for (SignalId sigId : partition->getOutputBoundarySignals()) {
    auto it = boundarySignals.find(sigId);
    if (it != boundarySignals.end()) {
      const SignalValue &value = baseScheduler.getSignalValue(sigId);
      it->second->write(value);
      partition->getStatistics().boundaryWrites.fetch_add(1);
    }
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      endTime - startTime);

  partition->getStatistics().eventsProcessed.fetch_add(eventsProcessed);
  partition->getStatistics().executionTimeNs.fetch_add(duration.count());
}

size_t ParallelScheduler::executePartitionProcesses(Partition &partition) {
  size_t eventsProcessed = 0;

  for (ProcessId pid : partition.getProcesses()) {
    if (baseScheduler.isAbortRequested())
      break;
    Process *process = baseScheduler.getProcess(pid);
    if (!process)
      continue;

    if (process->getState() == ProcessState::Ready) {
      process->setState(ProcessState::Running);
      process->execute();
      eventsProcessed++;

      // Check if process should be rescheduled
      if (process->getState() == ProcessState::Running) {
        process->setState(ProcessState::Ready);
      }
    }
  }

  return eventsProcessed;
}

bool ParallelScheduler::executeParallelDeltaCycle() {
  if (baseScheduler.isAbortRequested())
    return false;
  stats.totalDeltaCycles.fetch_add(1);

  // Activate partitions that have work
  bool anyWork = false;
  for (auto &partition : partitions) {
    bool hasWork = false;
    for (ProcessId pid : partition->getProcesses()) {
      Process *process = baseScheduler.getProcess(pid);
      if (process && process->getState() == ProcessState::Ready) {
        hasWork = true;
        break;
      }
    }
    partition->setActive(hasWork);
    anyWork = anyWork || hasWork;
  }

  if (!anyWork) {
    return false;
  }

  // Distribute work to queues
  distributeWork();

  // Signal workers
  {
    std::lock_guard<std::mutex> lock(distributionMutex);
    workAvailable.store(true);
  }
  workCondition.notify_all();

  // Wait for workers to complete
  barrier->wait();

  // Synchronize boundary signals
  synchronizeBoundaries();

  workAvailable.store(false);
  stats.parallelDeltaCycles.fetch_add(1);

  return true;
}

size_t ParallelScheduler::executeCurrentTimeParallel() {
  if (baseScheduler.isAbortRequested())
    return 0;
  size_t deltaCycles = 0;
  while (executeParallelDeltaCycle()) {
    deltaCycles++;
  }
  return deltaCycles;
}

void ParallelScheduler::synchronizeBoundaries() {
  auto startTime = std::chrono::high_resolution_clock::now();

  // Boundary synchronization is done in executePartition by reading
  // and writing boundary signals. This method performs any additional
  // global synchronization if needed.

  stats.boundarySync.fetch_add(1);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      endTime - startTime);
  stats.syncOverheadNs.fetch_add(duration.count());
}

SimTime ParallelScheduler::runParallel(uint64_t maxTimeFemtoseconds) {
  if (!running.load()) {
    startWorkers();
  }

  auto &eventScheduler = baseScheduler.getEventScheduler();
  const SimTime &currentTime = eventScheduler.getCurrentTime();

  while (currentTime.realTime < maxTimeFemtoseconds) {
    // Execute all delta cycles at current time
    executeCurrentTimeParallel();

    // Advance to next event time
    if (!baseScheduler.advanceTime()) {
      break; // No more events
    }
  }

  return baseScheduler.getCurrentTime();
}

void ParallelScheduler::distributeWork() {
  // Round-robin distribution of active partitions to thread queues
  size_t threadIdx = 0;
  for (auto &partition : partitions) {
    if (partition->isActive()) {
      workQueues[threadIdx]->push(partition->getId());
      threadIdx = (threadIdx + 1) % numThreads;
    }
  }
}

bool ParallelScheduler::stealWork(size_t thiefThread) {
  // Try to steal from a random victim
  static thread_local std::mt19937 rng(
      std::chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<size_t> dist(0, numThreads - 1);

  for (size_t attempts = 0; attempts < numThreads; ++attempts) {
    size_t victim = dist(rng);
    if (victim == thiefThread)
      continue;

    PartitionId stolen;
    if (workQueues[victim]->steal(stolen)) {
      executePartition(stolen);
      return true;
    }
  }

  return false;
}

void ParallelScheduler::printStatistics(llvm::raw_ostream &os) const {
  os << "=== ParallelScheduler Statistics ===\n";
  os << "Threads: " << numThreads << "\n";
  os << "Partitions: " << partitions.size() << "\n";
  os << "Boundary signals: " << boundarySignals.size() << "\n";
  os << "Cut size: " << partitionGraph.getCutSize() << "\n";
  os << "\n";

  os << "Delta cycles:\n";
  os << "  Total: " << stats.totalDeltaCycles.load() << "\n";
  os << "  Parallel: " << stats.parallelDeltaCycles.load() << "\n";
  os << "  Sequential: " << stats.sequentialDeltaCycles.load() << "\n";
  os << "\n";

  os << "Synchronization:\n";
  os << "  Boundary syncs: " << stats.boundarySync.load() << "\n";
  os << "  Barrier waits: " << stats.barrierWaits.load() << "\n";
  os << "  Work steals: " << stats.workSteals.load() << "\n";
  os << "\n";

  os << "Timing:\n";
  os << "  Total execution: " << stats.totalExecutionTimeNs.load() / 1e6
     << " ms\n";
  os << "  Sync overhead: " << stats.syncOverheadNs.load() / 1e6 << " ms\n";
  os << "\n";

  os << "Per-partition statistics:\n";
  for (const auto &partition : partitions) {
    const auto &pstats = partition->getStatistics();
    os << "  Partition " << partition->getId() << " (" << partition->getName()
       << "):\n";
    os << "    Processes: " << partition->getProcessCount() << "\n";
    os << "    Events: " << pstats.eventsProcessed.load() << "\n";
    os << "    Boundary reads: " << pstats.boundaryReads.load() << "\n";
    os << "    Boundary writes: " << pstats.boundaryWrites.load() << "\n";
    os << "    Execution time: " << pstats.executionTimeNs.load() / 1e6
       << " ms\n";
  }
}

//===----------------------------------------------------------------------===//
// PartitionBalancer Implementation
//===----------------------------------------------------------------------===//

double PartitionBalancer::calculateImbalance(
    const std::vector<std::unique_ptr<Partition>> &partitions) {
  if (partitions.empty())
    return 1.0;

  double totalLoad = 0;
  double maxLoad = 0;
  for (const auto &partition : partitions) {
    double load = partition->getLoad();
    totalLoad += load;
    maxLoad = std::max(maxLoad, load);
  }

  double avgLoad = totalLoad / partitions.size();
  return avgLoad > 0 ? maxLoad / avgLoad : 1.0;
}

std::vector<std::pair<ProcessId, PartitionId>>
PartitionBalancer::suggestMoves(
    const std::vector<std::unique_ptr<Partition>> &partitions,
    const PartitionGraph &graph, double targetImbalance) {
  std::vector<std::pair<ProcessId, PartitionId>> moves;

  if (partitions.size() < 2)
    return moves;

  // Find overloaded and underloaded partitions
  double totalLoad = 0;
  for (const auto &partition : partitions) {
    totalLoad += partition->getLoad();
  }
  double avgLoad = totalLoad / partitions.size();
  double threshold = avgLoad * targetImbalance;

  // Find partitions above threshold
  std::vector<PartitionId> overloaded;
  std::vector<PartitionId> underloaded;

  for (const auto &partition : partitions) {
    if (partition->getLoad() > threshold) {
      overloaded.push_back(partition->getId());
    } else if (partition->getLoad() < avgLoad * 0.8) {
      underloaded.push_back(partition->getId());
    }
  }

  // Suggest moving processes from overloaded to underloaded
  for (PartitionId srcId : overloaded) {
    if (underloaded.empty())
      break;

    const auto &srcProcesses = partitions[srcId]->getProcesses();
    if (srcProcesses.empty())
      continue;

    // Move first process to least loaded underloaded partition
    ProcessId toMove = srcProcesses.front();
    PartitionId destId = underloaded.front();
    moves.push_back({toMove, destId});
  }

  return moves;
}

bool PartitionBalancer::needsRebalancing(
    const std::vector<std::unique_ptr<Partition>> &partitions,
    double threshold) {
  return calculateImbalance(partitions) > threshold;
}
