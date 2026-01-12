//===- DesignPartitioner.cpp - Design partitioning analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the design partitioner for parallel simulation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/DesignPartitioner.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <queue>
#include <random>

#define DEBUG_TYPE "sim-partitioner"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// RoundRobinStrategy Implementation
//===----------------------------------------------------------------------===//

std::vector<PartitionId>
RoundRobinStrategy::partition(const DependencyGraph &graph,
                              size_t numPartitions) {
  std::vector<PartitionId> assignment(graph.getNodeCount());

  size_t processIdx = 0;
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    const auto &node = graph.getNode(i);
    if (node.isProcess()) {
      assignment[i] = static_cast<PartitionId>(processIdx % numPartitions);
      processIdx++;
    } else {
      // Signals are assigned with their primary writing process
      assignment[i] = InvalidPartitionId;
    }
  }

  return assignment;
}

//===----------------------------------------------------------------------===//
// BFSStrategy Implementation
//===----------------------------------------------------------------------===//

std::vector<PartitionId>
BFSStrategy::partition(const DependencyGraph &graph, size_t numPartitions) {
  std::vector<PartitionId> assignment(graph.getNodeCount(), InvalidPartitionId);
  std::vector<bool> visited(graph.getNodeCount(), false);

  // Find seed nodes (processes with most connections)
  std::vector<std::pair<size_t, size_t>> nodeConnectivity;
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (graph.getNode(i).isProcess()) {
      size_t connectivity =
          graph.getOutEdges(i).size() + graph.getInEdges(i).size();
      nodeConnectivity.push_back({i, connectivity});
    }
  }

  // Sort by connectivity (descending)
  std::sort(nodeConnectivity.begin(), nodeConnectivity.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  // Target size per partition
  size_t processCount = nodeConnectivity.size();
  size_t targetSize = (processCount + numPartitions - 1) / numPartitions;

  std::vector<size_t> partitionSizes(numPartitions, 0);
  PartitionId currentPartition = 0;

  // BFS from each seed
  for (const auto &[seed, _] : nodeConnectivity) {
    if (visited[seed])
      continue;

    // Find partition with room
    while (currentPartition < numPartitions &&
           partitionSizes[currentPartition] >= targetSize) {
      currentPartition++;
    }
    if (currentPartition >= numPartitions)
      currentPartition = 0;

    // BFS from this seed
    std::queue<size_t> queue;
    queue.push(seed);
    visited[seed] = true;

    while (!queue.empty() &&
           partitionSizes[currentPartition] < targetSize * 1.2) {
      size_t node = queue.front();
      queue.pop();

      if (graph.getNode(node).isProcess()) {
        assignment[node] = currentPartition;
        partitionSizes[currentPartition]++;
      }

      // Add neighbors
      for (size_t edgeIdx : graph.getOutEdges(node)) {
        size_t neighbor = graph.getEdge(edgeIdx).destNode;
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
      for (size_t edgeIdx : graph.getInEdges(node)) {
        size_t neighbor = graph.getEdge(edgeIdx).sourceNode;
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          queue.push(neighbor);
        }
      }
    }
  }

  // Assign any unassigned nodes
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (assignment[i] == InvalidPartitionId && graph.getNode(i).isProcess()) {
      // Find partition with smallest size
      PartitionId minPart = 0;
      for (PartitionId p = 1; p < numPartitions; ++p) {
        if (partitionSizes[p] < partitionSizes[minPart])
          minPart = p;
      }
      assignment[i] = minPart;
      partitionSizes[minPart]++;
    }
  }

  return assignment;
}

//===----------------------------------------------------------------------===//
// FMStrategy Implementation
//===----------------------------------------------------------------------===//

std::vector<PartitionId>
FMStrategy::partition(const DependencyGraph &graph, size_t numPartitions) {
  // Start with BFS for initial partition
  BFSStrategy bfs;
  std::vector<PartitionId> assignment = bfs.partition(graph, numPartitions);

  // Iteratively improve
  for (size_t pass = 0; pass < config.maxPasses; ++pass) {
    if (!performPass(graph, assignment, numPartitions))
      break;
  }

  return assignment;
}

int FMStrategy::calculateGain(const DependencyGraph &graph,
                              const std::vector<PartitionId> &assignment,
                              size_t node, PartitionId targetPartition) {
  if (!graph.getNode(node).isProcess())
    return 0;

  PartitionId currentPartition = assignment[node];
  if (currentPartition == targetPartition)
    return 0;

  int gain = 0;

  // Check outgoing edges
  for (size_t edgeIdx : graph.getOutEdges(node)) {
    const auto &edge = graph.getEdge(edgeIdx);
    PartitionId neighborPart = assignment[edge.destNode];
    if (neighborPart == currentPartition) {
      // Moving would cut this edge
      gain -= static_cast<int>(edge.weight);
    } else if (neighborPart == targetPartition) {
      // Moving would merge this edge
      gain += static_cast<int>(edge.weight);
    }
  }

  // Check incoming edges
  for (size_t edgeIdx : graph.getInEdges(node)) {
    const auto &edge = graph.getEdge(edgeIdx);
    PartitionId neighborPart = assignment[edge.sourceNode];
    if (neighborPart == currentPartition) {
      gain -= static_cast<int>(edge.weight);
    } else if (neighborPart == targetPartition) {
      gain += static_cast<int>(edge.weight);
    }
  }

  return gain;
}

bool FMStrategy::performPass(const DependencyGraph &graph,
                             std::vector<PartitionId> &assignment,
                             size_t numPartitions) {
  // Track partition sizes
  std::vector<size_t> partitionSizes(numPartitions, 0);
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (graph.getNode(i).isProcess() && assignment[i] < numPartitions) {
      partitionSizes[assignment[i]]++;
    }
  }

  size_t totalNodes =
      std::accumulate(partitionSizes.begin(), partitionSizes.end(), size_t(0));
  double avgSize = static_cast<double>(totalNodes) / numPartitions;
  double minSize = avgSize / config.balanceTolerance;
  double maxSize = avgSize * config.balanceTolerance;

  std::vector<bool> locked(graph.getNodeCount(), false);
  bool improved = false;

  // Find best moves until no improvement
  while (true) {
    int bestGain = 0;
    size_t bestNode = SIZE_MAX;
    PartitionId bestTarget = InvalidPartitionId;

    // Find the best move
    for (size_t i = 0; i < graph.getNodeCount(); ++i) {
      if (locked[i] || !graph.getNode(i).isProcess())
        continue;
      if (assignment[i] >= numPartitions)
        continue;

      PartitionId currentPart = assignment[i];
      if (partitionSizes[currentPart] <= minSize)
        continue; // Don't move from small partitions

      for (PartitionId target = 0; target < numPartitions; ++target) {
        if (target == currentPart)
          continue;
        if (partitionSizes[target] >= maxSize)
          continue; // Don't move to large partitions

        int gain = calculateGain(graph, assignment, i, target);
        if (gain > bestGain) {
          bestGain = gain;
          bestNode = i;
          bestTarget = target;
        }
      }
    }

    if (bestGain <= 0)
      break;

    // Apply the move
    PartitionId oldPart = assignment[bestNode];
    assignment[bestNode] = bestTarget;
    partitionSizes[oldPart]--;
    partitionSizes[bestTarget]++;
    locked[bestNode] = true;
    improved = true;

    LLVM_DEBUG(llvm::dbgs() << "FM: Moved node " << bestNode << " from partition "
                            << oldPart << " to " << bestTarget << " (gain "
                            << bestGain << ")\n");
  }

  return improved;
}

//===----------------------------------------------------------------------===//
// KLStrategy Implementation
//===----------------------------------------------------------------------===//

std::vector<PartitionId>
KLStrategy::partition(const DependencyGraph &graph, size_t numPartitions) {
  // For simplicity, use FM for now (KL is similar but swaps pairs)
  FMStrategy fm;
  return fm.partition(graph, numPartitions);
}

//===----------------------------------------------------------------------===//
// SpectralStrategy Implementation
//===----------------------------------------------------------------------===//

std::vector<PartitionId>
SpectralStrategy::partition(const DependencyGraph &graph,
                            size_t numPartitions) {
  if (graph.getNodeCount() == 0)
    return {};

  // Compute Fiedler vector
  std::vector<double> fiedler = computeFiedlerVector(graph);

  // Get process nodes and their Fiedler values
  std::vector<std::pair<size_t, double>> processValues;
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (graph.getNode(i).isProcess()) {
      processValues.push_back({i, fiedler[i]});
    }
  }

  // Sort by Fiedler value
  std::sort(processValues.begin(), processValues.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

  // Assign to partitions based on sorted order
  std::vector<PartitionId> assignment(graph.getNodeCount(), InvalidPartitionId);
  size_t processesPerPartition =
      (processValues.size() + numPartitions - 1) / numPartitions;

  for (size_t i = 0; i < processValues.size(); ++i) {
    size_t nodeIdx = processValues[i].first;
    PartitionId partition =
        static_cast<PartitionId>(i / processesPerPartition);
    partition = std::min(partition, static_cast<PartitionId>(numPartitions - 1));
    assignment[nodeIdx] = partition;
  }

  return assignment;
}

std::vector<double>
SpectralStrategy::computeFiedlerVector(const DependencyGraph &graph) {
  size_t n = graph.getNodeCount();
  if (n == 0)
    return {};

  // Initialize random vector
  std::vector<double> v(n);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dist(rng);
  }

  // Normalize
  double norm = 0;
  for (double x : v)
    norm += x * x;
  norm = std::sqrt(norm);
  for (double &x : v)
    x /= norm;

  // Compute degrees
  std::vector<double> degrees(n, 0);
  for (size_t i = 0; i < n; ++i) {
    degrees[i] = static_cast<double>(graph.getOutEdges(i).size() +
                                     graph.getInEdges(i).size());
  }

  // Power iteration to find Fiedler vector
  std::vector<double> Lv(n);
  for (size_t iter = 0; iter < config.powerIterations; ++iter) {
    // Compute L * v (Laplacian times v)
    for (size_t i = 0; i < n; ++i) {
      Lv[i] = degrees[i] * v[i];
      for (size_t edgeIdx : graph.getOutEdges(i)) {
        size_t j = graph.getEdge(edgeIdx).destNode;
        Lv[i] -= v[j];
      }
      for (size_t edgeIdx : graph.getInEdges(i)) {
        size_t j = graph.getEdge(edgeIdx).sourceNode;
        Lv[i] -= v[j];
      }
    }

    // Deflate against constant vector (first eigenvector)
    double mean = 0;
    for (double x : Lv)
      mean += x;
    mean /= n;
    for (double &x : Lv)
      x -= mean;

    // Normalize
    norm = 0;
    for (double x : Lv)
      norm += x * x;
    norm = std::sqrt(norm);
    if (norm < config.tolerance)
      break;

    for (size_t i = 0; i < n; ++i) {
      v[i] = Lv[i] / norm;
    }
  }

  return v;
}

//===----------------------------------------------------------------------===//
// DesignPartitioner Implementation
//===----------------------------------------------------------------------===//

DesignPartitioner::DesignPartitioner(ProcessScheduler &scheduler, Config config)
    : scheduler(scheduler), config(config) {}

DesignPartitioner::~DesignPartitioner() = default;

void DesignPartitioner::buildDependencyGraph() {
  graph.clear();

  // Add process nodes
  const auto &processes = scheduler.getProcesses();
  for (const auto &kv : processes) {
    graph.addProcessNode(kv.first);
    stats.nodesAnalyzed++;
  }

  // Analyze dependencies through sensitivity lists
  for (const auto &kv : processes) {
    ProcessId pid = kv.first;
    const Process *process = kv.second.get();
    if (!process)
      continue;

    size_t procNode = graph.getProcessNode(pid);
    const SensitivityList &sensitivity = process->getSensitivityList();

    // Add edges from signals this process is sensitive to
    for (const auto &entry : sensitivity.getEntries()) {
      SignalId sigId = entry.signalId;

      // Find or create signal node
      size_t sigNode = graph.getSignalNode(sigId);
      if (sigNode == SIZE_MAX) {
        sigNode = graph.addSignalNode(sigId);
        stats.nodesAnalyzed++;
      }

      // Add edge from signal to process (dependency)
      graph.addEdge(sigNode, procNode);
      stats.edgesAnalyzed++;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Built dependency graph: " << graph.getNodeCount()
                          << " nodes, " << graph.getEdgeCount() << " edges\n");
}

void DesignPartitioner::analyzeProcessDependencies(ProcessId pid) {
  // This is a placeholder for more detailed per-process analysis
  // A full implementation would trace signal reads/writes during execution
}

std::unique_ptr<PartitioningStrategy> DesignPartitioner::createStrategy() {
  switch (config.strategy) {
  case Config::Strategy::RoundRobin:
    return std::make_unique<RoundRobinStrategy>();
  case Config::Strategy::BFS:
    return std::make_unique<BFSStrategy>();
  case Config::Strategy::FM:
    return std::make_unique<FMStrategy>();
  case Config::Strategy::KL:
    return std::make_unique<KLStrategy>();
  case Config::Strategy::Spectral:
    return std::make_unique<SpectralStrategy>();
  }
  return std::make_unique<FMStrategy>();
}

void DesignPartitioner::partition() {
  auto startTime = std::chrono::high_resolution_clock::now();

  // Determine target number of partitions
  size_t targetPartitions = config.targetPartitions;
  if (targetPartitions == 0) {
    targetPartitions = std::thread::hardware_concurrency();
    if (targetPartitions == 0)
      targetPartitions = 4;
  }

  // Ensure we don't have more partitions than processes
  size_t processCount = 0;
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (graph.getNode(i).isProcess())
      processCount++;
  }

  targetPartitions = std::min(
      targetPartitions, processCount / config.minProcessesPerPartition);
  targetPartitions = std::max(targetPartitions, size_t(1));

  LLVM_DEBUG(llvm::dbgs() << "Partitioning " << processCount
                          << " processes into " << targetPartitions
                          << " partitions\n");

  // Create and apply partitioning strategy
  auto strategy = createStrategy();
  assignment = strategy->partition(graph, targetPartitions);

  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      endTime - startTime);
  stats.partitioningTimeMs = duration.count();

  // Calculate quality metrics
  stats.cutEdges = calculateCutSize();
  stats.imbalanceRatio = calculateImbalance();

  LLVM_DEBUG(llvm::dbgs() << "Partitioning complete: cut=" << stats.cutEdges
                          << ", imbalance=" << stats.imbalanceRatio << "\n");
}

void DesignPartitioner::applyPartitioning(ParallelScheduler &parallelScheduler) {
  // Create partitions
  size_t numPartitions = getPartitionCount();
  for (size_t i = 0; i < numPartitions; ++i) {
    parallelScheduler.createPartition("partition_" + std::to_string(i));
  }

  // Assign processes to partitions
  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    const auto &node = graph.getNode(i);
    if (node.isProcess() && assignment[i] != InvalidPartitionId) {
      parallelScheduler.assignProcess(node.processId, assignment[i]);
    }
  }

  // Identify and declare boundary signals
  for (size_t edgeIdx = 0; edgeIdx < graph.getEdgeCount(); ++edgeIdx) {
    const auto &edge = graph.getEdge(edgeIdx);
    PartitionId srcPart = assignment[edge.sourceNode];
    PartitionId dstPart = assignment[edge.destNode];

    if (srcPart != dstPart && srcPart != InvalidPartitionId &&
        dstPart != InvalidPartitionId) {
      // This is a boundary edge
      const auto &srcNode = graph.getNode(edge.sourceNode);
      if (srcNode.isSignal()) {
        llvm::SmallVector<PartitionId, 4> dests;
        dests.push_back(dstPart);
        parallelScheduler.declareBoundarySignal(
            srcNode.signalId, srcPart, dests);
      }
    }
  }
}

PartitionId DesignPartitioner::getProcessPartition(ProcessId pid) const {
  size_t nodeIdx = graph.getProcessNode(pid);
  if (nodeIdx == SIZE_MAX || nodeIdx >= assignment.size())
    return InvalidPartitionId;
  return assignment[nodeIdx];
}

size_t DesignPartitioner::getPartitionCount() const {
  PartitionId maxPart = 0;
  for (PartitionId part : assignment) {
    if (part != InvalidPartitionId && part > maxPart)
      maxPart = part;
  }
  return maxPart + 1;
}

size_t DesignPartitioner::calculateCutSize() const {
  size_t cutEdges = 0;
  for (size_t edgeIdx = 0; edgeIdx < graph.getEdgeCount(); ++edgeIdx) {
    const auto &edge = graph.getEdge(edgeIdx);
    if (edge.sourceNode >= assignment.size() ||
        edge.destNode >= assignment.size())
      continue;

    PartitionId srcPart = assignment[edge.sourceNode];
    PartitionId dstPart = assignment[edge.destNode];
    if (srcPart != dstPart && srcPart != InvalidPartitionId &&
        dstPart != InvalidPartitionId) {
      cutEdges += edge.weight;
    }
  }
  return cutEdges;
}

double DesignPartitioner::calculateImbalance() const {
  std::vector<size_t> sizes = getPartitionSizes();
  if (sizes.empty())
    return 1.0;

  size_t maxSize = *std::max_element(sizes.begin(), sizes.end());
  size_t total = std::accumulate(sizes.begin(), sizes.end(), size_t(0));
  double avgSize = static_cast<double>(total) / sizes.size();

  return avgSize > 0 ? maxSize / avgSize : 1.0;
}

std::vector<size_t> DesignPartitioner::getPartitionSizes() const {
  size_t numPartitions = getPartitionCount();
  std::vector<size_t> sizes(numPartitions, 0);

  for (size_t i = 0; i < graph.getNodeCount(); ++i) {
    if (graph.getNode(i).isProcess() && assignment[i] < numPartitions) {
      sizes[assignment[i]]++;
    }
  }

  return sizes;
}

void DesignPartitioner::printResults(llvm::raw_ostream &os) const {
  os << "=== Design Partitioning Results ===\n";
  os << "Total nodes: " << stats.nodesAnalyzed << "\n";
  os << "Total edges: " << stats.edgesAnalyzed << "\n";
  os << "Partitions: " << getPartitionCount() << "\n";
  os << "Cut edges: " << stats.cutEdges << "\n";
  os << "Imbalance ratio: " << stats.imbalanceRatio << "\n";
  os << "Partitioning time: " << stats.partitioningTimeMs << " ms\n";
  os << "\n";

  os << "Partition sizes:\n";
  std::vector<size_t> sizes = getPartitionSizes();
  for (size_t i = 0; i < sizes.size(); ++i) {
    os << "  Partition " << i << ": " << sizes[i] << " processes\n";
  }
}

//===----------------------------------------------------------------------===//
// SignalDependencyAnalyzer Implementation
//===----------------------------------------------------------------------===//

llvm::DenseSet<SignalId> SignalDependencyAnalyzer::emptySet;

void SignalDependencyAnalyzer::analyzeAll() {
  // This is a placeholder for runtime dependency tracking
  // A full implementation would instrument process execution
  // to track actual signal reads and writes
}

const llvm::DenseSet<SignalId> &
SignalDependencyAnalyzer::getReads(ProcessId pid) const {
  auto it = processReads.find(pid);
  return it != processReads.end() ? it->second : emptySet;
}

const llvm::DenseSet<SignalId> &
SignalDependencyAnalyzer::getWrites(ProcessId pid) const {
  auto it = processWrites.find(pid);
  return it != processWrites.end() ? it->second : emptySet;
}

bool SignalDependencyAnalyzer::hasDependency(ProcessId writer,
                                             ProcessId reader) const {
  const auto &writes = getWrites(writer);
  const auto &reads = getReads(reader);

  for (SignalId sig : writes) {
    if (reads.count(sig))
      return true;
  }
  return false;
}

llvm::SmallVector<ProcessId, 8>
SignalDependencyAnalyzer::getDependents(ProcessId pid) const {
  llvm::SmallVector<ProcessId, 8> dependents;
  const auto &writes = getWrites(pid);

  for (const auto &kv : processReads) {
    if (kv.first == pid)
      continue;
    for (SignalId sig : writes) {
      if (kv.second.count(sig)) {
        dependents.push_back(kv.first);
        break;
      }
    }
  }

  return dependents;
}

llvm::SmallVector<ProcessId, 8>
SignalDependencyAnalyzer::getDependencies(ProcessId pid) const {
  llvm::SmallVector<ProcessId, 8> dependencies;
  const auto &reads = getReads(pid);

  for (const auto &kv : processWrites) {
    if (kv.first == pid)
      continue;
    for (SignalId sig : reads) {
      if (kv.second.count(sig)) {
        dependencies.push_back(kv.first);
        break;
      }
    }
  }

  return dependencies;
}
