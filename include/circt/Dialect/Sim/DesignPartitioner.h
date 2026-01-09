//===- DesignPartitioner.h - Design partitioning analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the design partitioner that analyzes signal dependencies
// and creates optimal partitions for parallel simulation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_DESIGNPARTITIONER_H
#define CIRCT_DIALECT_SIM_DESIGNPARTITIONER_H

#include "circt/Dialect/Sim/ParallelScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// DependencyGraph - Signal and process dependency graph
//===----------------------------------------------------------------------===//

/// Represents a node in the dependency graph.
struct DependencyNode {
  enum class Kind { Process, Signal };

  Kind kind;
  union {
    ProcessId processId;
    SignalId signalId;
  };

  DependencyNode(ProcessId pid) : kind(Kind::Process), processId(pid) {}
  DependencyNode(SignalId sid) : kind(Kind::Signal), signalId(sid) {}

  bool isProcess() const { return kind == Kind::Process; }
  bool isSignal() const { return kind == Kind::Signal; }

  bool operator==(const DependencyNode &other) const {
    if (kind != other.kind)
      return false;
    return isProcess() ? processId == other.processId
                       : signalId == other.signalId;
  }
};

/// A dependency edge between nodes.
struct DependencyEdge {
  size_t sourceNode;
  size_t destNode;
  uint32_t weight; // Edge weight for cut optimization

  DependencyEdge(size_t src, size_t dst, uint32_t w = 1)
      : sourceNode(src), destNode(dst), weight(w) {}
};

/// The dependency graph for partitioning analysis.
class DependencyGraph {
public:
  /// Add a process node and return its index.
  size_t addProcessNode(ProcessId pid) {
    size_t idx = nodes.size();
    nodes.emplace_back(pid);
    processToNode[pid] = idx;
    return idx;
  }

  /// Add a signal node and return its index.
  size_t addSignalNode(SignalId sid) {
    size_t idx = nodes.size();
    nodes.emplace_back(sid);
    signalToNode[sid] = idx;
    return idx;
  }

  /// Add an edge between nodes.
  void addEdge(size_t src, size_t dst, uint32_t weight = 1) {
    edges.emplace_back(src, dst, weight);
    outEdges[src].push_back(edges.size() - 1);
    inEdges[dst].push_back(edges.size() - 1);
  }

  /// Get a node by index.
  const DependencyNode &getNode(size_t idx) const { return nodes[idx]; }

  /// Get node index for a process.
  size_t getProcessNode(ProcessId pid) const {
    auto it = processToNode.find(pid);
    return it != processToNode.end() ? it->second : SIZE_MAX;
  }

  /// Get node index for a signal.
  size_t getSignalNode(SignalId sid) const {
    auto it = signalToNode.find(sid);
    return it != signalToNode.end() ? it->second : SIZE_MAX;
  }

  /// Get all outgoing edge indices for a node.
  const llvm::SmallVector<size_t, 8> &getOutEdges(size_t node) const {
    static const llvm::SmallVector<size_t, 8> empty;
    auto it = outEdges.find(node);
    return it != outEdges.end() ? it->second : empty;
  }

  /// Get all incoming edge indices for a node.
  const llvm::SmallVector<size_t, 8> &getInEdges(size_t node) const {
    static const llvm::SmallVector<size_t, 8> empty;
    auto it = inEdges.find(node);
    return it != inEdges.end() ? it->second : empty;
  }

  /// Get an edge by index.
  const DependencyEdge &getEdge(size_t idx) const { return edges[idx]; }

  /// Get the number of nodes.
  size_t getNodeCount() const { return nodes.size(); }

  /// Get the number of edges.
  size_t getEdgeCount() const { return edges.size(); }

  /// Clear the graph.
  void clear() {
    nodes.clear();
    edges.clear();
    processToNode.clear();
    signalToNode.clear();
    outEdges.clear();
    inEdges.clear();
  }

private:
  std::vector<DependencyNode> nodes;
  std::vector<DependencyEdge> edges;
  llvm::DenseMap<ProcessId, size_t> processToNode;
  llvm::DenseMap<SignalId, size_t> signalToNode;
  llvm::DenseMap<size_t, llvm::SmallVector<size_t, 8>> outEdges;
  llvm::DenseMap<size_t, llvm::SmallVector<size_t, 8>> inEdges;
};

//===----------------------------------------------------------------------===//
// PartitioningStrategy - Abstract base for partitioning algorithms
//===----------------------------------------------------------------------===//

/// Abstract base class for partitioning strategies.
class PartitioningStrategy {
public:
  virtual ~PartitioningStrategy() = default;

  /// Partition the dependency graph into the given number of partitions.
  /// Returns a map from node index to partition ID.
  virtual std::vector<PartitionId>
  partition(const DependencyGraph &graph, size_t numPartitions) = 0;

  /// Get the name of this strategy.
  virtual const char *getName() const = 0;
};

//===----------------------------------------------------------------------===//
// RoundRobinStrategy - Simple round-robin partitioning
//===----------------------------------------------------------------------===//

/// Simple round-robin partitioning strategy.
/// Fast but doesn't optimize for cut edges.
class RoundRobinStrategy : public PartitioningStrategy {
public:
  std::vector<PartitionId> partition(const DependencyGraph &graph,
                                     size_t numPartitions) override;
  const char *getName() const override { return "RoundRobin"; }
};

//===----------------------------------------------------------------------===//
// BFSStrategy - BFS-based partitioning
//===----------------------------------------------------------------------===//

/// BFS-based partitioning that groups connected nodes.
/// Starts from seed nodes and grows partitions via BFS.
class BFSStrategy : public PartitioningStrategy {
public:
  std::vector<PartitionId> partition(const DependencyGraph &graph,
                                     size_t numPartitions) override;
  const char *getName() const override { return "BFS"; }
};

//===----------------------------------------------------------------------===//
// FMStrategy - Fiduccia-Mattheyses partitioning
//===----------------------------------------------------------------------===//

/// Fiduccia-Mattheyses (FM) algorithm for graph partitioning.
/// Iteratively improves partitioning by moving nodes to reduce cut edges.
class FMStrategy : public PartitioningStrategy {
public:
  struct Config {
    /// Maximum number of passes.
    size_t maxPasses;

    /// Balance tolerance (ratio of max/min partition sizes).
    double balanceTolerance;

    Config() : maxPasses(10), balanceTolerance(1.2) {}
  };

  FMStrategy(Config config = Config()) : config(config) {}

  std::vector<PartitionId> partition(const DependencyGraph &graph,
                                     size_t numPartitions) override;
  const char *getName() const override { return "FM"; }

private:
  Config config;

  /// Calculate the gain from moving a node to a different partition.
  int calculateGain(const DependencyGraph &graph,
                    const std::vector<PartitionId> &assignment, size_t node,
                    PartitionId targetPartition);

  /// Perform one pass of the FM algorithm.
  bool performPass(const DependencyGraph &graph,
                   std::vector<PartitionId> &assignment, size_t numPartitions);
};

//===----------------------------------------------------------------------===//
// KLStrategy - Kernighan-Lin partitioning
//===----------------------------------------------------------------------===//

/// Kernighan-Lin (KL) algorithm for graph partitioning.
/// Swaps pairs of nodes between partitions to reduce cut edges.
class KLStrategy : public PartitioningStrategy {
public:
  struct Config {
    /// Maximum number of passes.
    size_t maxPasses;

    Config() : maxPasses(10) {}
  };

  KLStrategy(Config config = Config()) : config(config) {}

  std::vector<PartitionId> partition(const DependencyGraph &graph,
                                     size_t numPartitions) override;
  const char *getName() const override { return "KL"; }

private:
  Config config;
};

//===----------------------------------------------------------------------===//
// SpectralStrategy - Spectral partitioning
//===----------------------------------------------------------------------===//

/// Spectral partitioning using Laplacian eigenvectors.
/// High quality partitioning but more computationally expensive.
class SpectralStrategy : public PartitioningStrategy {
public:
  struct Config {
    /// Number of power iterations for eigenvector computation.
    size_t powerIterations;

    /// Convergence tolerance.
    double tolerance;

    Config() : powerIterations(100), tolerance(1e-6) {}
  };

  SpectralStrategy(Config config = Config()) : config(config) {}

  std::vector<PartitionId> partition(const DependencyGraph &graph,
                                     size_t numPartitions) override;
  const char *getName() const override { return "Spectral"; }

private:
  Config config;

  /// Compute the Fiedler vector (second smallest eigenvector of Laplacian).
  std::vector<double> computeFiedlerVector(const DependencyGraph &graph);
};

//===----------------------------------------------------------------------===//
// DesignPartitioner - Main partitioning interface
//===----------------------------------------------------------------------===//

/// The main design partitioner that builds dependency graphs and
/// applies partitioning strategies.
class DesignPartitioner {
public:
  /// Configuration for the design partitioner.
  struct Config {
    /// Target number of partitions (0 = auto based on thread count).
    size_t targetPartitions;

    /// Minimum processes per partition.
    size_t minProcessesPerPartition;

    /// Maximum imbalance ratio between partition sizes.
    double maxImbalance;

    /// Strategy to use for partitioning.
    enum class Strategy { RoundRobin, BFS, FM, KL, Spectral };
    Strategy strategy;

    /// Enable multi-level partitioning (coarsen, partition, uncoarsen).
    bool multiLevel;

    Config()
        : targetPartitions(0), minProcessesPerPartition(10), maxImbalance(1.3),
          strategy(Strategy::FM), multiLevel(true) {}
  };

  DesignPartitioner(ProcessScheduler &scheduler, Config config = Config());
  ~DesignPartitioner();

  //===------------------------------------------------------------------===//
  // Dependency Analysis
  //===------------------------------------------------------------------===//

  /// Build the dependency graph from the scheduler.
  void buildDependencyGraph();

  /// Get the dependency graph.
  const DependencyGraph &getDependencyGraph() const { return graph; }

  /// Analyze dependencies for a specific process.
  void analyzeProcessDependencies(ProcessId pid);

  //===------------------------------------------------------------------===//
  // Partitioning
  //===------------------------------------------------------------------===//

  /// Perform partitioning.
  void partition();

  /// Apply the partitioning result to a ParallelScheduler.
  void applyPartitioning(ParallelScheduler &parallelScheduler);

  /// Get the partition assignment for a process.
  PartitionId getProcessPartition(ProcessId pid) const;

  /// Get the number of partitions created.
  size_t getPartitionCount() const;

  //===------------------------------------------------------------------===//
  // Quality Metrics
  //===------------------------------------------------------------------===//

  /// Calculate the number of cut edges.
  size_t calculateCutSize() const;

  /// Calculate the load imbalance ratio.
  double calculateImbalance() const;

  /// Get partition sizes.
  std::vector<size_t> getPartitionSizes() const;

  //===------------------------------------------------------------------===//
  // Statistics
  //===------------------------------------------------------------------===//

  struct Statistics {
    size_t nodesAnalyzed = 0;
    size_t edgesAnalyzed = 0;
    size_t cutEdges = 0;
    double imbalanceRatio = 0;
    uint64_t partitioningTimeMs = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Print partitioning results to the given stream.
  void printResults(llvm::raw_ostream &os) const;

private:
  /// Create the partitioning strategy based on config.
  std::unique_ptr<PartitioningStrategy> createStrategy();

  /// Coarsen the graph for multi-level partitioning.
  DependencyGraph coarsenGraph(const DependencyGraph &graph);

  /// Uncoarsen the partition assignment.
  std::vector<PartitionId>
  uncoarsenAssignment(const DependencyGraph &coarseGraph,
                      const std::vector<PartitionId> &coarseAssignment);

  ProcessScheduler &scheduler;
  Config config;
  DependencyGraph graph;
  std::vector<PartitionId> assignment;
  Statistics stats;

  // Mapping for multi-level partitioning
  std::vector<size_t> coarseToFine;
};

//===----------------------------------------------------------------------===//
// SignalDependencyAnalyzer - Analyzes signal read/write dependencies
//===----------------------------------------------------------------------===//

/// Analyzes which signals are read and written by each process.
class SignalDependencyAnalyzer {
public:
  SignalDependencyAnalyzer(ProcessScheduler &scheduler) : scheduler(scheduler) {}

  /// Analyze dependencies for all processes.
  void analyzeAll();

  /// Get signals read by a process.
  const llvm::DenseSet<SignalId> &getReads(ProcessId pid) const;

  /// Get signals written by a process.
  const llvm::DenseSet<SignalId> &getWrites(ProcessId pid) const;

  /// Check if two processes have a dependency (one writes what other reads).
  bool hasDependency(ProcessId writer, ProcessId reader) const;

  /// Get all processes that depend on a given process.
  llvm::SmallVector<ProcessId, 8> getDependents(ProcessId pid) const;

  /// Get all processes that a given process depends on.
  llvm::SmallVector<ProcessId, 8> getDependencies(ProcessId pid) const;

private:
  ProcessScheduler &scheduler;
  llvm::DenseMap<ProcessId, llvm::DenseSet<SignalId>> processReads;
  llvm::DenseMap<ProcessId, llvm::DenseSet<SignalId>> processWrites;
  static llvm::DenseSet<SignalId> emptySet;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_DESIGNPARTITIONER_H
