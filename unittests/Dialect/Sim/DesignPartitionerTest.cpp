//===- DesignPartitionerTest.cpp - DesignPartitioner unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/DesignPartitioner.h"
#include "gtest/gtest.h"

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// DependencyGraph Tests
//===----------------------------------------------------------------------===//

TEST(DependencyGraphTest, AddNodes) {
  DependencyGraph graph;

  size_t p1 = graph.addProcessNode(1);
  size_t p2 = graph.addProcessNode(2);
  size_t s1 = graph.addSignalNode(100);

  EXPECT_EQ(graph.getNodeCount(), 3u);
  EXPECT_EQ(p1, 0u);
  EXPECT_EQ(p2, 1u);
  EXPECT_EQ(s1, 2u);

  EXPECT_TRUE(graph.getNode(p1).isProcess());
  EXPECT_TRUE(graph.getNode(p2).isProcess());
  EXPECT_TRUE(graph.getNode(s1).isSignal());
}

TEST(DependencyGraphTest, AddEdges) {
  DependencyGraph graph;

  size_t p1 = graph.addProcessNode(1);
  size_t p2 = graph.addProcessNode(2);
  size_t s1 = graph.addSignalNode(100);

  graph.addEdge(p1, s1, 1);
  graph.addEdge(s1, p2, 1);

  EXPECT_EQ(graph.getEdgeCount(), 2u);

  const auto &outEdges = graph.getOutEdges(p1);
  EXPECT_EQ(outEdges.size(), 1u);
  EXPECT_EQ(graph.getEdge(outEdges[0]).destNode, s1);

  const auto &inEdges = graph.getInEdges(p2);
  EXPECT_EQ(inEdges.size(), 1u);
  EXPECT_EQ(graph.getEdge(inEdges[0]).sourceNode, s1);
}

TEST(DependencyGraphTest, GetNodeIndex) {
  DependencyGraph graph;

  graph.addProcessNode(1);
  graph.addProcessNode(2);
  graph.addSignalNode(100);

  EXPECT_EQ(graph.getProcessNode(1), 0u);
  EXPECT_EQ(graph.getProcessNode(2), 1u);
  EXPECT_EQ(graph.getSignalNode(100), 2u);
  EXPECT_EQ(graph.getProcessNode(99), SIZE_MAX); // Not found
}

TEST(DependencyGraphTest, Clear) {
  DependencyGraph graph;

  graph.addProcessNode(1);
  graph.addProcessNode(2);
  graph.addEdge(0, 1);

  EXPECT_EQ(graph.getNodeCount(), 2u);
  EXPECT_EQ(graph.getEdgeCount(), 1u);

  graph.clear();

  EXPECT_EQ(graph.getNodeCount(), 0u);
  EXPECT_EQ(graph.getEdgeCount(), 0u);
}

//===----------------------------------------------------------------------===//
// PartitioningStrategy Tests
//===----------------------------------------------------------------------===//

TEST(RoundRobinStrategyTest, BasicPartition) {
  DependencyGraph graph;

  // Add 10 processes
  for (ProcessId i = 0; i < 10; ++i) {
    graph.addProcessNode(i);
  }

  RoundRobinStrategy strategy;
  auto assignment = strategy.partition(graph, 3);

  EXPECT_EQ(assignment.size(), 10u);

  // Count processes per partition
  std::vector<int> counts(3, 0);
  for (size_t i = 0; i < assignment.size(); ++i) {
    if (assignment[i] < 3) {
      counts[assignment[i]]++;
    }
  }

  // Should be roughly balanced
  for (int count : counts) {
    EXPECT_GE(count, 3);
    EXPECT_LE(count, 4);
  }
}

TEST(BFSStrategyTest, BasicPartition) {
  DependencyGraph graph;

  // Create a chain of processes connected by signals
  for (ProcessId i = 0; i < 8; ++i) {
    graph.addProcessNode(i);
  }

  // Add edges to create connectivity
  for (size_t i = 0; i < 7; ++i) {
    graph.addEdge(i, i + 1);
  }

  BFSStrategy strategy;
  auto assignment = strategy.partition(graph, 2);

  EXPECT_EQ(assignment.size(), 8u);

  // All processes should be assigned
  for (size_t i = 0; i < 8; ++i) {
    if (graph.getNode(i).isProcess()) {
      EXPECT_LT(assignment[i], 2u);
    }
  }
}

TEST(FMStrategyTest, BasicPartition) {
  DependencyGraph graph;

  // Create two clusters of processes
  for (ProcessId i = 0; i < 10; ++i) {
    graph.addProcessNode(i);
  }

  // Create internal edges within clusters
  for (size_t i = 0; i < 4; ++i) {
    graph.addEdge(i, i + 1);
  }
  for (size_t i = 5; i < 9; ++i) {
    graph.addEdge(i, i + 1);
  }

  // Create one cross-cluster edge
  graph.addEdge(2, 7);

  FMStrategy strategy;
  auto assignment = strategy.partition(graph, 2);

  EXPECT_EQ(assignment.size(), graph.getNodeCount());
}

//===----------------------------------------------------------------------===//
// DesignPartitioner Tests
//===----------------------------------------------------------------------===//

TEST(DesignPartitionerTest, BuildDependencyGraph) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  // Register processes with sensitivities
  ProcessId p1 = scheduler.registerProcess("proc1", []() {});
  ProcessId p2 = scheduler.registerProcess("proc2", []() {});

  SignalId s1 = scheduler.registerSignal("sig1");
  SignalId s2 = scheduler.registerSignal("sig2");

  scheduler.addSensitivity(p1, s1);
  scheduler.addSensitivity(p2, s2);

  DesignPartitioner partitioner(scheduler);
  partitioner.buildDependencyGraph();

  const DependencyGraph &graph = partitioner.getDependencyGraph();

  // Should have nodes for processes and signals
  EXPECT_GE(graph.getNodeCount(), 2u);
}

TEST(DesignPartitionerTest, Partition) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  // Register 20 processes
  for (int i = 0; i < 20; ++i) {
    scheduler.registerProcess("proc" + std::to_string(i), []() {});
  }

  DesignPartitioner::Config partConfig;
  partConfig.targetPartitions = 4;
  partConfig.minProcessesPerPartition = 5;
  partConfig.strategy = DesignPartitioner::Config::Strategy::RoundRobin;

  DesignPartitioner partitioner(scheduler, partConfig);
  partitioner.buildDependencyGraph();
  partitioner.partition();

  EXPECT_EQ(partitioner.getPartitionCount(), 4u);

  // All processes should be assigned to valid partitions
  std::vector<size_t> sizes = partitioner.getPartitionSizes();
  EXPECT_EQ(sizes.size(), 4u);

  size_t total = 0;
  for (size_t s : sizes) {
    total += s;
  }
  EXPECT_EQ(total, 20u);
}

TEST(DesignPartitionerTest, QualityMetrics) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  for (int i = 0; i < 16; ++i) {
    scheduler.registerProcess("proc" + std::to_string(i), []() {});
  }

  DesignPartitioner::Config partConfig;
  partConfig.targetPartitions = 4;
  partConfig.strategy = DesignPartitioner::Config::Strategy::RoundRobin;

  DesignPartitioner partitioner(scheduler, partConfig);
  partitioner.buildDependencyGraph();
  partitioner.partition();

  // Check metrics
  size_t cutSize = partitioner.calculateCutSize();
  double imbalance = partitioner.calculateImbalance();

  // For round-robin with no edges, cut should be 0
  EXPECT_EQ(cutSize, 0u);

  // Imbalance should be close to 1.0 for even distribution
  EXPECT_LE(imbalance, 1.1);
}

TEST(DesignPartitionerTest, Statistics) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  for (int i = 0; i < 10; ++i) {
    scheduler.registerProcess("proc" + std::to_string(i), []() {});
  }

  DesignPartitioner partitioner(scheduler);
  partitioner.buildDependencyGraph();
  partitioner.partition();

  const auto &stats = partitioner.getStatistics();
  EXPECT_GT(stats.nodesAnalyzed, 0u);
}

//===----------------------------------------------------------------------===//
// SignalDependencyAnalyzer Tests
//===----------------------------------------------------------------------===//

TEST(SignalDependencyAnalyzerTest, EmptyAnalyzer) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  SignalDependencyAnalyzer analyzer(scheduler);
  analyzer.analyzeAll();

  // Should not crash on empty scheduler
  const auto &reads = analyzer.getReads(1);
  EXPECT_TRUE(reads.empty());
}

TEST(SignalDependencyAnalyzerTest, HasDependency) {
  ProcessScheduler::Config config;
  ProcessScheduler scheduler(config);

  SignalDependencyAnalyzer analyzer(scheduler);

  // With no actual writes/reads recorded, should return false
  EXPECT_FALSE(analyzer.hasDependency(1, 2));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
