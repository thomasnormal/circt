//===- ParallelSchedulerTest.cpp - ParallelScheduler unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ParallelScheduler.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Partition Tests
//===----------------------------------------------------------------------===//

TEST(PartitionTest, BasicPartition) {
  Partition partition(0, "test_partition");

  EXPECT_EQ(partition.getId(), 0u);
  EXPECT_EQ(partition.getName(), "test_partition");
  EXPECT_EQ(partition.getProcessCount(), 0u);
  EXPECT_EQ(partition.getSignalCount(), 0u);
  EXPECT_FALSE(partition.isActive());
}

TEST(PartitionTest, AddProcesses) {
  Partition partition(0, "test");

  partition.addProcess(1);
  partition.addProcess(2);
  partition.addProcess(3);

  EXPECT_EQ(partition.getProcessCount(), 3u);
  EXPECT_EQ(partition.getProcesses().size(), 3u);
  EXPECT_EQ(partition.getProcesses()[0], 1u);
  EXPECT_EQ(partition.getProcesses()[1], 2u);
  EXPECT_EQ(partition.getProcesses()[2], 3u);
}

TEST(PartitionTest, AddSignals) {
  Partition partition(0, "test");

  partition.addInternalSignal(10);
  partition.addInternalSignal(20);

  EXPECT_EQ(partition.getSignalCount(), 2u);
}

TEST(PartitionTest, BoundarySignals) {
  Partition partition(0, "test");

  partition.addInputBoundary(100);
  partition.addOutputBoundary(200);

  EXPECT_EQ(partition.getInputBoundarySignals().size(), 1u);
  EXPECT_EQ(partition.getOutputBoundarySignals().size(), 1u);
  EXPECT_EQ(partition.getInputBoundarySignals()[0], 100u);
  EXPECT_EQ(partition.getOutputBoundarySignals()[0], 200u);
}

TEST(PartitionTest, ActiveState) {
  Partition partition(0, "test");

  EXPECT_FALSE(partition.isActive());
  partition.setActive(true);
  EXPECT_TRUE(partition.isActive());
  partition.setActive(false);
  EXPECT_FALSE(partition.isActive());
}

TEST(PartitionTest, LoadCalculation) {
  Partition partition(0, "test");

  double emptyLoad = partition.getLoad();
  EXPECT_EQ(emptyLoad, 0.0);

  partition.addProcess(1);
  partition.addProcess(2);
  double loadWithProcesses = partition.getLoad();
  EXPECT_GT(loadWithProcesses, emptyLoad);

  partition.addInternalSignal(10);
  double loadWithSignals = partition.getLoad();
  EXPECT_GT(loadWithSignals, loadWithProcesses);
}

//===----------------------------------------------------------------------===//
// PartitionGraph Tests
//===----------------------------------------------------------------------===//

TEST(PartitionGraphTest, BasicGraph) {
  PartitionGraph graph;

  graph.addEdge(0, 1, 100);
  graph.addEdge(0, 2, 101);
  graph.addEdge(1, 2, 102);

  EXPECT_EQ(graph.getCutSize(), 3u);
}

TEST(PartitionGraphTest, GetEdges) {
  PartitionGraph graph;

  graph.addEdge(0, 1, 100);
  graph.addEdge(0, 2, 101);

  const auto &outEdges = graph.getOutEdges(0);
  EXPECT_EQ(outEdges.size(), 2u);

  const auto &inEdges = graph.getInEdges(1);
  EXPECT_EQ(inEdges.size(), 1u);
  EXPECT_EQ(inEdges[0].first, 0u);
  EXPECT_EQ(inEdges[0].second, 100u);
}

//===----------------------------------------------------------------------===//
// ThreadBarrier Tests
//===----------------------------------------------------------------------===//

TEST(ThreadBarrierTest, SingleThread) {
  ThreadBarrier barrier(1);
  // Single thread should pass through immediately
  barrier.wait();
}

TEST(ThreadBarrierTest, MultipleThreads) {
  const size_t numThreads = 4;
  ThreadBarrier barrier(numThreads);
  std::atomic<int> counter{0};
  std::vector<std::thread> threads;

  for (size_t i = 0; i < numThreads; ++i) {
    threads.emplace_back([&barrier, &counter]() {
      counter.fetch_add(1);
      barrier.wait();
      // After barrier, all threads should have incremented
      EXPECT_EQ(counter.load(), static_cast<int>(numThreads));
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  EXPECT_EQ(counter.load(), static_cast<int>(numThreads));
}

//===----------------------------------------------------------------------===//
// WorkStealingQueue Tests
//===----------------------------------------------------------------------===//

TEST(WorkStealingQueueTest, PushPop) {
  WorkStealingQueue<int> queue(16);

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0u);

  EXPECT_TRUE(queue.push(1));
  EXPECT_TRUE(queue.push(2));
  EXPECT_TRUE(queue.push(3));

  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.size(), 3u);

  int value;
  EXPECT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 3); // LIFO for pop

  EXPECT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 2);

  EXPECT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 1);

  EXPECT_TRUE(queue.empty());
  EXPECT_FALSE(queue.pop(value));
}

TEST(WorkStealingQueueTest, Steal) {
  WorkStealingQueue<int> queue(16);

  queue.push(1);
  queue.push(2);
  queue.push(3);

  int value;
  EXPECT_TRUE(queue.steal(value));
  EXPECT_EQ(value, 1); // FIFO for steal

  EXPECT_TRUE(queue.steal(value));
  EXPECT_EQ(value, 2);

  EXPECT_TRUE(queue.steal(value));
  EXPECT_EQ(value, 3);

  EXPECT_FALSE(queue.steal(value));
}

TEST(WorkStealingQueueTest, MixedOperations) {
  WorkStealingQueue<int> queue(16);

  queue.push(1);
  queue.push(2);

  int value;
  // Steal takes from bottom (oldest)
  EXPECT_TRUE(queue.steal(value));
  EXPECT_EQ(value, 1);

  queue.push(3);

  // Pop takes from top (newest)
  EXPECT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 3);

  EXPECT_TRUE(queue.pop(value));
  EXPECT_EQ(value, 2);

  EXPECT_TRUE(queue.empty());
}

//===----------------------------------------------------------------------===//
// ParallelScheduler Tests
//===----------------------------------------------------------------------===//

TEST(ParallelSchedulerTest, Creation) {
  ProcessScheduler::Config psConfig;
  ProcessScheduler baseScheduler(psConfig);

  ParallelScheduler::Config config;
  config.numThreads = 2;
  ParallelScheduler scheduler(baseScheduler, config);

  EXPECT_EQ(scheduler.getNumThreads(), 2u);
  EXPECT_EQ(scheduler.getPartitionCount(), 0u);
  EXPECT_FALSE(scheduler.isRunning());
}

TEST(ParallelSchedulerTest, CreatePartitions) {
  ProcessScheduler::Config psConfig;
  ProcessScheduler baseScheduler(psConfig);

  ParallelScheduler scheduler(baseScheduler);

  PartitionId p1 = scheduler.createPartition("partition1");
  PartitionId p2 = scheduler.createPartition("partition2");

  EXPECT_EQ(p1, 0u);
  EXPECT_EQ(p2, 1u);
  EXPECT_EQ(scheduler.getPartitionCount(), 2u);

  const Partition *partition1 = scheduler.getPartition(p1);
  const Partition *partition2 = scheduler.getPartition(p2);

  EXPECT_NE(partition1, nullptr);
  EXPECT_NE(partition2, nullptr);
  EXPECT_EQ(partition1->getName(), "partition1");
  EXPECT_EQ(partition2->getName(), "partition2");
}

TEST(ParallelSchedulerTest, AssignProcess) {
  ProcessScheduler::Config psConfig;
  ProcessScheduler baseScheduler(psConfig);

  ParallelScheduler scheduler(baseScheduler);

  PartitionId p1 = scheduler.createPartition("partition1");
  PartitionId p2 = scheduler.createPartition("partition2");

  // Register some processes
  ProcessId proc1 = baseScheduler.registerProcess("proc1", []() {});
  ProcessId proc2 = baseScheduler.registerProcess("proc2", []() {});
  ProcessId proc3 = baseScheduler.registerProcess("proc3", []() {});

  scheduler.assignProcess(proc1, p1);
  scheduler.assignProcess(proc2, p1);
  scheduler.assignProcess(proc3, p2);

  EXPECT_EQ(scheduler.getPartitionForProcess(proc1), p1);
  EXPECT_EQ(scheduler.getPartitionForProcess(proc2), p1);
  EXPECT_EQ(scheduler.getPartitionForProcess(proc3), p2);

  const Partition *partition1 = scheduler.getPartition(p1);
  const Partition *partition2 = scheduler.getPartition(p2);

  EXPECT_EQ(partition1->getProcessCount(), 2u);
  EXPECT_EQ(partition2->getProcessCount(), 1u);
}

TEST(ParallelSchedulerTest, DeclareBoundarySignal) {
  ProcessScheduler::Config psConfig;
  ProcessScheduler baseScheduler(psConfig);

  ParallelScheduler scheduler(baseScheduler);

  PartitionId p1 = scheduler.createPartition("partition1");
  PartitionId p2 = scheduler.createPartition("partition2");

  SignalId sig1 = baseScheduler.registerSignal("sig1");

  llvm::SmallVector<PartitionId, 4> dests = {p2};
  scheduler.declareBoundarySignal(sig1, p1, dests);

  const Partition *partition1 = scheduler.getPartition(p1);
  const Partition *partition2 = scheduler.getPartition(p2);

  EXPECT_EQ(partition1->getOutputBoundarySignals().size(), 1u);
  EXPECT_EQ(partition2->getInputBoundarySignals().size(), 1u);

  const PartitionGraph &graph = scheduler.getPartitionGraph();
  EXPECT_EQ(graph.getCutSize(), 1u);
}

TEST(ParallelSchedulerTest, AutoPartition) {
  ProcessScheduler::Config psConfig;
  ProcessScheduler baseScheduler(psConfig);

  // Register multiple processes
  for (int i = 0; i < 20; ++i) {
    baseScheduler.registerProcess("proc" + std::to_string(i), []() {});
  }

  ParallelScheduler::Config config;
  config.numThreads = 4;
  config.minProcessesPerPartition = 5;
  ParallelScheduler scheduler(baseScheduler, config);

  scheduler.autoPartition();

  // Should create 4 partitions with 5 processes each
  EXPECT_EQ(scheduler.getPartitionCount(), 4u);

  // Each partition should have 5 processes
  size_t totalProcesses = 0;
  for (size_t i = 0; i < scheduler.getPartitionCount(); ++i) {
    const Partition *p = scheduler.getPartition(i);
    totalProcesses += p->getProcessCount();
  }
  EXPECT_EQ(totalProcesses, 20u);
}

//===----------------------------------------------------------------------===//
// PartitionBalancer Tests
//===----------------------------------------------------------------------===//

TEST(PartitionBalancerTest, CalculateImbalance) {
  std::vector<std::unique_ptr<Partition>> partitions;

  // Empty should return 1.0
  EXPECT_EQ(PartitionBalancer::calculateImbalance(partitions), 1.0);

  // Single partition
  partitions.push_back(std::make_unique<Partition>(0, "p0"));
  partitions[0]->addProcess(1);
  EXPECT_EQ(PartitionBalancer::calculateImbalance(partitions), 1.0);

  // Two equal partitions
  partitions.push_back(std::make_unique<Partition>(1, "p1"));
  partitions[1]->addProcess(2);
  EXPECT_EQ(PartitionBalancer::calculateImbalance(partitions), 1.0);

  // Unequal partitions
  partitions[0]->addProcess(3);
  partitions[0]->addProcess(4);
  double imbalance = PartitionBalancer::calculateImbalance(partitions);
  EXPECT_GT(imbalance, 1.0);
}

TEST(PartitionBalancerTest, NeedsRebalancing) {
  std::vector<std::unique_ptr<Partition>> partitions;

  // Balanced case
  partitions.push_back(std::make_unique<Partition>(0, "p0"));
  partitions.push_back(std::make_unique<Partition>(1, "p1"));
  partitions[0]->addProcess(1);
  partitions[1]->addProcess(2);

  EXPECT_FALSE(PartitionBalancer::needsRebalancing(partitions, 1.5));

  // Unbalanced case
  partitions[0]->addProcess(3);
  partitions[0]->addProcess(4);
  partitions[0]->addProcess(5);

  EXPECT_TRUE(PartitionBalancer::needsRebalancing(partitions, 1.5));
}

TEST(ParallelSchedulerTest, AbortStopsExecution) {
  ProcessScheduler scheduler;
  int counter = 0;
  ProcessId pid = 0;

  pid = scheduler.registerProcess("test", [&]() { counter++; });
  (void)pid;
  scheduler.setShouldAbortCallback([]() { return true; });
  scheduler.initialize();

  ParallelScheduler::Config config;
  config.numThreads = 1;
  ParallelScheduler parallel(scheduler, config);
  parallel.autoPartition();

  size_t deltas = parallel.executeCurrentTimeParallel();
  EXPECT_EQ(deltas, 0u);
  EXPECT_EQ(counter, 0);
}
