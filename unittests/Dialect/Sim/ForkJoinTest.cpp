//===- ForkJoinTest.cpp - Fork/Join unit tests ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "gtest/gtest.h"

using namespace circt::sim;

namespace {

class ForkJoinTest : public ::testing::Test {
protected:
  void SetUp() override {
    scheduler = std::make_unique<ProcessScheduler>();
    forkManager = std::make_unique<ForkJoinManager>(*scheduler);
  }

  std::unique_ptr<ProcessScheduler> scheduler;
  std::unique_ptr<ForkJoinManager> forkManager;
};

TEST_F(ForkJoinTest, CreateForkGroup) {
  // Register a parent process
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ASSERT_NE(parent, InvalidProcessId);

  // Create a fork group
  ForkId forkId = forkManager->createFork(parent, ForkJoinType::Join);
  ASSERT_NE(forkId, InvalidForkId);

  // Verify the fork group exists
  ForkGroup *group = forkManager->getForkGroup(forkId);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->parentProcess, parent);
  EXPECT_EQ(group->joinType, ForkJoinType::Join);
  EXPECT_TRUE(group->childProcesses.empty());
}

TEST_F(ForkJoinTest, AddChildToFork) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});
  ProcessId child2 = scheduler->registerProcess("child2", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::Join);

  forkManager->addChildToFork(forkId, child1);
  forkManager->addChildToFork(forkId, child2);

  ForkGroup *group = forkManager->getForkGroup(forkId);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->childProcesses.size(), 2u);
  EXPECT_EQ(group->childProcesses[0], child1);
  EXPECT_EQ(group->childProcesses[1], child2);
}

TEST_F(ForkJoinTest, JoinWaitsForAll) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});
  ProcessId child2 = scheduler->registerProcess("child2", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::Join);
  forkManager->addChildToFork(forkId, child1);
  forkManager->addChildToFork(forkId, child2);

  // Fork should not be complete yet
  EXPECT_FALSE(forkManager->join(forkId));

  // Mark first child complete
  forkManager->markChildComplete(child1);
  EXPECT_FALSE(forkManager->join(forkId));

  // Mark second child complete
  forkManager->markChildComplete(child2);
  EXPECT_TRUE(forkManager->join(forkId));
}

TEST_F(ForkJoinTest, JoinAnyWaitsForFirst) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});
  ProcessId child2 = scheduler->registerProcess("child2", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::JoinAny);
  forkManager->addChildToFork(forkId, child1);
  forkManager->addChildToFork(forkId, child2);

  // Fork should not be complete yet
  EXPECT_FALSE(forkManager->joinAny(forkId));

  // Mark first child complete - should now be complete
  forkManager->markChildComplete(child1);
  EXPECT_TRUE(forkManager->joinAny(forkId));
}

TEST_F(ForkJoinTest, JoinNoneCompletesImmediately) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::JoinNone);
  forkManager->addChildToFork(forkId, child1);

  ForkGroup *group = forkManager->getForkGroup(forkId);
  EXPECT_TRUE(group->isComplete());
}

TEST_F(ForkJoinTest, DisableForkTerminatesChildren) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});
  ProcessId child2 = scheduler->registerProcess("child2", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::JoinNone);
  forkManager->addChildToFork(forkId, child1);
  forkManager->addChildToFork(forkId, child2);

  // Disable the fork
  forkManager->disableFork(forkId);

  // Children should be terminated
  Process *proc1 = scheduler->getProcess(child1);
  Process *proc2 = scheduler->getProcess(child2);
  EXPECT_EQ(proc1->getState(), ProcessState::Terminated);
  EXPECT_EQ(proc2->getState(), ProcessState::Terminated);
}

TEST_F(ForkJoinTest, DisableForkCompletesForWaitFork) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child = scheduler->registerProcess("child", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::JoinNone);
  forkManager->addChildToFork(forkId, child);

  EXPECT_FALSE(forkManager->waitFork(parent));
  EXPECT_TRUE(forkManager->hasActiveChildren(parent));

  forkManager->disableFork(forkId);

  EXPECT_TRUE(forkManager->waitFork(parent));
  EXPECT_FALSE(forkManager->hasActiveChildren(parent));
}

TEST_F(ForkJoinTest, WaitForkChecksAllJoinNone) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child1 = scheduler->registerProcess("child1", []() {});
  ProcessId child2 = scheduler->registerProcess("child2", []() {});

  // Create two join_none forks
  ForkId fork1 = forkManager->createFork(parent, ForkJoinType::JoinNone);
  forkManager->addChildToFork(fork1, child1);

  ForkId fork2 = forkManager->createFork(parent, ForkJoinType::JoinNone);
  forkManager->addChildToFork(fork2, child2);

  // wait_fork should not complete until both are done
  EXPECT_FALSE(forkManager->waitFork(parent));

  forkManager->markChildComplete(child1);
  EXPECT_FALSE(forkManager->waitFork(parent));

  forkManager->markChildComplete(child2);
  EXPECT_TRUE(forkManager->waitFork(parent));
}

TEST_F(ForkJoinTest, GetForksForParent) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});

  ForkId fork1 = forkManager->createFork(parent, ForkJoinType::Join);
  ForkId fork2 = forkManager->createFork(parent, ForkJoinType::JoinAny);
  ForkId fork3 = forkManager->createFork(parent, ForkJoinType::JoinNone);

  auto forks = forkManager->getForksForParent(parent);
  EXPECT_EQ(forks.size(), 3u);
  EXPECT_EQ(forks[0], fork1);
  EXPECT_EQ(forks[1], fork2);
  EXPECT_EQ(forks[2], fork3);
}

TEST_F(ForkJoinTest, GetForkGroupForChild) {
  ProcessId parent = scheduler->registerProcess("parent", []() {});
  ProcessId child = scheduler->registerProcess("child", []() {});

  ForkId forkId = forkManager->createFork(parent, ForkJoinType::Join);
  forkManager->addChildToFork(forkId, child);

  ForkGroup *group = forkManager->getForkGroupForChild(child);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->id, forkId);
}

} // namespace
