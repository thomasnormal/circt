//===- UVMPhaseManagerTest.cpp - Tests for UVMPhaseManager ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/UVMPhaseManager.h"
#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "gtest/gtest.h"
#include <string>
#include <vector>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class UVMPhaseManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    scheduler = std::make_unique<ProcessScheduler>();
    phaseManager = std::make_unique<UVMPhaseManager>(*scheduler);
  }

  void TearDown() override {
    phaseManager.reset();
    scheduler.reset();
  }

  std::unique_ptr<ProcessScheduler> scheduler;
  std::unique_ptr<UVMPhaseManager> phaseManager;
};

//===----------------------------------------------------------------------===//
// Phase Enumeration Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, PhaseName) {
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Build), "build");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Connect), "connect");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::EndOfElaboration), "end_of_elaboration");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::StartOfSimulation), "start_of_simulation");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Run), "run");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Extract), "extract");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Check), "check");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Report), "report");
  EXPECT_STREQ(getUVMPhaseName(UVMPhase::Final), "final");
}

TEST_F(UVMPhaseManagerTest, ParsePhase) {
  EXPECT_EQ(parseUVMPhase("build"), UVMPhase::Build);
  EXPECT_EQ(parseUVMPhase("connect"), UVMPhase::Connect);
  EXPECT_EQ(parseUVMPhase("run"), UVMPhase::Run);
  EXPECT_EQ(parseUVMPhase("final"), UVMPhase::Final);
  EXPECT_EQ(parseUVMPhase("invalid"), UVMPhase::None);
}

TEST_F(UVMPhaseManagerTest, TimeConsumingPhase) {
  EXPECT_FALSE(isTimeConsumingPhase(UVMPhase::Build));
  EXPECT_FALSE(isTimeConsumingPhase(UVMPhase::Connect));
  EXPECT_TRUE(isTimeConsumingPhase(UVMPhase::Run));
  EXPECT_FALSE(isTimeConsumingPhase(UVMPhase::Final));
}

//===----------------------------------------------------------------------===//
// Component Registration Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, RegisterComponent) {
  auto *callback = phaseManager->registerComponent("test_component");
  ASSERT_NE(callback, nullptr);
  EXPECT_EQ(callback->getName(), "test_component");
}

TEST_F(UVMPhaseManagerTest, GetComponent) {
  phaseManager->registerComponent("my_component");

  auto *found = phaseManager->getComponent("my_component");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->getName(), "my_component");

  auto *notFound = phaseManager->getComponent("nonexistent");
  EXPECT_EQ(notFound, nullptr);
}

TEST_F(UVMPhaseManagerTest, UnregisterComponent) {
  phaseManager->registerComponent("to_remove");
  EXPECT_NE(phaseManager->getComponent("to_remove"), nullptr);

  phaseManager->unregisterComponent("to_remove");
  EXPECT_EQ(phaseManager->getComponent("to_remove"), nullptr);
}

//===----------------------------------------------------------------------===//
// Phase Callback Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, PhaseCallback) {
  std::vector<UVMPhase> executedPhases;

  auto *callback = phaseManager->registerComponent("callback_test");
  callback->setPhaseCallback(UVMPhase::Build, [&](UVMPhase phase) {
    executedPhases.push_back(phase);
  });
  callback->setPhaseCallback(UVMPhase::Connect, [&](UVMPhase phase) {
    executedPhases.push_back(phase);
  });

  phaseManager->runPhase(UVMPhase::Build);
  phaseManager->runPhase(UVMPhase::Connect);

  ASSERT_EQ(executedPhases.size(), 2u);
  EXPECT_EQ(executedPhases[0], UVMPhase::Build);
  EXPECT_EQ(executedPhases[1], UVMPhase::Connect);
}

TEST_F(UVMPhaseManagerTest, MultipleComponentCallbacks) {
  std::vector<std::string> executionOrder;

  auto *comp1 = phaseManager->registerComponent("comp1");
  auto *comp2 = phaseManager->registerComponent("comp2");

  comp1->setPhaseCallback(UVMPhase::Build, [&](UVMPhase) {
    executionOrder.push_back("comp1");
  });
  comp2->setPhaseCallback(UVMPhase::Build, [&](UVMPhase) {
    executionOrder.push_back("comp2");
  });

  phaseManager->runPhase(UVMPhase::Build);

  ASSERT_EQ(executionOrder.size(), 2u);
  // Order depends on StringMap iteration order
}

//===----------------------------------------------------------------------===//
// Phase State Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, CurrentPhase) {
  EXPECT_EQ(phaseManager->getCurrentPhase(), UVMPhase::None);

  phaseManager->runPhase(UVMPhase::Build);
  EXPECT_EQ(phaseManager->getCurrentPhase(), UVMPhase::Build);
}

TEST_F(UVMPhaseManagerTest, IsInPhase) {
  phaseManager->runPhase(UVMPhase::Connect);

  EXPECT_TRUE(phaseManager->isInPhase(UVMPhase::Connect));
  EXPECT_FALSE(phaseManager->isInPhase(UVMPhase::Build));
}

TEST_F(UVMPhaseManagerTest, JumpToPhase) {
  phaseManager->runPhase(UVMPhase::Build);
  EXPECT_EQ(phaseManager->getCurrentPhase(), UVMPhase::Build);

  phaseManager->jumpToPhase(UVMPhase::Run);
  EXPECT_EQ(phaseManager->getCurrentPhase(), UVMPhase::Run);
}

//===----------------------------------------------------------------------===//
// Objection Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, RaiseObjection) {
  phaseManager->runPhase(UVMPhase::Run);

  auto id = phaseManager->raiseObjection("test_driver", "Waiting for stimulus");
  EXPECT_NE(id, InvalidObjectionId);
  EXPECT_TRUE(phaseManager->hasObjections());
  EXPECT_EQ(phaseManager->getObjectionCount(), 1u);
}

TEST_F(UVMPhaseManagerTest, DropObjection) {
  phaseManager->runPhase(UVMPhase::Run);

  auto id = phaseManager->raiseObjection("test_driver");
  EXPECT_TRUE(phaseManager->hasObjections());

  phaseManager->dropObjection(id);
  EXPECT_FALSE(phaseManager->hasObjections());
  EXPECT_EQ(phaseManager->getObjectionCount(), 0u);
}

TEST_F(UVMPhaseManagerTest, DropObjectionByName) {
  phaseManager->runPhase(UVMPhase::Run);

  phaseManager->raiseObjection("test_monitor");
  EXPECT_TRUE(phaseManager->hasObjections());

  phaseManager->dropObjection("test_monitor");
  EXPECT_FALSE(phaseManager->hasObjections());
}

TEST_F(UVMPhaseManagerTest, MultipleObjections) {
  phaseManager->runPhase(UVMPhase::Run);

  auto id1 = phaseManager->raiseObjection("comp1");
  auto id2 = phaseManager->raiseObjection("comp2");

  EXPECT_EQ(phaseManager->getObjectionCount(), 2u);

  phaseManager->dropObjection(id1);
  EXPECT_EQ(phaseManager->getObjectionCount(), 1u);
  EXPECT_TRUE(phaseManager->hasObjections());

  phaseManager->dropObjection(id2);
  EXPECT_FALSE(phaseManager->hasObjections());
}

TEST_F(UVMPhaseManagerTest, ObjectionCount) {
  phaseManager->runPhase(UVMPhase::Run);

  // Raising objection for same component increases count
  auto id = phaseManager->raiseObjection("driver");
  phaseManager->raiseObjection("driver"); // Same component

  // Should still be 1 objection (same component, increased count)
  EXPECT_EQ(phaseManager->getObjectionCount(), 1u);

  // Need to drop twice to fully remove
  phaseManager->dropObjection(id);
  EXPECT_TRUE(phaseManager->hasObjections()); // Count was 2, now 1

  phaseManager->dropObjection(id);
  EXPECT_FALSE(phaseManager->hasObjections()); // Count was 1, now removed
}

TEST_F(UVMPhaseManagerTest, AllDroppedCallback) {
  bool callbackFired = false;

  phaseManager->setAllDroppedCallback([&]() {
    callbackFired = true;
  });

  phaseManager->runPhase(UVMPhase::Run);
  auto id = phaseManager->raiseObjection("test");

  EXPECT_FALSE(callbackFired);

  phaseManager->dropObjection(id);
  EXPECT_TRUE(callbackFired);
}

//===----------------------------------------------------------------------===//
// Phase Control Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, DrainTime) {
  phaseManager->setDrainTime(1000);
  EXPECT_EQ(phaseManager->getDrainTime(), 1000u);
}

TEST_F(UVMPhaseManagerTest, RunPhaseTimeout) {
  phaseManager->setRunPhaseTimeout(1000000);
  EXPECT_EQ(phaseManager->getRunPhaseTimeout(), 1000000u);
}

TEST_F(UVMPhaseManagerTest, RequestPhaseEnd) {
  phaseManager->runPhase(UVMPhase::Run);

  EXPECT_FALSE(phaseManager->isPhaseEndRequested());

  phaseManager->requestPhaseEnd();
  EXPECT_TRUE(phaseManager->isPhaseEndRequested());
}

//===----------------------------------------------------------------------===//
// Statistics Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, Statistics) {
  phaseManager->registerComponent("comp1");
  phaseManager->registerComponent("comp2");

  phaseManager->runPhase(UVMPhase::Build);
  phaseManager->runPhase(UVMPhase::Connect);

  auto &stats = phaseManager->getStatistics();
  EXPECT_EQ(stats.componentsRegistered, 2u);
  EXPECT_EQ(stats.phasesExecuted, 2u);
}

TEST_F(UVMPhaseManagerTest, ObjectionStatistics) {
  phaseManager->runPhase(UVMPhase::Run);

  auto id1 = phaseManager->raiseObjection("comp1");
  auto id2 = phaseManager->raiseObjection("comp2");

  phaseManager->dropObjection(id1);
  phaseManager->dropObjection(id2);

  auto &stats = phaseManager->getStatistics();
  EXPECT_EQ(stats.objectionsRaised, 2u);
  EXPECT_EQ(stats.objectionsDropped, 2u);
}

//===----------------------------------------------------------------------===//
// Reset Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, Reset) {
  phaseManager->registerComponent("test");
  phaseManager->runPhase(UVMPhase::Build);
  phaseManager->raiseObjection("test");

  phaseManager->reset();

  EXPECT_EQ(phaseManager->getCurrentPhase(), UVMPhase::None);
  EXPECT_FALSE(phaseManager->hasObjections());
  EXPECT_EQ(phaseManager->getStatistics().phasesExecuted, 0u);
}

//===----------------------------------------------------------------------===//
// Run All Phases Tests
//===----------------------------------------------------------------------===//

TEST_F(UVMPhaseManagerTest, RunAllPhases) {
  std::vector<UVMPhase> phases;

  auto *callback = phaseManager->registerComponent("all_phases_test");

  for (uint8_t i = 0; i < static_cast<uint8_t>(UVMPhase::NumPhases); ++i) {
    callback->setPhaseCallback(static_cast<UVMPhase>(i), [&phases](UVMPhase p) {
      phases.push_back(p);
    });
  }

  phaseManager->runAllPhases();

  EXPECT_EQ(phases.size(), static_cast<size_t>(UVMPhase::NumPhases));
  EXPECT_EQ(phases[0], UVMPhase::Build);
  EXPECT_EQ(phases.back(), UVMPhase::Final);
}
