//===- SimulationControlTest.cpp - Tests for SimulationControl ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimulationControl.h"
#include "gtest/gtest.h"
#include <string>
#include <sstream>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class SimulationControlTest : public ::testing::Test {
protected:
  void SetUp() override {
    SimulationControl::Config config;
    config.messageOutput = &llvm::nulls(); // Suppress output for tests
    control = std::make_unique<SimulationControl>(config);
  }

  void TearDown() override { control.reset(); }

  std::unique_ptr<SimulationControl> control;
};

//===----------------------------------------------------------------------===//
// Status Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, InitialStatus) {
  EXPECT_EQ(control->getStatus(), SimulationStatus::Running);
  EXPECT_TRUE(control->shouldContinue());
}

TEST_F(SimulationControlTest, StatusNames) {
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Running), "running");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Completed), "completed");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Stopped), "stopped");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Finished), "finished");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::ErrorLimit), "error_limit");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Timeout), "timeout");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Fatal), "fatal");
  EXPECT_STREQ(getSimulationStatusName(SimulationStatus::Aborted), "aborted");
}

//===----------------------------------------------------------------------===//
// Finish Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Finish) {
  control->finish(0);

  EXPECT_EQ(control->getStatus(), SimulationStatus::Finished);
  EXPECT_FALSE(control->shouldContinue());
  EXPECT_EQ(control->getExitCode(), 0);
}

TEST_F(SimulationControlTest, FinishWithExitCode) {
  control->finish(42);

  EXPECT_EQ(control->getStatus(), SimulationStatus::Finished);
  EXPECT_EQ(control->getExitCode(), 42);
}

TEST_F(SimulationControlTest, FinishCallback) {
  int callbackCode = -1;
  control->setFinishCallback([&](int code) { callbackCode = code; });

  control->finish(5);

  EXPECT_EQ(callbackCode, 5);
}

TEST_F(SimulationControlTest, FinishOnlyOnce) {
  control->finish(1);
  control->finish(2); // Should be ignored

  EXPECT_EQ(control->getExitCode(), 1);
}

//===----------------------------------------------------------------------===//
// Stop Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Stop) {
  control->stop();

  EXPECT_EQ(control->getStatus(), SimulationStatus::Stopped);
  EXPECT_FALSE(control->shouldContinue());
}

TEST_F(SimulationControlTest, StopCallback) {
  bool callbackCalled = false;
  control->setStopCallback([&]() { callbackCalled = true; });

  control->stop();

  EXPECT_TRUE(callbackCalled);
}

//===----------------------------------------------------------------------===//
// Abort Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Abort) {
  control->abort();

  EXPECT_EQ(control->getStatus(), SimulationStatus::Aborted);
  EXPECT_FALSE(control->shouldContinue());
}

//===----------------------------------------------------------------------===//
// Message Reporting Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, MessageSeverityNames) {
  EXPECT_STREQ(getMessageSeverityName(MessageSeverity::Info), "INFO");
  EXPECT_STREQ(getMessageSeverityName(MessageSeverity::Warning), "WARNING");
  EXPECT_STREQ(getMessageSeverityName(MessageSeverity::Error), "ERROR");
  EXPECT_STREQ(getMessageSeverityName(MessageSeverity::Fatal), "FATAL");
}

TEST_F(SimulationControlTest, InfoMessage) {
  control->info("TEST", "This is info");

  EXPECT_EQ(control->getInfoCount(), 1u);
  EXPECT_EQ(control->getWarningCount(), 0u);
  EXPECT_EQ(control->getErrorCount(), 0u);
}

TEST_F(SimulationControlTest, WarningMessage) {
  control->warning("TEST", "This is a warning");

  EXPECT_EQ(control->getWarningCount(), 1u);
}

TEST_F(SimulationControlTest, ErrorMessage) {
  control->error("TEST", "This is an error");

  EXPECT_EQ(control->getErrorCount(), 1u);
}

TEST_F(SimulationControlTest, FatalMessage) {
  control->fatal("TEST", "This is fatal");

  EXPECT_EQ(control->getStatus(), SimulationStatus::Fatal);
  EXPECT_EQ(control->getErrorCount(), 1u);
}

TEST_F(SimulationControlTest, MessageCallback) {
  SimulationMessage receivedMsg(MessageSeverity::Info, "", "");

  control->setMessageCallback([&](const SimulationMessage& msg) {
    receivedMsg = msg;
  });

  control->addFilter(MessageFilter("*", MessageAction::Callback));
  control->info("TESTID", "Test message");

  EXPECT_EQ(receivedMsg.id, "TESTID");
  EXPECT_EQ(receivedMsg.text, "Test message");
}

//===----------------------------------------------------------------------===//
// Message History Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, MessageHistory) {
  control->info("ID1", "Message 1");
  control->warning("ID2", "Message 2");
  control->error("ID3", "Message 3");

  auto& history = control->getMessageHistory();
  EXPECT_EQ(history.size(), 3u);
  EXPECT_EQ(history[0].id, "ID1");
  EXPECT_EQ(history[1].id, "ID2");
  EXPECT_EQ(history[2].id, "ID3");
}

TEST_F(SimulationControlTest, ClearMessageHistory) {
  control->info("TEST", "Message");
  EXPECT_FALSE(control->getMessageHistory().empty());

  control->clearMessageHistory();
  EXPECT_TRUE(control->getMessageHistory().empty());
}

TEST_F(SimulationControlTest, MaxHistorySize) {
  control->setMaxHistorySize(3);

  for (int i = 0; i < 5; ++i) {
    control->info("TEST", "Message " + std::to_string(i));
  }

  EXPECT_EQ(control->getMessageHistory().size(), 3u);
}

//===----------------------------------------------------------------------===//
// Message Counts Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, ResetCounts) {
  control->info("TEST", "Info");
  control->warning("TEST", "Warning");
  control->error("TEST", "Error");

  EXPECT_GT(control->getInfoCount(), 0u);
  EXPECT_GT(control->getWarningCount(), 0u);
  EXPECT_GT(control->getErrorCount(), 0u);

  control->resetCounts();

  EXPECT_EQ(control->getInfoCount(), 0u);
  EXPECT_EQ(control->getWarningCount(), 0u);
  EXPECT_EQ(control->getErrorCount(), 0u);
}

//===----------------------------------------------------------------------===//
// Error Limits Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, ErrorLimit) {
  control->setMaxErrors(3);

  control->error("TEST", "Error 1");
  control->error("TEST", "Error 2");
  EXPECT_TRUE(control->shouldContinue());

  control->error("TEST", "Error 3");
  EXPECT_EQ(control->getStatus(), SimulationStatus::ErrorLimit);
  EXPECT_FALSE(control->shouldContinue());
}

//===----------------------------------------------------------------------===//
// Message Filtering Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, FilterByAction) {
  // Suppress all TEST_* messages
  control->addFilter(MessageFilter("TEST_*", MessageAction::Count));

  control->info("TEST_1", "Should be filtered");
  control->info("TEST_2", "Should be filtered");
  control->info("OTHER", "Should be displayed");

  // Counts should still be recorded
  EXPECT_EQ(control->getInfoCount(), 3u);

  auto& stats = control->getStatistics();
  EXPECT_EQ(stats.messagesFiltered, 2u);
}

TEST_F(SimulationControlTest, FilterWithSeverity) {
  MessageFilter filter("TEST", MessageSeverity::Warning, MessageAction::Count);
  control->addFilter(filter);

  control->info("TEST", "Info not filtered");
  control->warning("TEST", "Warning filtered");

  auto& stats = control->getStatistics();
  EXPECT_EQ(stats.messagesFiltered, 1u);
}

TEST_F(SimulationControlTest, RemoveFilter) {
  control->addFilter(MessageFilter("TEST", MessageAction::Count));
  control->removeFilter("TEST");

  // Filter should be removed, messages displayed normally
  auto& stats = control->getStatistics();
  size_t before = stats.messagesFiltered;

  control->info("TEST", "Should not be filtered");

  EXPECT_EQ(stats.messagesFiltered, before);
}

TEST_F(SimulationControlTest, ClearFilters) {
  control->addFilter(MessageFilter("A", MessageAction::Count));
  control->addFilter(MessageFilter("B", MessageAction::Count));

  control->clearFilters();

  // No filters, nothing should be filtered
  control->info("A", "Message A");
  control->info("B", "Message B");

  auto& stats = control->getStatistics();
  EXPECT_EQ(stats.messagesFiltered, 0u);
}

TEST_F(SimulationControlTest, SetMessageAction) {
  control->setMessageAction("CUSTOM_ID", MessageAction::Stop);

  control->info("CUSTOM_ID", "This should stop simulation");

  EXPECT_EQ(control->getStatus(), SimulationStatus::Stopped);
}

TEST_F(SimulationControlTest, SetSeverityOverride) {
  // Make info messages behave as errors
  control->setSeverityOverride("PROMOTE", MessageSeverity::Error);

  control->info("PROMOTE", "This is promoted to error");

  EXPECT_EQ(control->getErrorCount(), 1u);
}

//===----------------------------------------------------------------------===//
// Timeout Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, GlobalTimeout) {
  control->setGlobalTimeout(1000000);
  EXPECT_EQ(control->getGlobalTimeout(), 1000000u);
}

TEST_F(SimulationControlTest, HasTimedOut) {
  control->setGlobalTimeout(1000);

  SimTime earlyTime(500);
  EXPECT_FALSE(control->hasTimedOut(earlyTime));

  SimTime lateTime(1500);
  EXPECT_TRUE(control->hasTimedOut(lateTime));
}

TEST_F(SimulationControlTest, NoTimeoutWhenZero) {
  control->setGlobalTimeout(0);

  SimTime largeTime(9999999999);
  EXPECT_FALSE(control->hasTimedOut(largeTime));
}

//===----------------------------------------------------------------------===//
// Watchdog Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, WatchdogDisabledByDefault) {
  EXPECT_FALSE(control->getWatchdog().isEnabled());
}

TEST_F(SimulationControlTest, EnableWatchdog) {
  control->enableWatchdog(10000);

  EXPECT_TRUE(control->getWatchdog().isEnabled());
  EXPECT_EQ(control->getWatchdog().getTimeout(), 10000u);
}

TEST_F(SimulationControlTest, DisableWatchdog) {
  control->enableWatchdog(10000);
  control->disableWatchdog();

  EXPECT_FALSE(control->getWatchdog().isEnabled());
}

TEST_F(SimulationControlTest, KickWatchdog) {
  control->enableWatchdog(10000);

  // Initially at time 0
  EXPECT_FALSE(control->getWatchdog().hasTimedOut(5000));

  // Kick at time 5000
  control->kickWatchdog(5000);

  // Check timeout relative to kick
  EXPECT_FALSE(control->getWatchdog().hasTimedOut(10000)); // 5000 since kick
  EXPECT_TRUE(control->getWatchdog().hasTimedOut(20000));  // 15000 since kick
}

//===----------------------------------------------------------------------===//
// Watchdog Class Tests
//===----------------------------------------------------------------------===//

TEST(WatchdogTest, Enable) {
  Watchdog wd;

  EXPECT_FALSE(wd.isEnabled());

  wd.enable(5000);
  EXPECT_TRUE(wd.isEnabled());
  EXPECT_EQ(wd.getTimeout(), 5000u);
}

TEST(WatchdogTest, Disable) {
  Watchdog wd;
  wd.enable(5000);
  wd.disable();

  EXPECT_FALSE(wd.isEnabled());
}

TEST(WatchdogTest, HasTimedOut) {
  Watchdog wd;
  wd.enable(1000);

  EXPECT_FALSE(wd.hasTimedOut(500));
  EXPECT_FALSE(wd.hasTimedOut(1000));
  EXPECT_TRUE(wd.hasTimedOut(1001));
}

TEST(WatchdogTest, Kick) {
  Watchdog wd;
  wd.enable(1000);
  wd.kick(500);

  EXPECT_FALSE(wd.hasTimedOut(1000));
  EXPECT_FALSE(wd.hasTimedOut(1500));
  EXPECT_TRUE(wd.hasTimedOut(1501));
}

TEST(WatchdogTest, TimeoutCallback) {
  Watchdog wd;
  bool triggered = false;

  wd.setTimeoutCallback([&]() { triggered = true; });
  wd.triggerTimeout();

  EXPECT_TRUE(triggered);
}

//===----------------------------------------------------------------------===//
// Verbosity Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Verbosity) {
  control->setVerbosity(3);
  EXPECT_EQ(control->getVerbosity(), 3);
}

TEST_F(SimulationControlTest, ShouldDisplay) {
  control->setVerbosity(2);

  EXPECT_TRUE(control->shouldDisplay(0));
  EXPECT_TRUE(control->shouldDisplay(1));
  EXPECT_TRUE(control->shouldDisplay(2));
  EXPECT_FALSE(control->shouldDisplay(3));
  EXPECT_FALSE(control->shouldDisplay(4));
}

//===----------------------------------------------------------------------===//
// UVM Compatibility Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, UVMReportInfo) {
  control->setVerbosity(2);

  control->uvmReportInfo("UVM_TEST", "Info at verbosity 2", 2);
  EXPECT_EQ(control->getInfoCount(), 1u);

  control->uvmReportInfo("UVM_TEST", "Info at verbosity 3", 3);
  EXPECT_EQ(control->getInfoCount(), 1u); // Should not be counted (filtered)
}

TEST_F(SimulationControlTest, UVMReportWarning) {
  control->uvmReportWarning("UVM_TEST", "Warning message");
  EXPECT_EQ(control->getWarningCount(), 1u);
}

TEST_F(SimulationControlTest, UVMReportError) {
  control->uvmReportError("UVM_TEST", "Error message");
  EXPECT_EQ(control->getErrorCount(), 1u);
}

TEST_F(SimulationControlTest, UVMReportFatal) {
  control->uvmReportFatal("UVM_TEST", "Fatal message");
  EXPECT_EQ(control->getStatus(), SimulationStatus::Fatal);
}

//===----------------------------------------------------------------------===//
// Statistics Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Statistics) {
  control->info("TEST", "Info");
  control->finish(0);
  control->stop(); // Should be ignored since already finished

  auto& stats = control->getStatistics();
  EXPECT_GE(stats.messagesReported, 1u);
  EXPECT_EQ(stats.finishCalls, 1u);
  EXPECT_EQ(stats.stopCalls, 0u); // Stop was ignored
}

//===----------------------------------------------------------------------===//
// Reset Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, Reset) {
  control->info("TEST", "Message");
  control->error("TEST", "Error");
  control->finish(1);
  control->enableWatchdog(1000);

  control->reset();

  EXPECT_EQ(control->getStatus(), SimulationStatus::Running);
  EXPECT_EQ(control->getExitCode(), 0);
  EXPECT_EQ(control->getInfoCount(), 0u);
  EXPECT_EQ(control->getErrorCount(), 0u);
  EXPECT_TRUE(control->getMessageHistory().empty());
  EXPECT_FALSE(control->getWatchdog().isEnabled());
}

//===----------------------------------------------------------------------===//
// SimulationControlGuard Tests
//===----------------------------------------------------------------------===//

TEST_F(SimulationControlTest, ControlGuard) {
  // Add a filter that suppresses TEST messages
  {
    SimulationControlGuard guard(*control, "TEST");

    control->info("TEST", "Should be suppressed");
    control->info("OTHER", "Should not be suppressed");

    auto& stats = control->getStatistics();
    EXPECT_EQ(stats.messagesFiltered, 1u);
  }

  // After guard destruction, filter should be removed
  size_t filteredBefore = control->getStatistics().messagesFiltered;
  control->info("TEST", "Should not be suppressed now");

  EXPECT_EQ(control->getStatistics().messagesFiltered, filteredBefore);
}

//===----------------------------------------------------------------------===//
// SimulationMessage Tests
//===----------------------------------------------------------------------===//

TEST(SimulationMessageTest, Construction) {
  SimulationMessage msg(MessageSeverity::Error, "TEST_ID", "Test message");

  EXPECT_EQ(msg.severity, MessageSeverity::Error);
  EXPECT_EQ(msg.id, "TEST_ID");
  EXPECT_EQ(msg.text, "Test message");
}

TEST(SimulationMessageTest, WithFileInfo) {
  SimulationMessage msg(MessageSeverity::Warning, "WARN_ID", "Warning text",
                        "test.sv", 42);

  EXPECT_EQ(msg.fileName, "test.sv");
  EXPECT_EQ(msg.lineNumber, 42);
}
