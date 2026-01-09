//===- PerformanceProfilerTest.cpp - PerformanceProfiler unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/PerformanceProfiler.h"
#include "gtest/gtest.h"
#include <thread>

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// HighResolutionTimer Tests
//===----------------------------------------------------------------------===//

TEST(HighResolutionTimerTest, BasicTiming) {
  HighResolutionTimer timer;

  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  auto duration = timer.stop();

  double ms = HighResolutionTimer::toMilliseconds(duration);
  EXPECT_GE(ms, 9.0);  // Allow some tolerance
  EXPECT_LT(ms, 100.0); // Should not take too long
}

TEST(HighResolutionTimerTest, Elapsed) {
  HighResolutionTimer timer;
  timer.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto elapsed1 = timer.elapsed();

  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto elapsed2 = timer.elapsed();

  EXPECT_GT(HighResolutionTimer::toMilliseconds(elapsed2),
            HighResolutionTimer::toMilliseconds(elapsed1));
}

TEST(HighResolutionTimerTest, Conversions) {
  auto duration = std::chrono::nanoseconds(1500000); // 1.5 ms

  EXPECT_NEAR(HighResolutionTimer::toSeconds(duration), 0.0015, 0.0001);
  EXPECT_NEAR(HighResolutionTimer::toMilliseconds(duration), 1.5, 0.01);
  EXPECT_NEAR(HighResolutionTimer::toMicroseconds(duration), 1500.0, 1.0);
}

//===----------------------------------------------------------------------===//
// ScopedProfiler Tests
//===----------------------------------------------------------------------===//

TEST(ScopedProfilerTest, BasicScoping) {
  PerformanceProfiler::Config config;
  config.enabled = true;
  PerformanceProfiler profiler(config);

  profiler.startSession("test");

  {
    ScopedProfiler scope(&profiler, ProfileCategory::ProcessExecution, "test");
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }

  const auto &stats = profiler.getCategoryStats(ProfileCategory::ProcessExecution);
  EXPECT_EQ(stats.sampleCount.load(), 1u);
  EXPECT_GT(stats.totalTimeNs.load(), 0u);
}

TEST(ScopedProfilerTest, ManualStop) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  ScopedProfiler scope(&profiler, ProfileCategory::EventScheduling, "test");
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  scope.stop();

  const auto &stats = profiler.getCategoryStats(ProfileCategory::EventScheduling);
  EXPECT_EQ(stats.sampleCount.load(), 1u);
}

TEST(ScopedProfilerTest, NullProfiler) {
  // Should not crash with null profiler
  ScopedProfiler scope(nullptr, ProfileCategory::Custom);
}

//===----------------------------------------------------------------------===//
// PerformanceProfiler Tests
//===----------------------------------------------------------------------===//

TEST(PerformanceProfilerTest, EnableDisable) {
  PerformanceProfiler profiler;

  EXPECT_TRUE(profiler.isEnabled());

  profiler.setEnabled(false);
  EXPECT_FALSE(profiler.isEnabled());

  profiler.setEnabled(true);
  EXPECT_TRUE(profiler.isEnabled());
}

TEST(PerformanceProfilerTest, StartEndSession) {
  PerformanceProfiler profiler;

  profiler.startSession("test_session");
  EXPECT_EQ(profiler.getSessionName(), "test_session");

  profiler.endSession();
  EXPECT_FALSE(profiler.isEnabled());
}

TEST(PerformanceProfilerTest, RecordSample) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  ProfileSample sample;
  sample.category = ProfileCategory::SignalUpdate;
  sample.name = "test_signal";
  sample.duration = std::chrono::nanoseconds(1000);
  sample.context = 42;

  profiler.recordSample(sample);

  const auto &stats = profiler.getCategoryStats(ProfileCategory::SignalUpdate);
  EXPECT_EQ(stats.sampleCount.load(), 1u);
  EXPECT_EQ(stats.totalTimeNs.load(), 1000u);
}

TEST(PerformanceProfilerTest, CategoryStatistics) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  // Record multiple samples
  for (int i = 0; i < 100; ++i) {
    profiler.endOperation(ProfileCategory::DeltaCycle,
                          std::chrono::nanoseconds(1000 + i * 10),
                          "delta", i);
  }

  const auto &stats = profiler.getCategoryStats(ProfileCategory::DeltaCycle);
  EXPECT_EQ(stats.sampleCount.load(), 100u);
  EXPECT_GE(stats.totalTimeNs.load(), 100000u);
  EXPECT_EQ(stats.minTimeNs.load(), 1000u);
  EXPECT_EQ(stats.maxTimeNs.load(), 1990u);

  double avg = stats.getAverageNs();
  EXPECT_GT(avg, 1000);
  EXPECT_LT(avg, 2000);
}

TEST(PerformanceProfilerTest, Reset) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  profiler.endOperation(ProfileCategory::EventExecution,
                        std::chrono::nanoseconds(5000));

  const auto &stats1 = profiler.getCategoryStats(ProfileCategory::EventExecution);
  EXPECT_EQ(stats1.sampleCount.load(), 1u);

  profiler.reset();

  const auto &stats2 = profiler.getCategoryStats(ProfileCategory::EventExecution);
  EXPECT_EQ(stats2.sampleCount.load(), 0u);
}

//===----------------------------------------------------------------------===//
// Process Profiling Tests
//===----------------------------------------------------------------------===//

TEST(PerformanceProfilerTest, ProcessProfiling) {
  PerformanceProfiler::Config config;
  config.profileProcesses = true;
  PerformanceProfiler profiler(config);
  profiler.startSession("test");

  profiler.registerProcess(1, "test_process");

  profiler.recordProcessExecution(1, 5000);
  profiler.recordProcessExecution(1, 3000);
  profiler.recordProcessTrigger(1);
  profiler.recordProcessTrigger(1);
  profiler.recordProcessTrigger(1);

  ProcessProfile *profile = profiler.getProcessProfile(1);
  EXPECT_NE(profile, nullptr);
  EXPECT_EQ(profile->name, "test_process");
  EXPECT_EQ(profile->executionCount.load(), 2u);
  EXPECT_EQ(profile->totalTimeNs.load(), 8000u);
  EXPECT_EQ(profile->triggerCount.load(), 3u);
  EXPECT_EQ(profile->getAverageTimeNs(), 4000.0);
}

TEST(PerformanceProfilerTest, HottestProcesses) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  profiler.registerProcess(1, "slow_process");
  profiler.registerProcess(2, "fast_process");
  profiler.registerProcess(3, "medium_process");

  profiler.recordProcessExecution(1, 10000);
  profiler.recordProcessExecution(2, 1000);
  profiler.recordProcessExecution(3, 5000);

  auto hottest = profiler.getHottestProcesses(2);
  EXPECT_EQ(hottest.size(), 2u);
  EXPECT_EQ(hottest[0]->name, "slow_process");
  EXPECT_EQ(hottest[1]->name, "medium_process");
}

//===----------------------------------------------------------------------===//
// Signal Profiling Tests
//===----------------------------------------------------------------------===//

TEST(PerformanceProfilerTest, SignalProfiling) {
  PerformanceProfiler::Config config;
  config.profileSignals = true;
  PerformanceProfiler profiler(config);
  profiler.startSession("test");

  profiler.registerSignal(100, "test_signal");

  profiler.recordSignalUpdate(100, 500);
  profiler.recordSignalUpdate(100, 300);
  profiler.recordSignalTrigger(100);

  SignalProfile *profile = profiler.getSignalProfile(100);
  EXPECT_NE(profile, nullptr);
  EXPECT_EQ(profile->name, "test_signal");
  EXPECT_EQ(profile->updateCount.load(), 2u);
  EXPECT_EQ(profile->updateTimeNs.load(), 800u);
  EXPECT_EQ(profile->triggerCount.load(), 1u);
}

TEST(PerformanceProfilerTest, MostActiveSignals) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  profiler.registerSignal(1, "signal_a");
  profiler.registerSignal(2, "signal_b");
  profiler.registerSignal(3, "signal_c");

  // Make signal_c most active
  for (int i = 0; i < 100; ++i) {
    profiler.recordSignalUpdate(3, 100);
  }
  for (int i = 0; i < 50; ++i) {
    profiler.recordSignalUpdate(1, 100);
  }
  for (int i = 0; i < 10; ++i) {
    profiler.recordSignalUpdate(2, 100);
  }

  auto active = profiler.getMostActiveSignals(2);
  EXPECT_EQ(active.size(), 2u);
  EXPECT_EQ(active[0]->name, "signal_c");
  EXPECT_EQ(active[1]->name, "signal_a");
}

//===----------------------------------------------------------------------===//
// Histogram Tests
//===----------------------------------------------------------------------===//

TEST(PerformanceProfilerTest, Histograms) {
  PerformanceProfiler::Config config;
  config.collectHistograms = true;
  config.histogramBuckets = 10;
  PerformanceProfiler profiler(config);
  profiler.startSession("test");

  // Record samples with varying durations
  profiler.endOperation(ProfileCategory::EventScheduling,
                        std::chrono::nanoseconds(100));
  profiler.endOperation(ProfileCategory::EventScheduling,
                        std::chrono::nanoseconds(1000));
  profiler.endOperation(ProfileCategory::EventScheduling,
                        std::chrono::nanoseconds(10000));

  const auto &hist = profiler.getHistogram(ProfileCategory::EventScheduling);
  EXPECT_EQ(hist.size(), 10u);

  // At least some buckets should have counts
  size_t totalCount = 0;
  for (size_t count : hist) {
    totalCount += count;
  }
  EXPECT_EQ(totalCount, 3u);
}

TEST(PerformanceProfilerTest, HistogramBuckets) {
  PerformanceProfiler::Config config;
  config.histogramBuckets = 10;
  PerformanceProfiler profiler(config);

  auto buckets = profiler.getHistogramBuckets();
  EXPECT_EQ(buckets.size(), 10u);

  // Buckets should be in increasing order (log scale)
  for (size_t i = 1; i < buckets.size(); ++i) {
    EXPECT_GT(buckets[i], buckets[i - 1]);
  }
}

//===----------------------------------------------------------------------===//
// BottleneckAnalyzer Tests
//===----------------------------------------------------------------------===//

TEST(BottleneckAnalyzerTest, NoBottlenecks) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  // Record balanced samples
  profiler.endOperation(ProfileCategory::ProcessExecution,
                        std::chrono::nanoseconds(1000));
  profiler.endOperation(ProfileCategory::SignalUpdate,
                        std::chrono::nanoseconds(1000));

  BottleneckAnalyzer analyzer(profiler);
  auto bottlenecks = analyzer.analyze();

  // May or may not have bottlenecks depending on thresholds
  // Just verify it doesn't crash
}

TEST(BottleneckAnalyzerTest, SchedulingOverhead) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  // Create heavy scheduling overhead
  for (int i = 0; i < 100; ++i) {
    profiler.endOperation(ProfileCategory::EventScheduling,
                          std::chrono::nanoseconds(10000));
  }
  for (int i = 0; i < 100; ++i) {
    profiler.endOperation(ProfileCategory::ProcessExecution,
                          std::chrono::nanoseconds(1000));
  }

  BottleneckAnalyzer analyzer(profiler);
  EXPECT_TRUE(analyzer.hasSchedulingOverhead());
  EXPECT_EQ(analyzer.getDominantCategory(), ProfileCategory::EventScheduling);
}

TEST(BottleneckAnalyzerTest, SlowProcesses) {
  PerformanceProfiler::Config config;
  config.profileProcesses = true;
  PerformanceProfiler profiler(config);
  profiler.startSession("test");

  profiler.registerProcess(1, "very_slow_process");
  profiler.recordProcessExecution(1, 1000000); // 1ms

  BottleneckAnalyzer analyzer(profiler);
  EXPECT_TRUE(analyzer.hasSlowProcesses());
}

//===----------------------------------------------------------------------===//
// Profile Category Name Tests
//===----------------------------------------------------------------------===//

TEST(ProfileCategoryTest, Names) {
  EXPECT_STREQ(getProfileCategoryName(ProfileCategory::EventScheduling),
               "EventScheduling");
  EXPECT_STREQ(getProfileCategoryName(ProfileCategory::ProcessExecution),
               "ProcessExecution");
  EXPECT_STREQ(getProfileCategoryName(ProfileCategory::SignalUpdate),
               "SignalUpdate");
  EXPECT_STREQ(getProfileCategoryName(ProfileCategory::DeltaCycle),
               "DeltaCycle");
  EXPECT_STREQ(getProfileCategoryName(ProfileCategory::Custom),
               "Custom");
}

//===----------------------------------------------------------------------===//
// Export Tests
//===----------------------------------------------------------------------===//

TEST(PerformanceProfilerTest, ExportJSON) {
  PerformanceProfiler profiler;
  profiler.startSession("test_export");

  profiler.endOperation(ProfileCategory::ProcessExecution,
                        std::chrono::nanoseconds(5000), "proc1", 1);

  std::string output;
  llvm::raw_string_ostream os(output);
  profiler.exportJSON(os);

  // Basic JSON validation
  EXPECT_NE(output.find("\"session\""), std::string::npos);
  EXPECT_NE(output.find("test_export"), std::string::npos);
  EXPECT_NE(output.find("\"categories\""), std::string::npos);
  EXPECT_NE(output.find("\"samples\""), std::string::npos);
}

TEST(PerformanceProfilerTest, ExportCSV) {
  PerformanceProfiler profiler;
  profiler.startSession("test");

  profiler.endOperation(ProfileCategory::EventExecution,
                        std::chrono::nanoseconds(1000), "test", 0);

  std::string output;
  llvm::raw_string_ostream os(output);
  profiler.exportCSV(os);

  // Should have header
  EXPECT_NE(output.find("category,name,duration_ns,context"), std::string::npos);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
