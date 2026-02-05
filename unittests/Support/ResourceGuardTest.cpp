//===- ResourceGuardTest.cpp - Resource guard tests -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ResourceGuard.h"
#include "gtest/gtest.h"

#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace circt;

TEST(ResourceGuardTest, ParseMegabytes) {
  EXPECT_EQ(parseMegabytes("0"), std::optional<uint64_t>(0));
  EXPECT_EQ(parseMegabytes("  123 "), std::optional<uint64_t>(123));
  EXPECT_FALSE(parseMegabytes(""));
  EXPECT_FALSE(parseMegabytes(" "));
  EXPECT_FALSE(parseMegabytes("12MB"));
  EXPECT_FALSE(parseMegabytes("-1"));
}

TEST(ResourceGuardTest, WarnsWhenAllLimitsDisabledExplicitly) {
  ::setenv("CIRCT_MAX_RSS_MB", "0", 1);
  ::setenv("CIRCT_MAX_MALLOC_MB", "0", 1);
  ::setenv("CIRCT_MAX_VMEM_MB", "0", 1);
  ::setenv("CIRCT_MAX_WALL_MS", "0", 1);

  testing::internal::CaptureStderr();
  installResourceGuard();
  std::string stderrText = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      stderrText.find(
          "warning: resource guard enabled but all limits are disabled"),
      std::string::npos);

  ::unsetenv("CIRCT_MAX_RSS_MB");
  ::unsetenv("CIRCT_MAX_MALLOC_MB");
  ::unsetenv("CIRCT_MAX_VMEM_MB");
  ::unsetenv("CIRCT_MAX_WALL_MS");
}

TEST(ResourceGuardTest, PrintsEffectiveLimitsWhenVerbose) {
  ::setenv("CIRCT_RESOURCE_GUARD_VERBOSE", "1", 1);
  ::setenv("CIRCT_MAX_RSS_MB", "0", 1);
  ::setenv("CIRCT_MAX_MALLOC_MB", "0", 1);
  ::setenv("CIRCT_MAX_VMEM_MB", "0", 1);
  ::setenv("CIRCT_MAX_WALL_MS", "0", 1);
  ::setenv("CIRCT_RESOURCE_GUARD_INTERVAL_MS", "123", 1);

  testing::internal::CaptureStderr();
  installResourceGuard();
  std::string stderrText = testing::internal::GetCapturedStderr();
  EXPECT_NE(stderrText.find("note: resource guard: enabled"), std::string::npos);
  EXPECT_NE(stderrText.find("interval-ms=123"), std::string::npos);

  ::unsetenv("CIRCT_RESOURCE_GUARD_VERBOSE");
  ::unsetenv("CIRCT_MAX_RSS_MB");
  ::unsetenv("CIRCT_MAX_MALLOC_MB");
  ::unsetenv("CIRCT_MAX_VMEM_MB");
  ::unsetenv("CIRCT_MAX_WALL_MS");
  ::unsetenv("CIRCT_RESOURCE_GUARD_INTERVAL_MS");
}

TEST(ResourceGuardTest, ReportsPhaseOnAbort) {
  EXPECT_EXIT(
      {
        // Keep the limit low enough to reliably trigger in the child process.
        ::setenv("CIRCT_MAX_RSS_MB", "16", 1);
        ::unsetenv("CIRCT_MAX_WALL_MS");
        setResourceGuardPhase("unit-test");
        installResourceGuard();

        // Force RSS growth beyond the limit and give the watchdog time to
        // sample.
        std::vector<char> buffer(64 * 1024 * 1024, 0);
        for (size_t i = 0; i < buffer.size(); i += 4096)
          buffer[i] = 1;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::exit(0);
      },
      ::testing::ExitedWithCode(1), "phase: unit-test");
}

TEST(ResourceGuardTest, ReportsWallTimeOnAbort) {
  EXPECT_EXIT(
      {
        ::unsetenv("CIRCT_MAX_RSS_MB");
        ::unsetenv("CIRCT_MAX_MALLOC_MB");
        ::unsetenv("CIRCT_MAX_VMEM_MB");

        // Use a short wall-clock limit and a fast polling interval to keep the
        // test reliable and quick.
        ::setenv("CIRCT_MAX_WALL_MS", "50", 1);
        ::setenv("CIRCT_RESOURCE_GUARD_INTERVAL_MS", "1", 1);
        setResourceGuardPhase("unit-test-wall");
        installResourceGuard();

        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::exit(0);
      },
      ::testing::ExitedWithCode(1), "wall time.*phase: unit-test-wall");
}
