//===- WallClockTimeoutTest.cpp - Wall-clock timeout tests ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/WallClockTimeout.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <thread>

using namespace circt;

TEST(WallClockTimeoutTest, FiresAfterTimeout) {
  std::atomic<bool> fired{false};
  WallClockTimeout timeout(std::chrono::milliseconds(50),
                           [&]() { fired.store(true); });

  auto start = std::chrono::steady_clock::now();
  while (!fired.load()) {
    if (std::chrono::steady_clock::now() - start >
        std::chrono::milliseconds(500))
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  EXPECT_TRUE(fired.load());
  EXPECT_TRUE(timeout.hasFired());
}

TEST(WallClockTimeoutTest, CancelPreventsFire) {
  std::atomic<bool> fired{false};
  {
    WallClockTimeout timeout(std::chrono::milliseconds(200),
                             [&]() { fired.store(true); });
    timeout.cancel();
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  EXPECT_FALSE(fired.load());
}
