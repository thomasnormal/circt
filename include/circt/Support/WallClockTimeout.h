//===- WallClockTimeout.h - Wall-clock timeout helper --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper for triggering a callback after a wall-clock timeout.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_WALLCLOCKTIMEOUT_H
#define CIRCT_SUPPORT_WALLCLOCKTIMEOUT_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace circt {

class WallClockTimeout {
public:
  using Callback = std::function<void()>;

  WallClockTimeout(std::chrono::milliseconds timeout, Callback callback);
  ~WallClockTimeout();

  WallClockTimeout(const WallClockTimeout &) = delete;
  WallClockTimeout &operator=(const WallClockTimeout &) = delete;

  /// Stop the timer and join the worker thread.
  void cancel();

  /// Return true if the timeout callback fired.
  bool hasFired() const { return fired.load(); }

private:
  void run();

  std::chrono::milliseconds timeout;
  Callback callback;
  std::atomic<bool> stop{false};
  std::atomic<bool> fired{false};
  std::mutex mutex;
  std::condition_variable cv;
  std::thread worker;
};

} // namespace circt

#endif // CIRCT_SUPPORT_WALLCLOCKTIMEOUT_H
