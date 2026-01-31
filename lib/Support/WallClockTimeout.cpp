//===- WallClockTimeout.cpp - Wall-clock timeout helper -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/WallClockTimeout.h"

using namespace circt;

WallClockTimeout::WallClockTimeout(std::chrono::milliseconds timeout,
                                   Callback callback)
    : timeout(timeout), callback(std::move(callback)) {
  if (timeout.count() == 0)
    return;
  worker = std::thread([this]() { run(); });
}

WallClockTimeout::~WallClockTimeout() { cancel(); }

void WallClockTimeout::cancel() {
  stop.store(true);
  cv.notify_all();
  if (worker.joinable())
    worker.join();
}

void WallClockTimeout::run() {
  std::unique_lock<std::mutex> lock(mutex);
  if (cv.wait_for(lock, timeout, [this]() { return stop.load(); }))
    return;
  fired.store(true);
  lock.unlock();
  if (callback)
    callback();
}
