//===- ResourceGuard.cpp - Memory usage safeguards --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ResourceGuard.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace circt;
namespace cl = llvm::cl;

static cl::OptionCategory resourceGuardCategory("Resource Guard Options");

static cl::opt<unsigned> optMaxRSSMB(
    "max-rss-mb",
    cl::desc("Abort if resident set size (RSS) exceeds this many megabytes "
             "(0 = disabled; env: CIRCT_MAX_RSS_MB)"),
    cl::init(0), cl::cat(resourceGuardCategory));

static cl::opt<unsigned> optMaxMallocMB(
    "max-malloc-mb",
    cl::desc("Abort if malloc heap usage exceeds this many megabytes "
             "(0 = disabled; env: CIRCT_MAX_MALLOC_MB)"),
    cl::init(0), cl::cat(resourceGuardCategory));

static cl::opt<unsigned> optMaxVMemMB(
    "max-vmem-mb",
    cl::desc("Set an address-space (virtual memory) soft limit in megabytes "
             "(0 = disabled; env: CIRCT_MAX_VMEM_MB)"),
    cl::init(0), cl::cat(resourceGuardCategory));

static cl::opt<unsigned> optGuardIntervalMs(
    "resource-guard-interval-ms",
    cl::desc("Polling interval (ms) for the resource guard watchdog thread"),
    cl::init(250), cl::cat(resourceGuardCategory));

static std::optional<uint64_t> parseEnvMegabytes(llvm::StringRef envName) {
  auto textOpt = llvm::sys::Process::GetEnv(envName);
  if (!textOpt)
    return std::nullopt;
  return circt::parseMegabytes(*textOpt);
}

std::optional<uint64_t> circt::parseMegabytes(llvm::StringRef text) {
  text = text.trim();
  if (text.empty())
    return std::nullopt;
  uint64_t mb = 0;
  if (text.getAsInteger(/*Radix=*/10, mb))
    return std::nullopt;
  return mb;
}

static uint64_t megabytesToBytes(uint64_t mb) {
  return mb * 1024ull * 1024ull;
}

static uint64_t getMallocBytes() {
  return static_cast<uint64_t>(llvm::sys::Process::GetMallocUsage());
}

static uint64_t getRSSBytes() {
#if defined(__linux__)
  // /proc/self/statm: size resident shared text lib data dt
  // resident is in pages.
  FILE *f = ::fopen("/proc/self/statm", "r");
  if (!f)
    return 0;
  unsigned long sizePages = 0, residentPages = 0;
  int scanned = ::fscanf(f, "%lu %lu", &sizePages, &residentPages);
  ::fclose(f);
  if (scanned != 2)
    return 0;
  long pageSize = ::sysconf(_SC_PAGESIZE);
  if (pageSize <= 0)
    return 0;
  return static_cast<uint64_t>(residentPages) * static_cast<uint64_t>(pageSize);
#else
  // Fallback: best-effort approximation.
  return getMallocBytes();
#endif
}

static void setAddressSpaceLimitBytes(uint64_t bytes) {
#if defined(__unix__) || defined(__APPLE__)
  if (bytes == 0)
    return;
  struct rlimit rlim;
  if (::getrlimit(RLIMIT_AS, &rlim) != 0)
    return;
  // Keep the hard limit unchanged unless it is lower than our requested limit.
  rlim_t requested = static_cast<rlim_t>(bytes);
  if (rlim.rlim_max != RLIM_INFINITY && requested > rlim.rlim_max)
    requested = rlim.rlim_max;
  rlim.rlim_cur = requested;
  (void)::setrlimit(RLIMIT_AS, &rlim);
#else
  (void)bytes;
#endif
}

namespace {
struct GuardState {
  std::atomic<bool> started{false};
  std::atomic<uint64_t> maxRSSBytes{0};
  std::atomic<uint64_t> maxMallocBytes{0};
  std::atomic<unsigned> intervalMs{250};
};
} // namespace

static GuardState &getGuardState() {
  static GuardState state;
  return state;
}

static void watchdogThreadMain() {
  GuardState &state = getGuardState();
  while (true) {
    const unsigned ms = state.intervalMs.load(std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(ms ? ms : 250));

    const uint64_t rssLimit = state.maxRSSBytes.load(std::memory_order_relaxed);
    const uint64_t mallocLimit =
        state.maxMallocBytes.load(std::memory_order_relaxed);

    if (rssLimit) {
      const uint64_t rss = getRSSBytes();
      if (rss && rss > rssLimit) {
        llvm::errs() << "error: resource guard triggered: RSS "
                     << (rss / (1024ull * 1024ull)) << " MB exceeded limit "
                     << (rssLimit / (1024ull * 1024ull))
                     << " MB; aborting.\n";
        std::_Exit(1);
      }
    }

    if (mallocLimit) {
      const uint64_t used = getMallocBytes();
      if (used && used > mallocLimit) {
        llvm::errs() << "error: resource guard triggered: malloc usage "
                     << (used / (1024ull * 1024ull)) << " MB exceeded limit "
                     << (mallocLimit / (1024ull * 1024ull))
                     << " MB; aborting.\n";
        std::_Exit(1);
      }
    }
  }
}

void circt::installResourceGuard() {
  GuardState &state = getGuardState();

  auto readMB = [](unsigned optValue, unsigned occurrences,
                   llvm::StringRef env) -> uint64_t {
    if (occurrences > 0)
      return optValue;
    if (auto envMB = parseEnvMegabytes(env))
      return *envMB;
    return optValue;
  };

  const uint64_t maxRSSMB = readMB(optMaxRSSMB, optMaxRSSMB.getNumOccurrences(),
                                  "CIRCT_MAX_RSS_MB");
  const uint64_t maxMallocMB =
      readMB(optMaxMallocMB, optMaxMallocMB.getNumOccurrences(),
             "CIRCT_MAX_MALLOC_MB");
  const uint64_t maxVmemMB = readMB(optMaxVMemMB, optMaxVMemMB.getNumOccurrences(),
                                   "CIRCT_MAX_VMEM_MB");

  if (maxVmemMB)
    setAddressSpaceLimitBytes(megabytesToBytes(maxVmemMB));

  if (!maxRSSMB && !maxMallocMB)
    return;

  state.maxRSSBytes.store(megabytesToBytes(maxRSSMB),
                          std::memory_order_relaxed);
  state.maxMallocBytes.store(megabytesToBytes(maxMallocMB),
                             std::memory_order_relaxed);
  state.intervalMs.store(optGuardIntervalMs, std::memory_order_relaxed);

  bool expected = false;
  if (!state.started.compare_exchange_strong(expected, true))
    return;

  std::thread(watchdogThreadMain).detach();
}

llvm::cl::OptionCategory &circt::getResourceGuardCategory() {
  return resourceGuardCategory;
}
