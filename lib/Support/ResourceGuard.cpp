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
#include <cstring>
#include <cstdlib>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace circt;
namespace cl = llvm::cl;

static cl::OptionCategory resourceGuardCategory("Resource Guard Options");

static cl::opt<bool> optResourceGuard(
    "resource-guard",
    cl::desc("Enable default resource guard limits unless explicitly "
             "overridden (disable with --no-resource-guard)"),
    cl::init(true), cl::cat(resourceGuardCategory));

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

static std::optional<uint64_t> getSystemMemoryMegabytes() {
#if defined(__linux__)
  FILE *f = ::fopen("/proc/meminfo", "r");
  if (!f)
    return std::nullopt;
  char key[64];
  unsigned long valueKB = 0;
  char unit[32];
  while (::fscanf(f, "%63s %lu %31s", key, &valueKB, unit) == 3) {
    if (std::strcmp(key, "MemTotal:") == 0) {
      ::fclose(f);
      // meminfo reports kB.
      return static_cast<uint64_t>(valueKB / 1024ul);
    }
  }
  ::fclose(f);
  return std::nullopt;
#else
  return std::nullopt;
#endif
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

  auto envMaxRSSMB = parseEnvMegabytes("CIRCT_MAX_RSS_MB");
  auto envMaxMallocMB = parseEnvMegabytes("CIRCT_MAX_MALLOC_MB");
  auto envMaxVMemMB = parseEnvMegabytes("CIRCT_MAX_VMEM_MB");

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

  uint64_t effectiveMaxRSSMB = maxRSSMB;
  uint64_t effectiveMaxMallocMB = maxMallocMB;
  uint64_t effectiveMaxVMemMB = maxVmemMB;

  // If no explicit limits were provided, apply conservative defaults when the
  // guard is enabled. The goal is to prevent tools from consuming tens of GB of
  // RAM and effectively hanging a machine due to swapping/OOM thrashing.
  if (optResourceGuard && optMaxRSSMB.getNumOccurrences() == 0 &&
      optMaxMallocMB.getNumOccurrences() == 0 &&
      optMaxVMemMB.getNumOccurrences() == 0 && !envMaxRSSMB && !envMaxMallocMB &&
      !envMaxVMemMB) {
    // Default to 80% of system memory, but cap at 20GB.
    if (auto memTotalMB = getSystemMemoryMegabytes()) {
      uint64_t byPercent = (*memTotalMB * 80ull) / 100ull;
      effectiveMaxRSSMB = std::min<uint64_t>(20000ull, byPercent);
    } else {
      effectiveMaxRSSMB = 20000ull;
    }
  }

  if (effectiveMaxVMemMB)
    setAddressSpaceLimitBytes(megabytesToBytes(effectiveMaxVMemMB));

  if (!effectiveMaxRSSMB && !effectiveMaxMallocMB)
    return;

  state.maxRSSBytes.store(megabytesToBytes(effectiveMaxRSSMB),
                          std::memory_order_relaxed);
  state.maxMallocBytes.store(megabytesToBytes(effectiveMaxMallocMB),
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
