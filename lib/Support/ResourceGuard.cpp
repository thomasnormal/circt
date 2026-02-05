//===- ResourceGuard.cpp - Memory usage safeguards --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ResourceGuard.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
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
             "(0 = disabled; if left unspecified and --resource-guard is "
             "enabled, a conservative default is applied; env: "
             "CIRCT_MAX_RSS_MB)"),
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

static cl::opt<unsigned> optMaxWallMs(
    "max-wall-ms",
    cl::desc("Abort if wall-clock time exceeds this many milliseconds "
             "(0 = disabled; env: CIRCT_MAX_WALL_MS)"),
    cl::init(0), cl::cat(resourceGuardCategory));

static cl::opt<unsigned> optGuardIntervalMs(
    "resource-guard-interval-ms",
    cl::desc("Polling interval (ms) for the resource guard watchdog thread "
             "(env: CIRCT_RESOURCE_GUARD_INTERVAL_MS)"),
    cl::init(250), cl::cat(resourceGuardCategory));

static std::optional<uint64_t> parseEnvMegabytes(llvm::StringRef envName) {
  auto textOpt = llvm::sys::Process::GetEnv(envName);
  if (!textOpt)
    return std::nullopt;
  return circt::parseMegabytes(*textOpt);
}

static std::optional<uint64_t> parseEnvMilliseconds(llvm::StringRef envName) {
  auto textOpt = llvm::sys::Process::GetEnv(envName);
  if (!textOpt)
    return std::nullopt;
  llvm::StringRef text = llvm::StringRef(*textOpt).trim();
  if (text.empty())
    return std::nullopt;
  uint64_t value = 0;
  if (text.getAsInteger(10, value))
    return std::nullopt;
  return value;
}

static std::optional<uint64_t> getSystemMemoryMegabytes() {
#if defined(__linux__)
  auto readFileToString = [](const char *path) -> std::optional<std::string> {
    FILE *f = ::fopen(path, "r");
    if (!f)
      return std::nullopt;
    char buffer[256];
    size_t n = ::fread(buffer, 1, sizeof(buffer) - 1, f);
    ::fclose(f);
    if (!n)
      return std::nullopt;
    buffer[n] = '\0';
    return std::string(buffer);
  };

  auto parseBytesFromCGroupLimit =
      [&](const std::string &text) -> std::optional<uint64_t> {
    llvm::StringRef s(text);
    s = s.trim();
    if (s.empty() || s.equals_insensitive("max"))
      return std::nullopt;
    uint64_t bytes = 0;
    if (s.getAsInteger(10, bytes))
      return std::nullopt;
    // Some kernels report "unlimited" as a huge number close to UINT64_MAX.
    if (bytes > (1ull << 60))
      return std::nullopt;
    return bytes;
  };

  auto getCGroupMemoryLimitMegabytes = [&]() -> std::optional<uint64_t> {
    // Best-effort detection of a cgroup memory limit. Prefer cgroup v2 when
    // available; fall back to v1.
    FILE *f = ::fopen("/proc/self/cgroup", "r");
    if (!f)
      return std::nullopt;

    char line[512];
    std::optional<std::string> v2Path;
    std::optional<std::string> v1MemPath;
    while (::fgets(line, sizeof(line), f)) {
      llvm::StringRef l(line);
      l = l.trim();
      if (l.starts_with("0::")) {
        v2Path = l.drop_front(3).str();
        break;
      }
      // cgroup v1: ID:controllers:path
      // Find the memory controller entry.
      if (l.contains(":memory:") || l.contains(":memory,")) {
        auto lastColon = l.rfind(':');
        if (lastColon != llvm::StringRef::npos)
          v1MemPath = l.drop_front(lastColon + 1).str();
      }
    }
    ::fclose(f);

    if (v2Path) {
      std::string path = "/sys/fs/cgroup" + *v2Path + "/memory.max";
      if (auto textOpt = readFileToString(path.c_str()))
        if (auto bytesOpt = parseBytesFromCGroupLimit(*textOpt))
          return *bytesOpt / (1024ull * 1024ull);
    }

    if (v1MemPath) {
      std::string path =
          "/sys/fs/cgroup/memory" + *v1MemPath + "/memory.limit_in_bytes";
      if (auto textOpt = readFileToString(path.c_str()))
        if (auto bytesOpt = parseBytesFromCGroupLimit(*textOpt))
          return *bytesOpt / (1024ull * 1024ull);
    }
    return std::nullopt;
  };

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
      uint64_t memTotalMB = static_cast<uint64_t>(valueKB / 1024ul);
      if (auto cgroupLimitMB = getCGroupMemoryLimitMegabytes())
        memTotalMB = std::min<uint64_t>(memTotalMB, *cgroupLimitMB);
      return memTotalMB;
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
  std::atomic<uint64_t> maxWallMs{0};
  std::atomic<uint64_t> startMs{0};
  std::atomic<unsigned> intervalMs{250};
  std::atomic<const char *> phaseLabel{nullptr};
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

    const uint64_t wallLimitMs =
        state.maxWallMs.load(std::memory_order_relaxed);
    if (wallLimitMs) {
      const uint64_t startMs = state.startMs.load(std::memory_order_relaxed);
      const uint64_t nowMs = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
      if (nowMs >= startMs && (nowMs - startMs) > wallLimitMs) {
        const char *phase = state.phaseLabel.load(std::memory_order_relaxed);
        llvm::errs() << "error: resource guard triggered: wall time "
                     << (nowMs - startMs) << " ms exceeded limit "
                     << wallLimitMs << " ms";
        if (phase)
          llvm::errs() << " (phase: " << phase << ")";
        llvm::errs() << "; aborting.\n";
        std::_Exit(1);
      }
    }

    const uint64_t rssLimit = state.maxRSSBytes.load(std::memory_order_relaxed);
    const uint64_t mallocLimit =
        state.maxMallocBytes.load(std::memory_order_relaxed);

    if (rssLimit) {
      const uint64_t rss = getRSSBytes();
      if (rss && rss > rssLimit) {
        const char *phase = state.phaseLabel.load(std::memory_order_relaxed);
        llvm::errs() << "error: resource guard triggered: RSS "
                     << (rss / (1024ull * 1024ull)) << " MB exceeded limit "
                     << (rssLimit / (1024ull * 1024ull))
                     << " MB";
        if (phase)
          llvm::errs() << " (phase: " << phase << ")";
        llvm::errs() << "; aborting.\n";
        std::_Exit(1);
      }
    }

    if (mallocLimit) {
      const uint64_t used = getMallocBytes();
      if (used && used > mallocLimit) {
        const char *phase = state.phaseLabel.load(std::memory_order_relaxed);
        llvm::errs() << "error: resource guard triggered: malloc usage "
                     << (used / (1024ull * 1024ull)) << " MB exceeded limit "
                     << (mallocLimit / (1024ull * 1024ull))
                     << " MB";
        if (phase)
          llvm::errs() << " (phase: " << phase << ")";
        llvm::errs() << "; aborting.\n";
        std::_Exit(1);
      }
    }
  }
}

void circt::setResourceGuardPhase(llvm::StringRef phase) {
  GuardState &state = getGuardState();

  // Copy the string into stable storage; the watchdog thread may outlive the
  // caller's storage. This function may be called from pass instrumentation,
  // so we intern strings to avoid unbounded growth from repeated labels.
  static llvm::BumpPtrAllocator allocator;
  static llvm::StringMap<const char *> interned;
  static std::mutex internMutex;

  const char *label = nullptr;
  if (!phase.empty()) {
    std::lock_guard<std::mutex> lock(internMutex);
    auto it = interned.find(phase);
    if (it != interned.end()) {
      label = it->second;
    } else {
      char *storage =
          static_cast<char *>(allocator.Allocate(phase.size() + 1, 1));
      std::memcpy(storage, phase.data(), phase.size());
      storage[phase.size()] = '\0';
      interned[phase] = storage;
      label = storage;
    }
  }

  state.phaseLabel.store(label, std::memory_order_relaxed);
}

void circt::installResourceGuard() {
  GuardState &state = getGuardState();

  auto envMaxRSSMB = parseEnvMegabytes("CIRCT_MAX_RSS_MB");
  auto envMaxMallocMB = parseEnvMegabytes("CIRCT_MAX_MALLOC_MB");
  auto envMaxVMemMB = parseEnvMegabytes("CIRCT_MAX_VMEM_MB");
  auto envMaxWallMs = parseEnvMilliseconds("CIRCT_MAX_WALL_MS");
  auto envIntervalMs =
      parseEnvMilliseconds("CIRCT_RESOURCE_GUARD_INTERVAL_MS");

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
  const uint64_t maxWallMs =
      (optMaxWallMs.getNumOccurrences() > 0) ? optMaxWallMs
      : (envMaxWallMs ? *envMaxWallMs : optMaxWallMs);

  uint64_t effectiveMaxRSSMB = maxRSSMB;
  uint64_t effectiveMaxMallocMB = maxMallocMB;
  uint64_t effectiveMaxVMemMB = maxVmemMB;
  uint64_t effectiveMaxWallMs = maxWallMs;

  // If no explicit limits were provided, apply conservative defaults when the
  // guard is enabled. The goal is to prevent tools from consuming tens of GB of
  // RAM and effectively hanging a machine due to swapping/OOM thrashing.
  if (optResourceGuard && optMaxRSSMB.getNumOccurrences() == 0 &&
      optMaxMallocMB.getNumOccurrences() == 0 &&
      optMaxVMemMB.getNumOccurrences() == 0 && !envMaxRSSMB && !envMaxMallocMB &&
      !envMaxVMemMB && optMaxWallMs.getNumOccurrences() == 0 && !envMaxWallMs) {
    // Default to a conservative fraction of system memory, but cap at 12GB.
    // This is intentionally sized to prevent runaway memory growth on typical
    // developer workstations while remaining easy to override for large
    // one-off jobs.
    if (auto memTotalMB = getSystemMemoryMegabytes()) {
      uint64_t byPercent = 0;
      if (*memTotalMB <= 4096ull)
        byPercent = (*memTotalMB * 80ull) / 100ull;
      else if (*memTotalMB <= 16384ull)
        byPercent = (*memTotalMB * 60ull) / 100ull;
      else
        byPercent = (*memTotalMB * 40ull) / 100ull;
      effectiveMaxRSSMB = std::min<uint64_t>(12288ull, byPercent);
    } else {
      effectiveMaxRSSMB = 8192ull;
    }
  }

  if (effectiveMaxVMemMB)
    setAddressSpaceLimitBytes(megabytesToBytes(effectiveMaxVMemMB));

  if (!effectiveMaxRSSMB && !effectiveMaxMallocMB && !effectiveMaxWallMs)
    return;

  state.maxRSSBytes.store(megabytesToBytes(effectiveMaxRSSMB),
                          std::memory_order_relaxed);
  state.maxMallocBytes.store(megabytesToBytes(effectiveMaxMallocMB),
                             std::memory_order_relaxed);
  state.maxWallMs.store(effectiveMaxWallMs, std::memory_order_relaxed);
  unsigned effectiveIntervalMs = optGuardIntervalMs;
  if (optGuardIntervalMs.getNumOccurrences() == 0 && envIntervalMs)
    effectiveIntervalMs = static_cast<unsigned>(
        std::min<uint64_t>(*envIntervalMs, std::numeric_limits<unsigned>::max()));
  state.intervalMs.store(effectiveIntervalMs, std::memory_order_relaxed);

  bool expected = false;
  if (!state.started.compare_exchange_strong(expected, true))
    return;

  state.startMs.store(
      static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count()),
      std::memory_order_relaxed);
  std::thread(watchdogThreadMain).detach();
}

llvm::cl::OptionCategory &circt::getResourceGuardCategory() {
  return resourceGuardCategory;
}
