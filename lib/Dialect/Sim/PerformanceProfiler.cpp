//===- PerformanceProfiler.cpp - Simulation performance profiling ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the performance profiling infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/PerformanceProfiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#define DEBUG_TYPE "sim-profiler"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// ScopedProfiler Implementation
//===----------------------------------------------------------------------===//

ScopedProfiler::ScopedProfiler(PerformanceProfiler *profiler,
                               ProfileCategory category,
                               const std::string &name, uint64_t context)
    : profiler(profiler), category(category), name(name), context(context),
      stopped(false) {
  if (profiler && profiler->isEnabled()) {
    timer.start();
  }
}

ScopedProfiler::~ScopedProfiler() {
  if (!stopped) {
    stop();
  }
}

void ScopedProfiler::stop() {
  if (!stopped && profiler && profiler->isEnabled()) {
    auto duration = timer.stop();
    profiler->endOperation(category, duration, name, context);
    stopped = true;
  }
}

//===----------------------------------------------------------------------===//
// PerformanceProfiler Implementation
//===----------------------------------------------------------------------===//

PerformanceProfiler::PerformanceProfiler(Config config)
    : config(config), enabled(config.enabled) {
  if (config.collectHistograms) {
    for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
         ++i) {
      histograms[i].resize(config.histogramBuckets, 0);
    }
  }

  samples.reserve(config.maxSamples);
}

PerformanceProfiler::~PerformanceProfiler() = default;

void PerformanceProfiler::reset() {
  // Reset category stats
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    categoryStats[i].reset();
  }

  // Clear samples
  {
    std::lock_guard<std::mutex> lock(samplesMutex);
    samples.clear();
    sampleCounter.store(0);
  }

  // Clear process profiles
  {
    std::lock_guard<std::mutex> lock(processProfilesMutex);
    for (auto &kv : processProfiles) {
      kv.second->totalTimeNs.store(0);
      kv.second->executionCount.store(0);
      kv.second->sensitivityTimeNs.store(0);
      kv.second->triggerCount.store(0);
    }
  }

  // Clear signal profiles
  {
    std::lock_guard<std::mutex> lock(signalProfilesMutex);
    for (auto &kv : signalProfiles) {
      kv.second->updateCount.store(0);
      kv.second->triggerCount.store(0);
      kv.second->updateTimeNs.store(0);
    }
  }

  // Reset histograms
  if (config.collectHistograms) {
    std::lock_guard<std::mutex> lock(histogramMutex);
    for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
         ++i) {
      std::fill(histograms[i].begin(), histograms[i].end(), 0);
    }
  }
}

void PerformanceProfiler::startSession(const std::string &name) {
  sessionName = name;
  sessionTimer.start();
  reset();
  LLVM_DEBUG(llvm::dbgs() << "Started profiling session: " << name << "\n");
}

void PerformanceProfiler::endSession() {
  enabled = false;
  LLVM_DEBUG(llvm::dbgs() << "Ended profiling session: " << sessionName
                          << "\n");
}

void PerformanceProfiler::beginOperation(ProfileCategory category,
                                         const std::string &name,
                                         uint64_t context) {
  // The actual timing is handled by ScopedProfiler
}

void PerformanceProfiler::endOperation(ProfileCategory category,
                                       HighResolutionTimer::Duration duration,
                                       const std::string &name,
                                       uint64_t context) {
  if (!enabled)
    return;

  uint64_t durationNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

  // Update category stats
  categoryStats[static_cast<size_t>(category)].addSample(durationNs);

  // Add to histogram
  if (config.collectHistograms) {
    addToHistogram(category, durationNs);
  }

  // Record detailed sample if sampling
  size_t counter = sampleCounter.fetch_add(1);
  if (counter % config.sampleRate == 0) {
    std::lock_guard<std::mutex> lock(samplesMutex);
    if (samples.size() < config.maxSamples) {
      ProfileSample sample;
      sample.category = category;
      sample.name = name;
      sample.duration = duration;
      sample.context = context;
      sample.timestamp = HighResolutionTimer::now();
      samples.push_back(std::move(sample));
    }
  }
}

void PerformanceProfiler::recordSample(const ProfileSample &sample) {
  if (!enabled)
    return;

  uint64_t durationNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(sample.duration)
          .count();

  categoryStats[static_cast<size_t>(sample.category)].addSample(durationNs);

  std::lock_guard<std::mutex> lock(samplesMutex);
  if (samples.size() < config.maxSamples) {
    samples.push_back(sample);
  }
}

void PerformanceProfiler::registerProcess(ProcessId id,
                                          const std::string &name) {
  if (!config.profileProcesses)
    return;

  std::lock_guard<std::mutex> lock(processProfilesMutex);
  auto profile = std::make_unique<ProcessProfile>();
  profile->processId = id;
  profile->name = name;
  processProfiles[id] = std::move(profile);
}

void PerformanceProfiler::recordProcessExecution(ProcessId id,
                                                 uint64_t durationNs) {
  if (!enabled || !config.profileProcesses)
    return;

  std::lock_guard<std::mutex> lock(processProfilesMutex);
  auto it = processProfiles.find(id);
  if (it != processProfiles.end()) {
    it->second->totalTimeNs.fetch_add(durationNs);
    it->second->executionCount.fetch_add(1);
  }
}

void PerformanceProfiler::recordProcessTrigger(ProcessId id) {
  if (!enabled || !config.profileProcesses)
    return;

  std::lock_guard<std::mutex> lock(processProfilesMutex);
  auto it = processProfiles.find(id);
  if (it != processProfiles.end()) {
    it->second->triggerCount.fetch_add(1);
  }
}

ProcessProfile *PerformanceProfiler::getProcessProfile(ProcessId id) {
  std::lock_guard<std::mutex> lock(processProfilesMutex);
  auto it = processProfiles.find(id);
  return it != processProfiles.end() ? it->second.get() : nullptr;
}

void PerformanceProfiler::registerSignal(SignalId id, const std::string &name) {
  if (!config.profileSignals)
    return;

  std::lock_guard<std::mutex> lock(signalProfilesMutex);
  auto profile = std::make_unique<SignalProfile>();
  profile->signalId = id;
  profile->name = name;
  signalProfiles[id] = std::move(profile);
}

void PerformanceProfiler::recordSignalUpdate(SignalId id, uint64_t durationNs) {
  if (!enabled || !config.profileSignals)
    return;

  std::lock_guard<std::mutex> lock(signalProfilesMutex);
  auto it = signalProfiles.find(id);
  if (it != signalProfiles.end()) {
    it->second->updateCount.fetch_add(1);
    it->second->updateTimeNs.fetch_add(durationNs);
  }
}

void PerformanceProfiler::recordSignalTrigger(SignalId id) {
  if (!enabled || !config.profileSignals)
    return;

  std::lock_guard<std::mutex> lock(signalProfilesMutex);
  auto it = signalProfiles.find(id);
  if (it != signalProfiles.end()) {
    it->second->triggerCount.fetch_add(1);
  }
}

SignalProfile *PerformanceProfiler::getSignalProfile(SignalId id) {
  std::lock_guard<std::mutex> lock(signalProfilesMutex);
  auto it = signalProfiles.find(id);
  return it != signalProfiles.end() ? it->second.get() : nullptr;
}

std::vector<ProcessProfile *>
PerformanceProfiler::getHottestProcesses(size_t n) {
  std::vector<ProcessProfile *> result;

  std::lock_guard<std::mutex> lock(processProfilesMutex);
  for (auto &kv : processProfiles) {
    result.push_back(kv.second.get());
  }

  // Sort by total time (descending)
  std::sort(result.begin(), result.end(),
            [](const ProcessProfile *a, const ProcessProfile *b) {
              return a->totalTimeNs.load() > b->totalTimeNs.load();
            });

  if (result.size() > n) {
    result.resize(n);
  }

  return result;
}

std::vector<SignalProfile *>
PerformanceProfiler::getMostActiveSignals(size_t n) {
  std::vector<SignalProfile *> result;

  std::lock_guard<std::mutex> lock(signalProfilesMutex);
  for (auto &kv : signalProfiles) {
    result.push_back(kv.second.get());
  }

  // Sort by update count (descending)
  std::sort(result.begin(), result.end(),
            [](const SignalProfile *a, const SignalProfile *b) {
              return a->updateCount.load() > b->updateCount.load();
            });

  if (result.size() > n) {
    result.resize(n);
  }

  return result;
}

const std::vector<size_t> &
PerformanceProfiler::getHistogram(ProfileCategory category) const {
  return histograms[static_cast<size_t>(category)];
}

std::vector<double> PerformanceProfiler::getHistogramBuckets() const {
  std::vector<double> buckets(config.histogramBuckets);
  // Log-scale buckets from 1ns to 1s
  double minLog = 0;  // log10(1ns)
  double maxLog = 9;  // log10(1s)
  double step = (maxLog - minLog) / config.histogramBuckets;

  for (size_t i = 0; i < config.histogramBuckets; ++i) {
    buckets[i] = std::pow(10, minLog + i * step);
  }

  return buckets;
}

void PerformanceProfiler::addToHistogram(ProfileCategory category,
                                         uint64_t durationNs) {
  if (!config.collectHistograms || durationNs == 0)
    return;

  // Log-scale bucket index
  double logDuration = std::log10(static_cast<double>(durationNs));
  double minLog = 0;
  double maxLog = 9;
  double step = (maxLog - minLog) / config.histogramBuckets;

  size_t bucket = static_cast<size_t>((logDuration - minLog) / step);
  bucket = std::min(bucket, config.histogramBuckets - 1);

  std::lock_guard<std::mutex> lock(histogramMutex);
  histograms[static_cast<size_t>(category)][bucket]++;
}

HighResolutionTimer::Duration PerformanceProfiler::getTotalTime() const {
  return sessionTimer.elapsed();
}

void PerformanceProfiler::printSummary(llvm::raw_ostream &os) const {
  os << "=== Performance Profiling Summary ===\n";
  os << "Session: " << sessionName << "\n";
  os << "Total time: "
     << HighResolutionTimer::toMilliseconds(sessionTimer.elapsed())
     << " ms\n\n";

  // Calculate total time across all categories
  uint64_t totalCategoryTime = 0;
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    totalCategoryTime += categoryStats[i].totalTimeNs.load();
  }

  os << "Category breakdown:\n";
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    auto cat = static_cast<ProfileCategory>(i);
    const auto &stats = categoryStats[i];
    uint64_t total = stats.totalTimeNs.load();
    uint64_t count = stats.sampleCount.load();

    if (count == 0)
      continue;

    double percentage =
        totalCategoryTime > 0 ? (total * 100.0 / totalCategoryTime) : 0;

    os << llvm::format("  %-20s: %8.2f ms (%5.1f%%) | %8lu samples | avg "
                       "%8.2f us\n",
                       getProfileCategoryName(cat), total / 1e6, percentage,
                       count, stats.getAverageNs() / 1000.0);
  }

  os << "\nSamples collected: " << samples.size() << "\n";
}

void PerformanceProfiler::printCategoryBreakdown(llvm::raw_ostream &os) const {
  os << "=== Category Breakdown ===\n\n";

  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    auto cat = static_cast<ProfileCategory>(i);
    const auto &stats = categoryStats[i];

    if (stats.sampleCount.load() == 0)
      continue;

    os << getProfileCategoryName(cat) << ":\n";
    os << "  Total time: " << stats.totalTimeNs.load() / 1e6 << " ms\n";
    os << "  Sample count: " << stats.sampleCount.load() << "\n";
    os << "  Average: " << stats.getAverageNs() / 1000.0 << " us\n";

    uint64_t minNs = stats.minTimeNs.load();
    uint64_t maxNs = stats.maxTimeNs.load();
    if (minNs != UINT64_MAX) {
      os << "  Min: " << minNs / 1000.0 << " us\n";
    }
    os << "  Max: " << maxNs / 1000.0 << " us\n";
    os << "\n";
  }
}

void PerformanceProfiler::printProcessHotspots(llvm::raw_ostream &os,
                                               size_t n) const {
  os << "=== Process Hotspots (Top " << n << ") ===\n\n";

  std::vector<ProcessProfile *> profiles;
  {
    std::lock_guard<std::mutex> lock(
        const_cast<std::mutex &>(processProfilesMutex));
    for (auto &kv : processProfiles) {
      profiles.push_back(kv.second.get());
    }
  }

  std::sort(profiles.begin(), profiles.end(),
            [](const ProcessProfile *a, const ProcessProfile *b) {
              return a->totalTimeNs.load() > b->totalTimeNs.load();
            });

  if (profiles.size() > n) {
    profiles.resize(n);
  }

  uint64_t totalProcessTime = 0;
  for (const auto *p : profiles) {
    totalProcessTime += p->totalTimeNs.load();
  }

  for (size_t i = 0; i < profiles.size(); ++i) {
    const auto *p = profiles[i];
    uint64_t total = p->totalTimeNs.load();
    uint64_t count = p->executionCount.load();
    double percentage =
        totalProcessTime > 0 ? (total * 100.0 / totalProcessTime) : 0;

    os << llvm::format("%3zu. %-30s: %8.2f ms (%5.1f%%) | %8lu execs | avg "
                       "%8.2f us\n",
                       i + 1, p->name.c_str(), total / 1e6, percentage, count,
                       p->getAverageTimeNs() / 1000.0);
  }
}

void PerformanceProfiler::printSignalActivity(llvm::raw_ostream &os,
                                              size_t n) const {
  os << "=== Signal Activity (Top " << n << ") ===\n\n";

  std::vector<SignalProfile *> profiles;
  {
    std::lock_guard<std::mutex> lock(
        const_cast<std::mutex &>(signalProfilesMutex));
    for (auto &kv : signalProfiles) {
      profiles.push_back(kv.second.get());
    }
  }

  std::sort(profiles.begin(), profiles.end(),
            [](const SignalProfile *a, const SignalProfile *b) {
              return a->updateCount.load() > b->updateCount.load();
            });

  if (profiles.size() > n) {
    profiles.resize(n);
  }

  for (size_t i = 0; i < profiles.size(); ++i) {
    const auto *s = profiles[i];
    os << llvm::format("%3zu. %-30s: %8lu updates | %8lu triggers\n", i + 1,
                       s->name.c_str(), s->updateCount.load(),
                       s->triggerCount.load());
  }
}

void PerformanceProfiler::printHistogram(llvm::raw_ostream &os,
                                         ProfileCategory category) const {
  if (!config.collectHistograms)
    return;

  os << "=== Histogram for " << getProfileCategoryName(category) << " ===\n\n";

  const auto &hist = histograms[static_cast<size_t>(category)];
  auto buckets = getHistogramBuckets();

  // Find max for scaling
  size_t maxCount = *std::max_element(hist.begin(), hist.end());
  if (maxCount == 0)
    return;

  const size_t barWidth = 50;

  for (size_t i = 0; i < hist.size(); ++i) {
    double bucketNs = buckets[i];
    std::string unit = "ns";
    if (bucketNs >= 1e6) {
      bucketNs /= 1e6;
      unit = "ms";
    } else if (bucketNs >= 1e3) {
      bucketNs /= 1e3;
      unit = "us";
    }

    size_t barLen = (hist[i] * barWidth) / maxCount;
    std::string bar(barLen, '#');

    os << llvm::format("%8.1f %s: %8lu |%s\n", bucketNs, unit.c_str(), hist[i],
                       bar.c_str());
  }
}

void PerformanceProfiler::exportJSON(llvm::raw_ostream &os) const {
  os << "{\n";
  os << "  \"session\": \"" << sessionName << "\",\n";
  os << "  \"totalTimeMs\": "
     << HighResolutionTimer::toMilliseconds(sessionTimer.elapsed()) << ",\n";

  // Categories
  os << "  \"categories\": {\n";
  bool first = true;
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    const auto &stats = categoryStats[i];
    if (stats.sampleCount.load() == 0)
      continue;

    if (!first)
      os << ",\n";
    first = false;

    os << "    \"" << getProfileCategoryName(static_cast<ProfileCategory>(i))
       << "\": {\n";
    os << "      \"totalTimeNs\": " << stats.totalTimeNs.load() << ",\n";
    os << "      \"sampleCount\": " << stats.sampleCount.load() << ",\n";
    os << "      \"minTimeNs\": "
       << (stats.minTimeNs.load() == UINT64_MAX ? 0 : stats.minTimeNs.load())
       << ",\n";
    os << "      \"maxTimeNs\": " << stats.maxTimeNs.load() << "\n";
    os << "    }";
  }
  os << "\n  },\n";

  // Samples (first 100)
  os << "  \"samples\": [\n";
  size_t sampleLimit = std::min(samples.size(), size_t(100));
  for (size_t i = 0; i < sampleLimit; ++i) {
    const auto &s = samples[i];
    if (i > 0)
      os << ",\n";
    os << "    {\"category\": \"" << getProfileCategoryName(s.category)
       << "\", \"name\": \"" << s.name
       << "\", \"durationNs\": " << s.duration.count() << "}";
  }
  os << "\n  ]\n";

  os << "}\n";
}

void PerformanceProfiler::exportCSV(llvm::raw_ostream &os) const {
  // Header
  os << "category,name,duration_ns,context\n";

  // Samples
  for (const auto &s : samples) {
    os << getProfileCategoryName(s.category) << "," << s.name << ","
       << s.duration.count() << "," << s.context << "\n";
  }
}

//===----------------------------------------------------------------------===//
// BottleneckAnalyzer Implementation
//===----------------------------------------------------------------------===//

std::vector<BottleneckAnalyzer::Bottleneck> BottleneckAnalyzer::analyze() {
  std::vector<Bottleneck> bottlenecks;

  // Calculate total time
  uint64_t totalTime = 0;
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    totalTime += profiler.getCategoryStats(static_cast<ProfileCategory>(i))
                     .totalTimeNs.load();
  }

  if (totalTime == 0)
    return bottlenecks;

  // Check for scheduling overhead
  const auto &schedStats =
      profiler.getCategoryStats(ProfileCategory::EventScheduling);
  double schedPercent =
      schedStats.totalTimeNs.load() * 100.0 / totalTime;
  if (schedPercent > 20) {
    Bottleneck b;
    b.type = Bottleneck::Type::HighEventSchedulingOverhead;
    b.description = "Event scheduling overhead is " +
                    std::to_string(static_cast<int>(schedPercent)) +
                    "% of total time";
    b.severity = std::min(1.0, schedPercent / 50);
    b.suggestion = "Consider batching events or reducing event granularity";
    bottlenecks.push_back(std::move(b));
  }

  // Check for slow processes
  auto hotProcesses = const_cast<PerformanceProfiler &>(profiler)
                          .getHottestProcesses(5);
  if (!hotProcesses.empty()) {
    const auto *hottest = hotProcesses[0];
    double avgUs = hottest->getAverageTimeNs() / 1000.0;
    if (avgUs > 100) { // > 100us average
      Bottleneck b;
      b.type = Bottleneck::Type::SlowProcess;
      b.description = "Process '" + hottest->name +
                      "' has high average execution time: " +
                      std::to_string(static_cast<int>(avgUs)) + " us";
      b.severity = std::min(1.0, avgUs / 1000);
      b.suggestion = "Profile and optimize the process implementation";
      bottlenecks.push_back(std::move(b));
    }
  }

  // Check for synchronization issues
  const auto &syncStats =
      profiler.getCategoryStats(ProfileCategory::Synchronization);
  double syncPercent = syncStats.totalTimeNs.load() * 100.0 / totalTime;
  if (syncPercent > 15) {
    Bottleneck b;
    b.type = Bottleneck::Type::SynchronizationContention;
    b.description = "Synchronization overhead is " +
                    std::to_string(static_cast<int>(syncPercent)) +
                    "% of total time";
    b.severity = std::min(1.0, syncPercent / 30);
    b.suggestion =
        "Consider reducing cross-partition signals or using lock-free "
        "algorithms";
    bottlenecks.push_back(std::move(b));
  }

  // Sort by severity
  std::sort(bottlenecks.begin(), bottlenecks.end(),
            [](const Bottleneck &a, const Bottleneck &b) {
              return a.severity > b.severity;
            });

  return bottlenecks;
}

ProfileCategory BottleneckAnalyzer::getDominantCategory() const {
  uint64_t maxTime = 0;
  ProfileCategory dominant = ProfileCategory::Custom;

  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    uint64_t time = profiler.getCategoryStats(static_cast<ProfileCategory>(i))
                        .totalTimeNs.load();
    if (time > maxTime) {
      maxTime = time;
      dominant = static_cast<ProfileCategory>(i);
    }
  }

  return dominant;
}

bool BottleneckAnalyzer::hasSchedulingOverhead() const {
  uint64_t totalTime = 0;
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    totalTime += profiler.getCategoryStats(static_cast<ProfileCategory>(i))
                     .totalTimeNs.load();
  }

  if (totalTime == 0)
    return false;

  uint64_t schedTime =
      profiler.getCategoryStats(ProfileCategory::EventScheduling)
          .totalTimeNs.load();
  return (schedTime * 100.0 / totalTime) > 15;
}

bool BottleneckAnalyzer::hasSlowProcesses() const {
  auto hotProcesses =
      const_cast<PerformanceProfiler &>(profiler).getHottestProcesses(1);
  if (hotProcesses.empty())
    return false;

  return hotProcesses[0]->getAverageTimeNs() > 100000; // > 100us
}

bool BottleneckAnalyzer::hasSynchronizationIssues() const {
  uint64_t totalTime = 0;
  for (size_t i = 0; i < static_cast<size_t>(ProfileCategory::NumCategories);
       ++i) {
    totalTime += profiler.getCategoryStats(static_cast<ProfileCategory>(i))
                     .totalTimeNs.load();
  }

  if (totalTime == 0)
    return false;

  uint64_t syncTime = profiler.getCategoryStats(ProfileCategory::Synchronization)
                          .totalTimeNs.load();
  return (syncTime * 100.0 / totalTime) > 10;
}

void BottleneckAnalyzer::printReport(llvm::raw_ostream &os) {
  os << "=== Bottleneck Analysis Report ===\n\n";

  auto bottlenecks = analyze();

  if (bottlenecks.empty()) {
    os << "No significant bottlenecks detected.\n";
    return;
  }

  os << "Dominant category: " << getProfileCategoryName(getDominantCategory())
     << "\n\n";

  os << "Identified bottlenecks (sorted by severity):\n\n";

  for (size_t i = 0; i < bottlenecks.size(); ++i) {
    const auto &b = bottlenecks[i];
    os << i + 1 << ". [Severity: " << llvm::format("%.2f", b.severity)
       << "]\n";
    os << "   Type: ";
    switch (b.type) {
    case Bottleneck::Type::HighEventSchedulingOverhead:
      os << "High Event Scheduling Overhead";
      break;
    case Bottleneck::Type::SlowProcess:
      os << "Slow Process";
      break;
    case Bottleneck::Type::FrequentSignalUpdate:
      os << "Frequent Signal Update";
      break;
    case Bottleneck::Type::SynchronizationContention:
      os << "Synchronization Contention";
      break;
    case Bottleneck::Type::MemoryBound:
      os << "Memory Bound";
      break;
    case Bottleneck::Type::Other:
      os << "Other";
      break;
    }
    os << "\n";
    os << "   Description: " << b.description << "\n";
    os << "   Suggestion: " << b.suggestion << "\n\n";
  }
}

//===----------------------------------------------------------------------===//
// ProfilerIntegration Implementation
//===----------------------------------------------------------------------===//

ProfilerIntegration::ProfilerIntegration(PerformanceProfiler &profiler,
                                         ProcessScheduler &scheduler)
    : profiler(profiler), scheduler(scheduler), installed(false) {}

void ProfilerIntegration::install() {
  if (installed)
    return;

  // Register all processes
  const auto &processes = scheduler.getProcesses();
  for (const auto &kv : processes) {
    profiler.registerProcess(kv.first, kv.second->getName());
  }

  installed = true;
  LLVM_DEBUG(llvm::dbgs() << "Profiler integration installed\n");
}

void ProfilerIntegration::uninstall() {
  if (!installed)
    return;

  installed = false;
  LLVM_DEBUG(llvm::dbgs() << "Profiler integration uninstalled\n");
}
