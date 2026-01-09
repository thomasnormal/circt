//===- PerformanceProfiler.h - Simulation performance profiling ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the performance profiling infrastructure for simulation.
// It provides instrumentation for measuring event scheduling overhead,
// identifying bottlenecks, and generating performance reports.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_PERFORMANCEPROFILER_H
#define CIRCT_DIALECT_SIM_PERFORMANCEPROFILER_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// HighResolutionTimer - Precise time measurement
//===----------------------------------------------------------------------===//

/// High-resolution timer for performance measurement.
class HighResolutionTimer {
public:
  using TimePoint = std::chrono::high_resolution_clock::time_point;
  using Duration = std::chrono::nanoseconds;

  /// Start the timer.
  void start() { startTime = std::chrono::high_resolution_clock::now(); }

  /// Stop the timer and return elapsed time.
  Duration stop() {
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<Duration>(endTime - startTime);
  }

  /// Get elapsed time without stopping.
  Duration elapsed() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<Duration>(now - startTime);
  }

  /// Get the start time.
  TimePoint getStartTime() const { return startTime; }

  /// Get current timestamp.
  static TimePoint now() {
    return std::chrono::high_resolution_clock::now();
  }

  /// Convert duration to different units.
  static double toSeconds(Duration d) {
    return std::chrono::duration<double>(d).count();
  }

  static double toMilliseconds(Duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  }

  static double toMicroseconds(Duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  }

private:
  TimePoint startTime;
};

//===----------------------------------------------------------------------===//
// ProfileCategory - Categories of profiled operations
//===----------------------------------------------------------------------===//

/// Categories of operations being profiled.
enum class ProfileCategory : uint8_t {
  EventScheduling = 0,
  EventExecution,
  ProcessExecution,
  SignalUpdate,
  DeltaCycle,
  TimeAdvance,
  SensitivityEval,
  ForkJoin,
  Synchronization,
  MemoryAccess,
  Custom,
  NumCategories
};

/// Get the name of a profile category.
inline const char *getProfileCategoryName(ProfileCategory cat) {
  switch (cat) {
  case ProfileCategory::EventScheduling:
    return "EventScheduling";
  case ProfileCategory::EventExecution:
    return "EventExecution";
  case ProfileCategory::ProcessExecution:
    return "ProcessExecution";
  case ProfileCategory::SignalUpdate:
    return "SignalUpdate";
  case ProfileCategory::DeltaCycle:
    return "DeltaCycle";
  case ProfileCategory::TimeAdvance:
    return "TimeAdvance";
  case ProfileCategory::SensitivityEval:
    return "SensitivityEval";
  case ProfileCategory::ForkJoin:
    return "ForkJoin";
  case ProfileCategory::Synchronization:
    return "Synchronization";
  case ProfileCategory::MemoryAccess:
    return "MemoryAccess";
  case ProfileCategory::Custom:
    return "Custom";
  default:
    return "Unknown";
  }
}

//===----------------------------------------------------------------------===//
// ProfileSample - A single profiling sample
//===----------------------------------------------------------------------===//

/// A single profiling sample.
struct ProfileSample {
  /// Category of the operation.
  ProfileCategory category;

  /// Name/identifier of the specific operation.
  std::string name;

  /// Duration of the operation.
  HighResolutionTimer::Duration duration;

  /// Additional context (e.g., process ID, signal ID).
  uint64_t context;

  /// Timestamp when the sample was taken.
  HighResolutionTimer::TimePoint timestamp;

  ProfileSample() : category(ProfileCategory::Custom), context(0) {}
};

//===----------------------------------------------------------------------===//
// CategoryStatistics - Statistics for a profile category
//===----------------------------------------------------------------------===//

/// Aggregated statistics for a profile category.
struct CategoryStatistics {
  /// Total time spent in this category.
  std::atomic<uint64_t> totalTimeNs{0};

  /// Number of samples.
  std::atomic<uint64_t> sampleCount{0};

  /// Minimum duration.
  std::atomic<uint64_t> minTimeNs{UINT64_MAX};

  /// Maximum duration.
  std::atomic<uint64_t> maxTimeNs{0};

  /// Add a sample.
  void addSample(uint64_t durationNs) {
    totalTimeNs.fetch_add(durationNs);
    sampleCount.fetch_add(1);

    // Update min/max atomically
    uint64_t currentMin = minTimeNs.load();
    while (durationNs < currentMin &&
           !minTimeNs.compare_exchange_weak(currentMin, durationNs)) {
    }

    uint64_t currentMax = maxTimeNs.load();
    while (durationNs > currentMax &&
           !maxTimeNs.compare_exchange_weak(currentMax, durationNs)) {
    }
  }

  /// Get average time per sample.
  double getAverageNs() const {
    uint64_t count = sampleCount.load();
    return count > 0 ? static_cast<double>(totalTimeNs.load()) / count : 0;
  }

  /// Reset statistics.
  void reset() {
    totalTimeNs.store(0);
    sampleCount.store(0);
    minTimeNs.store(UINT64_MAX);
    maxTimeNs.store(0);
  }
};

//===----------------------------------------------------------------------===//
// ProcessProfile - Profile data for a single process
//===----------------------------------------------------------------------===//

/// Profile data for a simulation process.
struct ProcessProfile {
  /// Process ID.
  ProcessId processId;

  /// Process name.
  std::string name;

  /// Total execution time.
  std::atomic<uint64_t> totalTimeNs{0};

  /// Number of executions.
  std::atomic<uint64_t> executionCount{0};

  /// Time spent in sensitivity evaluation.
  std::atomic<uint64_t> sensitivityTimeNs{0};

  /// Number of times triggered.
  std::atomic<uint64_t> triggerCount{0};

  /// Get average execution time.
  double getAverageTimeNs() const {
    uint64_t count = executionCount.load();
    return count > 0 ? static_cast<double>(totalTimeNs.load()) / count : 0;
  }
};

//===----------------------------------------------------------------------===//
// SignalProfile - Profile data for a signal
//===----------------------------------------------------------------------===//

/// Profile data for a simulation signal.
struct SignalProfile {
  /// Signal ID.
  SignalId signalId;

  /// Signal name.
  std::string name;

  /// Number of updates.
  std::atomic<uint64_t> updateCount{0};

  /// Number of times the signal triggered processes.
  std::atomic<uint64_t> triggerCount{0};

  /// Total time spent in signal updates.
  std::atomic<uint64_t> updateTimeNs{0};

  /// Number of processes sensitive to this signal.
  size_t sensitiveProcessCount = 0;
};

//===----------------------------------------------------------------------===//
// ScopedProfiler - RAII profiling helper
//===----------------------------------------------------------------------===//

class PerformanceProfiler;

/// RAII helper for scoped profiling.
class ScopedProfiler {
public:
  ScopedProfiler(PerformanceProfiler *profiler, ProfileCategory category,
                 const std::string &name = "", uint64_t context = 0);
  ~ScopedProfiler();

  // Non-copyable
  ScopedProfiler(const ScopedProfiler &) = delete;
  ScopedProfiler &operator=(const ScopedProfiler &) = delete;

  /// Manually stop profiling early.
  void stop();

private:
  PerformanceProfiler *profiler;
  ProfileCategory category;
  std::string name;
  uint64_t context;
  HighResolutionTimer timer;
  bool stopped;
};

//===----------------------------------------------------------------------===//
// PerformanceProfiler - Main profiling interface
//===----------------------------------------------------------------------===//

/// The main performance profiler for simulation.
class PerformanceProfiler {
public:
  /// Configuration for the profiler.
  struct Config {
    /// Enable profiling.
    bool enabled;

    /// Sample rate (1 = all, N = every Nth operation).
    size_t sampleRate;

    /// Maximum number of samples to keep.
    size_t maxSamples;

    /// Enable per-process profiling.
    bool profileProcesses;

    /// Enable per-signal profiling.
    bool profileSignals;

    /// Enable detailed histogram collection.
    bool collectHistograms;

    /// Number of histogram buckets.
    size_t histogramBuckets;

    Config()
        : enabled(true), sampleRate(1), maxSamples(100000),
          profileProcesses(true), profileSignals(true),
          collectHistograms(false), histogramBuckets(100) {}
  };

  PerformanceProfiler(Config config = Config());
  ~PerformanceProfiler();

  //===------------------------------------------------------------------===//
  // Profiling Control
  //===------------------------------------------------------------------===//

  /// Enable or disable profiling.
  void setEnabled(bool enabled) { this->enabled = enabled; }

  /// Check if profiling is enabled.
  bool isEnabled() const { return enabled && config.enabled; }

  /// Reset all profiling data.
  void reset();

  /// Start a new profiling session.
  void startSession(const std::string &name = "");

  /// End the current profiling session.
  void endSession();

  //===------------------------------------------------------------------===//
  // Manual Profiling API
  //===------------------------------------------------------------------===//

  /// Record the start of a profiled operation.
  void beginOperation(ProfileCategory category, const std::string &name = "",
                      uint64_t context = 0);

  /// Record the end of a profiled operation.
  void endOperation(ProfileCategory category,
                    HighResolutionTimer::Duration duration,
                    const std::string &name = "", uint64_t context = 0);

  /// Record a complete sample.
  void recordSample(const ProfileSample &sample);

  /// Create a scoped profiler.
  ScopedProfiler scope(ProfileCategory category, const std::string &name = "",
                       uint64_t context = 0) {
    return ScopedProfiler(this, category, name, context);
  }

  //===------------------------------------------------------------------===//
  // Process Profiling
  //===------------------------------------------------------------------===//

  /// Register a process for profiling.
  void registerProcess(ProcessId id, const std::string &name);

  /// Record process execution.
  void recordProcessExecution(ProcessId id, uint64_t durationNs);

  /// Record process trigger.
  void recordProcessTrigger(ProcessId id);

  /// Get process profile.
  ProcessProfile *getProcessProfile(ProcessId id);

  //===------------------------------------------------------------------===//
  // Signal Profiling
  //===------------------------------------------------------------------===//

  /// Register a signal for profiling.
  void registerSignal(SignalId id, const std::string &name);

  /// Record signal update.
  void recordSignalUpdate(SignalId id, uint64_t durationNs);

  /// Record signal trigger.
  void recordSignalTrigger(SignalId id);

  /// Get signal profile.
  SignalProfile *getSignalProfile(SignalId id);

  //===------------------------------------------------------------------===//
  // Statistics Access
  //===------------------------------------------------------------------===//

  /// Get statistics for a category.
  const CategoryStatistics &getCategoryStats(ProfileCategory category) const {
    return categoryStats[static_cast<size_t>(category)];
  }

  /// Get all samples (for detailed analysis).
  const std::vector<ProfileSample> &getSamples() const { return samples; }

  /// Get the top N hottest processes.
  std::vector<ProcessProfile *> getHottestProcesses(size_t n);

  /// Get the top N most active signals.
  std::vector<SignalProfile *> getMostActiveSignals(size_t n);

  //===------------------------------------------------------------------===//
  // Histogram Access
  //===------------------------------------------------------------------===//

  /// Get histogram for a category.
  const std::vector<size_t> &getHistogram(ProfileCategory category) const;

  /// Get histogram bucket boundaries.
  std::vector<double> getHistogramBuckets() const;

  //===------------------------------------------------------------------===//
  // Reporting
  //===------------------------------------------------------------------===//

  /// Print a summary report.
  void printSummary(llvm::raw_ostream &os) const;

  /// Print detailed category breakdown.
  void printCategoryBreakdown(llvm::raw_ostream &os) const;

  /// Print process hotspots.
  void printProcessHotspots(llvm::raw_ostream &os, size_t n = 10) const;

  /// Print signal activity.
  void printSignalActivity(llvm::raw_ostream &os, size_t n = 10) const;

  /// Print histogram for a category.
  void printHistogram(llvm::raw_ostream &os, ProfileCategory category) const;

  /// Export profiling data to JSON.
  void exportJSON(llvm::raw_ostream &os) const;

  /// Export profiling data to CSV.
  void exportCSV(llvm::raw_ostream &os) const;

  //===------------------------------------------------------------------===//
  // Global Session Info
  //===------------------------------------------------------------------===//

  /// Get total profiling time.
  HighResolutionTimer::Duration getTotalTime() const;

  /// Get session name.
  const std::string &getSessionName() const { return sessionName; }

private:
  /// Add a sample to a histogram.
  void addToHistogram(ProfileCategory category, uint64_t durationNs);

  Config config;
  bool enabled;
  std::string sessionName;
  HighResolutionTimer sessionTimer;

  // Category statistics
  CategoryStatistics
      categoryStats[static_cast<size_t>(ProfileCategory::NumCategories)];

  // Detailed samples
  std::vector<ProfileSample> samples;
  std::mutex samplesMutex;
  std::atomic<size_t> sampleCounter{0};

  // Per-process profiles
  llvm::DenseMap<ProcessId, std::unique_ptr<ProcessProfile>> processProfiles;
  std::mutex processProfilesMutex;

  // Per-signal profiles
  llvm::DenseMap<SignalId, std::unique_ptr<SignalProfile>> signalProfiles;
  std::mutex signalProfilesMutex;

  // Histograms
  std::vector<size_t>
      histograms[static_cast<size_t>(ProfileCategory::NumCategories)];
  std::mutex histogramMutex;
};

//===----------------------------------------------------------------------===//
// ProfilerMacros - Convenience macros for profiling
//===----------------------------------------------------------------------===//

/// Macro for scoped profiling (creates a variable named __profiler_scope_N).
#define SIM_PROFILE_SCOPE(profiler, category)                                  \
  ScopedProfiler __profiler_scope_##__LINE__(profiler, category)

#define SIM_PROFILE_SCOPE_NAMED(profiler, category, name)                      \
  ScopedProfiler __profiler_scope_##__LINE__(profiler, category, name)

#define SIM_PROFILE_SCOPE_CTX(profiler, category, name, ctx)                   \
  ScopedProfiler __profiler_scope_##__LINE__(profiler, category, name, ctx)

/// Conditional profiling macros.
#define SIM_PROFILE_IF(profiler, condition, category)                          \
  ScopedProfiler __profiler_scope_##__LINE__(                                  \
      (condition) ? profiler : nullptr, category)

//===----------------------------------------------------------------------===//
// BottleneckAnalyzer - Identifies performance bottlenecks
//===----------------------------------------------------------------------===//

/// Analyzes profiling data to identify bottlenecks.
class BottleneckAnalyzer {
public:
  BottleneckAnalyzer(const PerformanceProfiler &profiler)
      : profiler(profiler) {}

  /// Bottleneck description.
  struct Bottleneck {
    enum class Type {
      HighEventSchedulingOverhead,
      SlowProcess,
      FrequentSignalUpdate,
      SynchronizationContention,
      MemoryBound,
      Other
    };

    Type type;
    std::string description;
    double severity; // 0-1, higher is worse
    std::string suggestion;
  };

  /// Analyze and return identified bottlenecks.
  std::vector<Bottleneck> analyze();

  /// Get the dominant bottleneck category.
  ProfileCategory getDominantCategory() const;

  /// Check for specific bottleneck types.
  bool hasSchedulingOverhead() const;
  bool hasSlowProcesses() const;
  bool hasSynchronizationIssues() const;

  /// Print bottleneck report.
  void printReport(llvm::raw_ostream &os);

private:
  const PerformanceProfiler &profiler;
};

//===----------------------------------------------------------------------===//
// ProfilerIntegration - Hooks for integrating with scheduler
//===----------------------------------------------------------------------===//

/// Helper class for integrating profiler with ProcessScheduler.
class ProfilerIntegration {
public:
  ProfilerIntegration(PerformanceProfiler &profiler,
                      ProcessScheduler &scheduler);

  /// Install profiling hooks.
  void install();

  /// Remove profiling hooks.
  void uninstall();

  /// Check if hooks are installed.
  bool isInstalled() const { return installed; }

private:
  PerformanceProfiler &profiler;
  ProcessScheduler &scheduler;
  bool installed;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_PERFORMANCEPROFILER_H
