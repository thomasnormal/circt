//===- EventQueue.h - Event-driven simulation infrastructure ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the core event queue infrastructure for event-driven
// simulation following IEEE 1800 (SystemVerilog) scheduling semantics.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_EVENTQUEUE_H
#define CIRCT_DIALECT_SIM_EVENTQUEUE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <vector>

namespace circt {
namespace sim {

//===----------------------------------------------------------------------===//
// SimTime - Simulation time representation
//===----------------------------------------------------------------------===//

/// Represents a point in simulation time following IEEE 1800 scheduling.
/// Time is represented as:
/// - realTime: Absolute simulation time in femtoseconds (10^-15 seconds)
/// - deltaStep: Delta cycle number within the same real time
/// - region: IEEE 1800 scheduling region within the delta cycle
struct SimTime {
  uint64_t realTime;  // Time in femtoseconds
  uint32_t deltaStep; // Delta cycle number
  uint8_t region;     // Scheduling region

  SimTime() : realTime(0), deltaStep(0), region(0) {}
  SimTime(uint64_t time, uint32_t delta = 0, uint8_t reg = 0)
      : realTime(time), deltaStep(delta), region(reg) {}

  bool operator==(const SimTime &other) const {
    return realTime == other.realTime && deltaStep == other.deltaStep &&
           region == other.region;
  }

  bool operator!=(const SimTime &other) const { return !(*this == other); }

  bool operator<(const SimTime &other) const {
    if (realTime != other.realTime)
      return realTime < other.realTime;
    if (deltaStep != other.deltaStep)
      return deltaStep < other.deltaStep;
    return region < other.region;
  }

  bool operator<=(const SimTime &other) const {
    return *this < other || *this == other;
  }

  bool operator>(const SimTime &other) const { return !(*this <= other); }

  bool operator>=(const SimTime &other) const { return !(*this < other); }

  /// Advance to the next delta cycle, resetting the region.
  SimTime nextDelta() const {
    return SimTime(realTime, deltaStep + 1, 0);
  }

  /// Advance to the next scheduling region.
  SimTime nextRegion() const {
    return SimTime(realTime, deltaStep, region + 1);
  }

  /// Advance real time, resetting delta and region.
  SimTime advanceTime(uint64_t femtoseconds) const {
    return SimTime(realTime + femtoseconds, 0, 0);
  }
};

//===----------------------------------------------------------------------===//
// SchedulingRegion - IEEE 1800 scheduling regions
//===----------------------------------------------------------------------===//

/// IEEE 1800 SystemVerilog scheduling regions within a time slot.
/// Events must be processed in order from Preponed to Postponed.
enum class SchedulingRegion : uint8_t {
  /// Sample values before any changes (for PLI sampling)
  Preponed = 0,

  /// Active region: blocking assignments, continuous assignments,
  /// $display, evaluate RHS of nonblocking assignments
  Active = 1,

  /// Inactive region: #0 blocking assignments
  Inactive = 2,

  /// NBA region: nonblocking assignment updates
  NBA = 3,

  /// Observed region: property evaluation for assertions
  Observed = 4,

  /// Reactive region: program block code, checker code
  Reactive = 5,

  /// Re-Inactive region: #0 delays in reactive region
  ReInactive = 6,

  /// Re-NBA region: nonblocking assignments from reactive region
  ReNBA = 7,

  /// Postponed region: $strobe, $monitor
  Postponed = 8,

  /// Number of regions (for array sizing)
  NumRegions = 9
};

/// Get the name of a scheduling region for debugging.
inline const char *getSchedulingRegionName(SchedulingRegion region) {
  switch (region) {
  case SchedulingRegion::Preponed:
    return "Preponed";
  case SchedulingRegion::Active:
    return "Active";
  case SchedulingRegion::Inactive:
    return "Inactive";
  case SchedulingRegion::NBA:
    return "NBA";
  case SchedulingRegion::Observed:
    return "Observed";
  case SchedulingRegion::Reactive:
    return "Reactive";
  case SchedulingRegion::ReInactive:
    return "ReInactive";
  case SchedulingRegion::ReNBA:
    return "ReNBA";
  case SchedulingRegion::Postponed:
    return "Postponed";
  default:
    return "Unknown";
  }
}

//===----------------------------------------------------------------------===//
// Event - Simulation event representation
//===----------------------------------------------------------------------===//

/// An event in the simulation. Events are callable objects scheduled at
/// specific simulation times.
class Event {
public:
  using Callback = std::function<void()>;

  Event() = default;
  explicit Event(Callback callback) : callback(std::move(callback)) {}

  void execute() {
    if (callback)
      callback();
  }

  bool isValid() const { return callback != nullptr; }

private:
  Callback callback;
};

//===----------------------------------------------------------------------===//
// DeltaCycleQueue - Per-region event queues within a delta cycle
//===----------------------------------------------------------------------===//

/// Manages event queues for all scheduling regions within a single delta cycle.
/// Provides O(1) insertion and removal of events per region.
class DeltaCycleQueue {
public:
  DeltaCycleQueue() = default;

  /// Schedule an event in the specified region.
  void schedule(SchedulingRegion region, Event event);

  /// Get the next non-empty region, starting from the given region.
  /// Returns NumRegions if all remaining regions are empty.
  SchedulingRegion getNextNonEmptyRegion(SchedulingRegion start) const;

  /// Check if a specific region has pending events.
  bool hasEvents(SchedulingRegion region) const;

  /// Check if any region has pending events.
  bool hasAnyEvents() const;

  /// Pop all events from the specified region.
  std::vector<Event> popRegionEvents(SchedulingRegion region);

  /// Get the number of events in a specific region.
  size_t getEventCount(SchedulingRegion region) const;

  /// Clear all events from all regions.
  void clear();

private:
  std::vector<Event>
      regionQueues[static_cast<size_t>(SchedulingRegion::NumRegions)];
};

//===----------------------------------------------------------------------===//
// TimeWheel - O(1) event scheduling data structure
//===----------------------------------------------------------------------===//

/// A time wheel data structure for efficient event scheduling.
/// Provides O(1) average-case insertion and O(1) extraction of the
/// minimum time event.
///
/// The implementation uses a hierarchical timing wheel with multiple levels:
/// - Level 0: Fine-grained slots (e.g., 1 femtosecond resolution)
/// - Level 1+: Coarser slots for far-future events
///
/// Events at the same real time are further organized by delta cycle
/// and scheduling region using DeltaCycleQueue.
class TimeWheel {
public:
  /// Configuration for the time wheel.
  struct Config {
    /// Number of slots per wheel level.
    size_t slotsPerLevel;

    /// Time resolution per slot at level 0 (in femtoseconds).
    uint64_t baseResolution;

    /// Number of levels in the hierarchy.
    size_t numLevels;

    Config() : slotsPerLevel(256), baseResolution(1), numLevels(4) {}
  };

  explicit TimeWheel(Config config = Config());
  ~TimeWheel();

  /// Schedule an event at the specified time and region.
  void schedule(const SimTime &time, SchedulingRegion region, Event event);

  /// Schedule an event at the specified time in the Active region.
  void schedule(const SimTime &time, Event event) {
    schedule(time, SchedulingRegion::Active, std::move(event));
  }

  /// Get the current simulation time.
  const SimTime &getCurrentTime() const { return currentTime; }

  /// Advance to the next scheduled event time.
  /// Returns false if there are no more events.
  bool advanceToNextEvent();

  /// Process all events in the current time slot's current region.
  /// Returns the number of events processed.
  size_t processCurrentRegion();

  /// Process all events in the current delta cycle (all regions).
  /// Returns the number of events processed.
  size_t processCurrentDelta();

  /// Process all events at the current real time (all deltas and regions).
  /// Returns the number of events processed.
  size_t processCurrentTime();

  /// Check if there are any pending events.
  bool hasEvents() const;

  /// Get the number of pending events (may be expensive).
  size_t getEventCount() const;

  /// Clear all pending events.
  void clear();

private:
  struct Slot {
    uint64_t baseTime;
    /// Map from delta step to event queue for that delta.
    /// This allows scheduling events at different delta steps within the same
    /// real time.
    std::map<uint32_t, DeltaCycleQueue> deltaQueues;
    bool hasEvents = false;

    /// Get or create a delta queue for the given delta step.
    DeltaCycleQueue &getDeltaQueue(uint32_t deltaStep) {
      return deltaQueues[deltaStep];
    }

    /// Check if there are any events in any delta queue.
    bool hasAnyEvents() const {
      for (const auto &[delta, queue] : deltaQueues)
        if (queue.hasAnyEvents())
          return true;
      return false;
    }

    /// Get the minimum delta step with events.
    uint32_t getMinDeltaStep() const {
      for (const auto &[delta, queue] : deltaQueues)
        if (queue.hasAnyEvents())
          return delta;
      return UINT32_MAX;
    }

    /// Clear all delta queues.
    void clear() {
      deltaQueues.clear();
      hasEvents = false;
    }
  };

  struct Level {
    std::vector<Slot> slots;
    size_t currentSlot = 0;
  };

  /// Get the slot index for a given time at a given level.
  size_t getSlotIndex(uint64_t time, size_t level) const;

  /// Move events from higher levels to lower levels as time advances.
  void cascade(size_t fromLevel);

  /// Find the next time with scheduled events.
  bool findNextEventTime(SimTime &nextTime);

  Config config;
  std::vector<Level> levels;
  SimTime currentTime;

  /// Overflow bucket for events beyond the wheel's range.
  std::map<uint64_t, DeltaCycleQueue> overflow;

  /// Total number of events across all slots.
  size_t totalEvents = 0;
};

//===----------------------------------------------------------------------===//
// EventScheduler - High-level simulation scheduler
//===----------------------------------------------------------------------===//

/// High-level interface for scheduling and executing simulation events.
/// Wraps the TimeWheel with additional functionality for delta cycle
/// management and time advancement.
class EventScheduler {
public:
  EventScheduler();
  ~EventScheduler();

  /// Schedule an event at the specified time and region.
  void schedule(const SimTime &time, SchedulingRegion region, Event event);

  /// Schedule an event at the current time in the specified region.
  void scheduleNow(SchedulingRegion region, Event event);

  /// Schedule an event for the next delta cycle.
  void scheduleNextDelta(SchedulingRegion region, Event event);

  /// Schedule an event at a future real time.
  void scheduleDelay(uint64_t delayFemtoseconds, SchedulingRegion region,
                     Event event);

  /// Get the current simulation time.
  const SimTime &getCurrentTime() const;

  /// Run the simulation until completion or the specified time limit.
  /// Returns the final simulation time.
  SimTime runUntil(uint64_t maxTimeFemtoseconds);

  /// Run the simulation for a specified number of delta cycles.
  /// Returns the number of deltas actually processed.
  size_t runDeltas(size_t maxDeltas);

  /// Step through one scheduling region.
  /// Returns false if simulation is complete.
  bool stepRegion();

  /// Step through one delta cycle.
  /// Returns false if simulation is complete.
  bool stepDelta();

  /// Advance time to the next scheduled event without processing it.
  /// This allows the caller to control when events are executed.
  /// Returns true if time was advanced, false if no events or already at next event.
  bool advanceToNextTime();

  /// Check if the simulation is complete (no more events).
  bool isComplete() const;

  /// Get statistics about the scheduler.
  struct Statistics {
    size_t eventsProcessed = 0;
    size_t deltaCycles = 0;
    size_t realTimeAdvances = 0;
  };

  const Statistics &getStatistics() const { return stats; }

  /// Reset the scheduler to initial state.
  void reset();

private:
  std::unique_ptr<TimeWheel> wheel;
  Statistics stats;
};

} // namespace sim
} // namespace circt

#endif // CIRCT_DIALECT_SIM_EVENTQUEUE_H
