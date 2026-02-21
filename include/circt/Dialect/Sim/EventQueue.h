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
#include <cstring>
#include <map>
#include <memory>
#include <queue>
#include <type_traits>
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
///
/// Uses an inline small-buffer optimization (64 bytes) to avoid heap
/// allocation for the common event lambdas (signal updates, process
/// resumes). This eliminates ~9% malloc/free overhead in hot simulation
/// loops where millions of events are created and destroyed per second.
class Event {
public:
  Event() = default;

  /// Construct from any callable. Small lambdas (<= 64 bytes) are stored
  /// inline to avoid heap allocation. Larger lambdas fall back to heap.
  template <typename Callable,
            typename = std::enable_if_t<
                !std::is_same_v<std::decay_t<Callable>, Event>>>
  explicit Event(Callable &&cb) {
    using T = std::decay_t<Callable>;
    if constexpr (sizeof(T) <= kInlineSize &&
                  alignof(T) <= alignof(std::max_align_t)) {
      // Inline storage — no heap allocation.
      new (storage) T(std::forward<Callable>(cb));
      invoker = &invokeImpl<T>;
      destroyer = &destroyImpl<T>;
    } else {
      // Heap fallback for large captures (rare path).
      auto *heap = new T(std::forward<Callable>(cb));
      std::memcpy(storage, &heap, sizeof(heap));
      invoker = &invokeHeapImpl<T>;
      destroyer = &destroyHeapImpl<T>;
    }
  }

  // Move constructor — uses memcpy for trivial relocation of the stored
  // callable. This is safe for all lambdas capturing pointers, integers,
  // APInt (inline mode), and SignalValue.
  Event(Event &&other) noexcept
      : invoker(other.invoker), destroyer(other.destroyer) {
    if (invoker) {
      std::memcpy(storage, other.storage, kInlineSize);
      other.invoker = nullptr;
      other.destroyer = nullptr;
    }
  }

  Event &operator=(Event &&other) noexcept {
    if (this != &other) {
      if (invoker && destroyer)
        destroyer(storage);
      invoker = other.invoker;
      destroyer = other.destroyer;
      if (invoker) {
        std::memcpy(storage, other.storage, kInlineSize);
        other.invoker = nullptr;
        other.destroyer = nullptr;
      }
    }
    return *this;
  }

  // No copy — events are move-only to avoid double-execution.
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;

  ~Event() {
    if (invoker && destroyer)
      destroyer(storage);
  }

  void execute() {
    if (invoker)
      invoker(storage);
  }

  bool isValid() const { return invoker != nullptr; }

private:
  /// 64 bytes is enough for the common hot-path event lambdas
  /// (signal updates ~40 bytes, process resumes ~16 bytes).
  static constexpr size_t kInlineSize = 64;

  // Inline-stored callable: invoke/destroy directly in storage.
  template <typename T>
  static void invokeImpl(void *p) {
    (*static_cast<T *>(p))();
  }

  template <typename T>
  static void destroyImpl(void *p) {
    static_cast<T *>(p)->~T();
  }

  // Heap-stored callable: pointer stored in first bytes of storage.
  template <typename T>
  static void invokeHeapImpl(void *p) {
    T *heap;
    std::memcpy(&heap, p, sizeof(heap));
    (*heap)();
  }

  template <typename T>
  static void destroyHeapImpl(void *p) {
    T *heap;
    std::memcpy(&heap, p, sizeof(heap));
    delete heap;
  }

  using FnPtr = void (*)(void *);
  FnPtr invoker = nullptr;
  FnPtr destroyer = nullptr;
  alignas(std::max_align_t) char storage[kInlineSize];
};

//===----------------------------------------------------------------------===//
// DeltaCycleQueue - Per-region event queues within a delta cycle
//===----------------------------------------------------------------------===//

/// Manages event queues for all scheduling regions within a single delta cycle.
/// Provides O(1) insertion and removal of events per region.
/// Uses a bitmask to track which regions have events for fast iteration.
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

  /// Check if any region has pending events (O(1) via bitmask).
  bool hasAnyEvents() const { return activeRegionMask != 0; }

  /// Pop all events from the specified region.
  std::vector<Event> popRegionEvents(SchedulingRegion region);

  /// Execute all events in the specified region in-place and clear.
  /// Returns the number of events executed. More efficient than
  /// popRegionEvents() as it avoids moving events into a temporary vector.
  size_t executeAndClearRegion(SchedulingRegion region);

  /// Get the number of events in a specific region.
  size_t getEventCount(SchedulingRegion region) const;

  /// Clear all events from all regions.
  void clear();

private:
  /// Bitmask tracking which regions have pending events.
  uint16_t activeRegionMask = 0;
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
    /// Fixed-size array of delta queues for delta steps 0..kInlineDeltaSlots-1.
    /// Avoids std::map RB-tree overhead for the common case where delta steps
    /// are small (typically 0 and 1 for clock toggle probe→drive patterns).
    static constexpr uint32_t kInlineDeltaSlots = 4;
    DeltaCycleQueue deltaQueues[kInlineDeltaSlots];
    /// Overflow map for delta steps >= kInlineDeltaSlots.
    std::map<uint32_t, DeltaCycleQueue> extraDeltaQueues;
    bool hasEvents = false;

    /// Get or create a delta queue for the given delta step.
    DeltaCycleQueue &getDeltaQueue(uint32_t deltaStep) {
      if (deltaStep < kInlineDeltaSlots)
        return deltaQueues[deltaStep];
      return extraDeltaQueues[deltaStep];
    }

    /// Find a delta queue for the given step, returning nullptr if not found.
    DeltaCycleQueue *findDeltaQueue(uint32_t deltaStep) {
      if (deltaStep < kInlineDeltaSlots)
        return deltaQueues[deltaStep].hasAnyEvents()
                   ? &deltaQueues[deltaStep]
                   : nullptr;
      auto it = extraDeltaQueues.find(deltaStep);
      return it != extraDeltaQueues.end() ? &it->second : nullptr;
    }

    /// Check if there are any events in any delta queue.
    bool hasAnyEvents() const {
      for (uint32_t i = 0; i < kInlineDeltaSlots; ++i)
        if (deltaQueues[i].hasAnyEvents())
          return true;
      for (const auto &[delta, queue] : extraDeltaQueues)
        if (queue.hasAnyEvents())
          return true;
      return false;
    }

    /// Get the minimum delta step with events.
    uint32_t getMinDeltaStep() const {
      for (uint32_t i = 0; i < kInlineDeltaSlots; ++i)
        if (deltaQueues[i].hasAnyEvents())
          return i;
      for (const auto &[delta, queue] : extraDeltaQueues)
        if (queue.hasAnyEvents())
          return delta;
      return UINT32_MAX;
    }

    /// Clear all delta queues.
    void clear() {
      for (uint32_t i = 0; i < kInlineDeltaSlots; ++i)
        deltaQueues[i].clear();
      extraDeltaQueues.clear();
      hasEvents = false;
    }

    /// Erase a specific delta queue (for cleanup after processing).
    void eraseDeltaQueue(uint32_t deltaStep) {
      if (deltaStep < kInlineDeltaSlots) {
        deltaQueues[deltaStep].clear();
      } else {
        extraDeltaQueues.erase(deltaStep);
      }
    }
  };

  struct Level {
    std::vector<Slot> slots;
    size_t currentSlot = 0;
    /// Bitmask tracking which slots have events (256 bits = 4 × uint64_t).
    /// Enables O(popcount) findNextEventTime instead of O(slotsPerLevel).
    static constexpr size_t kBitmaskWords = 4;
    uint64_t slotBitmask[kBitmaskWords] = {};
  };

  /// Get the slot index for a given time at a given level.
  size_t getSlotIndex(uint64_t time, size_t level) const;

  /// Bitmask helpers for fast slot scanning.
  void setSlotBit(size_t level, size_t slot) {
    levels[level].slotBitmask[slot / 64] |= (1ULL << (slot % 64));
  }
  void clearSlotBit(size_t level, size_t slot) {
    levels[level].slotBitmask[slot / 64] &= ~(1ULL << (slot % 64));
  }
  void updateSlotBit(size_t level, size_t slot, bool hasEvents) {
    if (hasEvents)
      setSlotBit(level, slot);
    else
      clearSlotBit(level, slot);
  }

  /// Move events from higher levels to lower levels as time advances.
  void cascade(size_t fromLevel);

  /// Find the next time with scheduled events.
  bool findNextEventTime(SimTime &nextTime);

  Config config;
  std::vector<Level> levels;
  SimTime currentTime;

  /// Precomputed resolution per level (avoids loop in getSlotIndex).
  std::vector<uint64_t> levelResolution;

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

  /// Advance the internal simulation time to the specified value (in fs).
  /// Does NOT process events — just moves the clock forward.
  void advanceTimeTo(uint64_t timeFs);

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
