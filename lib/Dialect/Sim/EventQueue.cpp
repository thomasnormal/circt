//===- EventQueue.cpp - Event-driven simulation infrastructure --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the core event queue infrastructure for event-driven
// simulation following IEEE 1800 (SystemVerilog) scheduling semantics.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <cassert>

#define DEBUG_TYPE "sim-event-queue"

using namespace circt;
using namespace circt::sim;

//===----------------------------------------------------------------------===//
// DeltaCycleQueue Implementation
//===----------------------------------------------------------------------===//

void DeltaCycleQueue::schedule(SchedulingRegion region, Event event) {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  size_t idx = static_cast<size_t>(region);
  regionQueues[idx].push_back(std::move(event));
  activeRegionMask |= (1u << idx);
}

SchedulingRegion
DeltaCycleQueue::getNextNonEmptyRegion(SchedulingRegion start) const {
  // Shift mask to skip regions before 'start', then find lowest set bit.
  unsigned startIdx = static_cast<unsigned>(start);
  uint16_t mask = activeRegionMask >> startIdx;
  if (mask == 0)
    return SchedulingRegion::NumRegions;
  unsigned offset = __builtin_ctz(mask);
  unsigned idx = startIdx + offset;
  if (idx >= static_cast<unsigned>(SchedulingRegion::NumRegions))
    return SchedulingRegion::NumRegions;
  return static_cast<SchedulingRegion>(idx);
}

bool DeltaCycleQueue::hasEvents(SchedulingRegion region) const {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  return (activeRegionMask & (1u << static_cast<size_t>(region))) != 0;
}

std::vector<Event> DeltaCycleQueue::popRegionEvents(SchedulingRegion region) {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  size_t idx = static_cast<size_t>(region);
  std::vector<Event> result;
  std::swap(result, regionQueues[idx]);
  activeRegionMask &= ~(1u << idx);
  return result;
}

size_t DeltaCycleQueue::executeAndClearRegion(SchedulingRegion region) {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  size_t idx = static_cast<size_t>(region);
  // Swap the queue out before executing — event callbacks may schedule
  // new events into the same region, so we must not iterate in-place.
  std::vector<Event> batch;
  std::swap(batch, regionQueues[idx]);
  activeRegionMask &= ~(1u << idx);
  size_t count = batch.size();
  for (auto &event : batch)
    event.execute();
  // Reclaim the buffer capacity: clear the executed events first (critical —
  // without this, stale events would be re-executed when new events are
  // pushed into the reused vector), then move the empty-but-allocated vector
  // back so its capacity is reused on the next schedule() call.
  batch.clear();
  if (regionQueues[idx].empty())
    regionQueues[idx] = std::move(batch);
  return count;
}

size_t DeltaCycleQueue::getEventCount(SchedulingRegion region) const {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  return regionQueues[static_cast<size_t>(region)].size();
}

void DeltaCycleQueue::clear() {
  // Only clear regions that have events (avoids touching cold cache lines).
  uint16_t mask = activeRegionMask;
  while (mask) {
    unsigned idx = __builtin_ctz(mask);
    regionQueues[idx].clear();
    mask &= mask - 1; // Clear lowest set bit
  }
  activeRegionMask = 0;
}

//===----------------------------------------------------------------------===//
// TimeWheel Implementation
//===----------------------------------------------------------------------===//

TimeWheel::TimeWheel(Config config) : config(config) {
  assert(config.slotsPerLevel <= 256 &&
         "Slot bitmask only supports up to 256 slots per level");
  levels.resize(config.numLevels);
  for (size_t i = 0; i < config.numLevels; ++i) {
    levels[i].slots.resize(config.slotsPerLevel);
    for (auto &slot : levels[i].slots) {
      slot.baseTime = 0;
      slot.hasEvents = false;
    }
  }
  // Precompute resolution per level to avoid loop in getSlotIndex.
  levelResolution.resize(config.numLevels);
  uint64_t res = config.baseResolution;
  for (size_t i = 0; i < config.numLevels; ++i) {
    levelResolution[i] = res;
    res *= config.slotsPerLevel;
  }
}

TimeWheel::~TimeWheel() = default;

size_t TimeWheel::getSlotIndex(uint64_t time, size_t level) const {
  return (time / levelResolution[level]) % config.slotsPerLevel;
}

void TimeWheel::schedule(const SimTime &time, SchedulingRegion region,
                         Event event) {
  if (time < currentTime) {
    LLVM_DEBUG(llvm::dbgs()
               << "Warning: Scheduling event in the past, using current time\n");
    // Schedule at current time instead
    schedule(currentTime, region, std::move(event));
    return;
  }

  uint64_t targetTime = time.realTime;
  uint64_t timeDelta = targetTime - currentTime.realTime;

  // Calculate which level this event belongs to.
  // Each level covers levelResolution[level] * slotsPerLevel femtoseconds.
  size_t level = 0;
  while (level < config.numLevels - 1 &&
         timeDelta >= levelResolution[level] * config.slotsPerLevel) {
    ++level;
  }

  // If beyond all levels, use overflow
  uint64_t levelCapacity = levelResolution[level] * config.slotsPerLevel;
  if (level >= config.numLevels || timeDelta >= levelCapacity) {
    overflow[targetTime].schedule(region, std::move(event));
    ++totalEvents;
    return;
  }

  // Calculate slot index
  size_t slotIndex = getSlotIndex(targetTime, level);
  auto &slot = levels[level].slots[slotIndex];

  // A wheel slot can host multiple deltas/regions for ONE absolute time.
  // If another absolute time hashes to the same slot while it's occupied,
  // storing it in-place would overwrite slot.baseTime and retime existing
  // events. Defer such collisions to overflow to preserve ordering.
  if (slot.hasEvents && slot.baseTime != targetTime) {
    overflow[targetTime].schedule(region, std::move(event));
    ++totalEvents;
    return;
  }

  // Determine the delta step to use.
  // If scheduling at current real time, use the requested delta step, but at
  // minimum use the current delta step (can't schedule in the past).
  // If scheduling at a future real time, always use delta step 0.
  uint32_t deltaStep = 0;
  if (time.realTime == currentTime.realTime) {
    // For same real time, use the max of requested delta and current delta
    deltaStep = std::max(time.deltaStep, currentTime.deltaStep);
  }
  // For future real times, deltaStep remains 0

  slot.getDeltaQueue(deltaStep).schedule(region, std::move(event));
  slot.hasEvents = true;
  slot.baseTime = targetTime;
  setSlotBit(level, slotIndex);
  ++totalEvents;
}

void TimeWheel::cascade(size_t fromLevel) {
  if (fromLevel >= config.numLevels)
    return;

  // Move events from higher-level slots to lower levels or execute them
  auto &level = levels[fromLevel];
  size_t slotIdx = level.currentSlot;
  auto &slot = level.slots[slotIdx];

  if (!slot.hasEvents)
    return;

  // If this is level 0, events are ready to execute
  if (fromLevel == 0)
    return;

  // Move events from this slot to lower levels
  if (slot.hasAnyEvents()) {
    // Helper to cascade events from a single delta queue.
    // Uses the bitmask to skip empty regions.
    auto cascadeQueue = [&](uint32_t deltaStep, DeltaCycleQueue &deltaQueue) {
      auto region = deltaQueue.getNextNonEmptyRegion(SchedulingRegion::Preponed);
      while (region != SchedulingRegion::NumRegions) {
        auto events = deltaQueue.popRegionEvents(region);
        for (auto &event : events) {
          SimTime eventTime(slot.baseTime, deltaStep,
                            static_cast<uint8_t>(region));
          // Decrement total since schedule will increment it again
          --totalEvents;
          schedule(eventTime, region, std::move(event));
        }
        unsigned nextIdx = static_cast<unsigned>(region) + 1;
        if (nextIdx >= static_cast<unsigned>(SchedulingRegion::NumRegions))
          break;
        region = deltaQueue.getNextNonEmptyRegion(
            static_cast<SchedulingRegion>(nextIdx));
      }
    };

    // Cascade inline delta queues
    for (uint32_t d = 0; d < Slot::kInlineDeltaSlots; ++d)
      if (slot.deltaQueues[d].hasAnyEvents())
        cascadeQueue(d, slot.deltaQueues[d]);

    // Cascade extra delta queues
    for (auto &[deltaStep, deltaQueue] : slot.extraDeltaQueues)
      cascadeQueue(deltaStep, deltaQueue);
  }
  slot.clear();
  clearSlotBit(fromLevel, slotIdx);
}

bool TimeWheel::findNextEventTime(SimTime &nextTime) {
  // First check current time's slot for any events at or after current delta.
  auto &currentSlot = levels[0].slots[levels[0].currentSlot];
  if (currentSlot.hasEvents) {
    // Check inline delta queues first
    for (uint32_t d = currentTime.deltaStep; d < Slot::kInlineDeltaSlots; ++d) {
      if (currentSlot.deltaQueues[d].hasAnyEvents()) {
        nextTime = SimTime(currentTime.realTime, d, 0);
        return true;
      }
    }
    // Check extra delta queues for steps >= kInlineDeltaSlots
    for (auto &[deltaStep, queue] : currentSlot.extraDeltaQueues) {
      if (deltaStep >= currentTime.deltaStep && queue.hasAnyEvents()) {
        nextTime = SimTime(currentTime.realTime, deltaStep, 0);
        return true;
      }
    }
  }

  // Bitmask scan: iterate only slots with events using per-level bitmasks.
  // This is O(popcount) instead of O(slotsPerLevel * numLevels).
  uint64_t minTime = UINT64_MAX;
  bool found = false;

  for (size_t level = 0; level < config.numLevels; ++level) {
    auto &lvl = levels[level];
    for (size_t w = 0; w < Level::kBitmaskWords; ++w) {
      uint64_t word = lvl.slotBitmask[w];
      while (word) {
        unsigned bit = __builtin_ctzll(word);
        size_t slotIdx = w * 64 + bit;
        auto &slot = lvl.slots[slotIdx];
        if (slot.baseTime > currentTime.realTime && slot.baseTime < minTime) {
          minTime = slot.baseTime;
          found = true;
        }
        word &= word - 1; // Clear lowest set bit
      }
    }
  }

  // Check overflow - it's a sorted map so begin() is the minimum
  if (!overflow.empty()) {
    auto it = overflow.begin();
    while (it != overflow.end() && it->first <= currentTime.realTime)
      ++it;
    if (it != overflow.end() && it->first < minTime) {
      minTime = it->first;
      found = true;
    }
  }

  if (found) {
    nextTime = SimTime(minTime, 0, 0);
    return true;
  }

  return false;
}

bool TimeWheel::advanceToNextEvent() {
  SimTime nextTime;
  if (!findNextEventTime(nextTime))
    return false;

  if (nextTime.realTime > currentTime.realTime) {
    currentTime = SimTime(nextTime.realTime, 0, 0);

    // Update slot positions
    for (size_t level = 0; level < config.numLevels; ++level) {
      levels[level].currentSlot = getSlotIndex(currentTime.realTime, level);
    }

    // Cascade events from higher levels
    for (size_t level = config.numLevels - 1; level > 0; --level) {
      cascade(level);
    }

    // Move overflow events if they're now in range.
    // Use bitmask to skip empty regions.
    while (!overflow.empty() && overflow.begin()->first == currentTime.realTime) {
      auto node = overflow.extract(overflow.begin());
      auto &deltaQueue = node.mapped();
      auto region = deltaQueue.getNextNonEmptyRegion(SchedulingRegion::Preponed);
      while (region != SchedulingRegion::NumRegions) {
        auto events = deltaQueue.popRegionEvents(region);
        for (auto &event : events) {
          --totalEvents;
          schedule(SimTime(currentTime.realTime, 0,
                           static_cast<uint8_t>(region)),
                   region, std::move(event));
        }
        unsigned nextIdx = static_cast<unsigned>(region) + 1;
        if (nextIdx >= static_cast<unsigned>(SchedulingRegion::NumRegions))
          break;
        region = deltaQueue.getNextNonEmptyRegion(
            static_cast<SchedulingRegion>(nextIdx));
      }
    }
  }

  return true;
}

size_t TimeWheel::processCurrentRegion() {
  size_t slotIdx = getSlotIndex(currentTime.realTime, 0);
  auto &slot = levels[0].slots[slotIdx];

  // Get the delta queue for the current delta step
  DeltaCycleQueue *deltaQueuePtr = nullptr;
  if (currentTime.deltaStep < Slot::kInlineDeltaSlots) {
    if (slot.deltaQueues[currentTime.deltaStep].hasAnyEvents())
      deltaQueuePtr = &slot.deltaQueues[currentTime.deltaStep];
  } else {
    auto it = slot.extraDeltaQueues.find(currentTime.deltaStep);
    if (it != slot.extraDeltaQueues.end())
      deltaQueuePtr = &it->second;
  }
  if (!deltaQueuePtr) {
    // No events at current delta step
    slot.hasEvents = slot.hasAnyEvents();
    updateSlotBit(0, slotIdx, slot.hasEvents);
    return 0;
  }

  auto &deltaQueue = *deltaQueuePtr;

  // Find the next non-empty region starting from current region
  auto region = deltaQueue.getNextNonEmptyRegion(
      static_cast<SchedulingRegion>(currentTime.region));
  if (region == SchedulingRegion::NumRegions) {
    // No events to process in current or later regions
    slot.hasEvents = slot.hasAnyEvents();
    updateSlotBit(0, slotIdx, slot.hasEvents);
    return 0;
  }

  // Update current region to match
  currentTime.region = static_cast<uint8_t>(region);

  size_t count = deltaQueue.executeAndClearRegion(region);
  totalEvents -= count;

  // Advance to next region
  auto nextRegion = deltaQueue.getNextNonEmptyRegion(
      static_cast<SchedulingRegion>(currentTime.region + 1));
  if (nextRegion != SchedulingRegion::NumRegions) {
    currentTime.region = static_cast<uint8_t>(nextRegion);
  } else {
    // No more events in this delta, check if we need to start a new delta
    // This will be handled by the scheduler
    slot.hasEvents = slot.hasAnyEvents();
    updateSlotBit(0, slotIdx, slot.hasEvents);
  }

  return count;
}

size_t TimeWheel::processCurrentDelta() {
  size_t total = 0;
  size_t slotIdx = getSlotIndex(currentTime.realTime, 0);
  auto &slot = levels[0].slots[slotIdx];

  // Get the delta queue for the current delta step.
  // Use inline array for steps < kInlineDeltaSlots (avoids RB-tree overhead).
  DeltaCycleQueue *deltaQueue = nullptr;
  bool isInline = (currentTime.deltaStep < Slot::kInlineDeltaSlots);
  if (isInline) {
    if (slot.deltaQueues[currentTime.deltaStep].hasAnyEvents())
      deltaQueue = &slot.deltaQueues[currentTime.deltaStep];
  } else {
    auto it = slot.extraDeltaQueues.find(currentTime.deltaStep);
    if (it != slot.extraDeltaQueues.end())
      deltaQueue = &it->second;
  }

  if (deltaQueue) {
    // Process all regions, repeating if event callbacks schedule new events
    // at the same delta step in already-processed regions. Without the outer
    // loop, such events become orphaned: they inflate totalEvents permanently
    // but are never processed (processCurrentDelta advances past this delta),
    // causing advanceTime() to spin forever.
    bool madeProgress = true;
    while (madeProgress) {
      madeProgress = false;
      // Use bitmask to skip empty regions.
      auto region = deltaQueue->getNextNonEmptyRegion(
          static_cast<SchedulingRegion>(currentTime.region));
      while (region != SchedulingRegion::NumRegions) {
        // Execute in-place to avoid vector move overhead.
        size_t count = deltaQueue->executeAndClearRegion(region);
        totalEvents -= count;
        total += count;
        madeProgress = true;

        // Get next non-empty region after the one we just processed.
        unsigned nextIdx = static_cast<unsigned>(region) + 1;
        if (nextIdx >= static_cast<unsigned>(SchedulingRegion::NumRegions))
          break;
        region = deltaQueue->getNextNonEmptyRegion(
            static_cast<SchedulingRegion>(nextIdx));
      }
    }

    // Clean up empty delta queue (only needed for overflow map entries)
    if (!deltaQueue->hasAnyEvents() && !isInline)
      slot.extraDeltaQueues.erase(currentTime.deltaStep);
  }

  // Reset region to start of next delta
  currentTime.region = 0;
  currentTime.deltaStep++;
  slot.hasEvents = slot.hasAnyEvents();
  updateSlotBit(0, slotIdx, slot.hasEvents);

  return total;
}

size_t TimeWheel::processCurrentTime() {
  size_t total = 0;

  // Process all delta cycles at current time
  while (true) {
    size_t processed = processCurrentDelta();
    if (processed == 0)
      break;
    total += processed;
  }

  return total;
}

bool TimeWheel::hasEvents() const { return totalEvents > 0; }

size_t TimeWheel::getEventCount() const { return totalEvents; }

void TimeWheel::clear() {
  for (auto &level : levels) {
    for (auto &slot : level.slots) {
      slot.clear();
    }
    level.currentSlot = 0;
    std::memset(level.slotBitmask, 0, sizeof(level.slotBitmask));
  }
  overflow.clear();
  currentTime = SimTime();
  totalEvents = 0;
}

//===----------------------------------------------------------------------===//
// EventScheduler Implementation
//===----------------------------------------------------------------------===//

EventScheduler::EventScheduler() : wheel(std::make_unique<TimeWheel>()) {}

EventScheduler::~EventScheduler() = default;

void EventScheduler::schedule(const SimTime &time, SchedulingRegion region,
                              Event event) {
  wheel->schedule(time, region, std::move(event));
}

void EventScheduler::scheduleNow(SchedulingRegion region, Event event) {
  wheel->schedule(wheel->getCurrentTime(), region, std::move(event));
}

void EventScheduler::scheduleNextDelta(SchedulingRegion region, Event event) {
  wheel->schedule(wheel->getCurrentTime().nextDelta(), region, std::move(event));
}

void EventScheduler::scheduleDelay(uint64_t delayFemtoseconds,
                                   SchedulingRegion region, Event event) {
  wheel->schedule(wheel->getCurrentTime().advanceTime(delayFemtoseconds), region,
                  std::move(event));
}

const SimTime &EventScheduler::getCurrentTime() const {
  return wheel->getCurrentTime();
}

SimTime EventScheduler::runUntil(uint64_t maxTimeFemtoseconds) {
  while (!isComplete() && wheel->getCurrentTime().realTime <= maxTimeFemtoseconds) {
    if (!stepDelta()) {
      // Try to advance to next event
      if (!wheel->advanceToNextEvent())
        break;
      ++stats.realTimeAdvances;
    }
  }
  return wheel->getCurrentTime();
}

size_t EventScheduler::runDeltas(size_t maxDeltas) {
  size_t deltasProcessed = 0;
  while (deltasProcessed < maxDeltas && !isComplete()) {
    if (stepDelta())
      ++deltasProcessed;
    else
      break;
  }
  return deltasProcessed;
}

bool EventScheduler::stepRegion() {
  if (!wheel->hasEvents())
    return false;

  size_t processed = wheel->processCurrentRegion();
  stats.eventsProcessed += processed;
  return processed > 0;
}

bool EventScheduler::stepDelta() {
  if (!wheel->hasEvents())
    return false;

  size_t processed = wheel->processCurrentDelta();
  stats.eventsProcessed += processed;
  if (processed > 0)
    ++stats.deltaCycles;
  return processed > 0;
}

bool EventScheduler::advanceToNextTime() {
  if (!wheel->hasEvents())
    return false;

  // Only advance if current time has no events to process
  // (This ensures we don't skip events at the current time)
  SimTime before = wheel->getCurrentTime();
  if (wheel->advanceToNextEvent()) {
    if (wheel->getCurrentTime().realTime > before.realTime) {
      ++stats.realTimeAdvances;
      return true;
    }
  }
  return false;
}

void EventScheduler::advanceTimeTo(uint64_t timeFs) {
  // Advance the internal simulation clock to the specified time without
  // processing any events. Used by minnow/clock-domain bypasses when no
  // TimeWheel events exist but sim time must advance for process wake-ups.
  if (timeFs > wheel->getCurrentTime().realTime) {
    wheel->setCurrentTime(SimTime(timeFs, 0, 0));
    ++stats.realTimeAdvances;
  }
}

bool EventScheduler::isComplete() const { return !wheel->hasEvents(); }

void EventScheduler::reset() {
  wheel->clear();
  stats = Statistics();
}
