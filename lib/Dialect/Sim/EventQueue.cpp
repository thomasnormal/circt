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
  regionQueues[static_cast<size_t>(region)].push_back(std::move(event));
}

SchedulingRegion
DeltaCycleQueue::getNextNonEmptyRegion(SchedulingRegion start) const {
  for (size_t i = static_cast<size_t>(start);
       i < static_cast<size_t>(SchedulingRegion::NumRegions); ++i) {
    if (!regionQueues[i].empty())
      return static_cast<SchedulingRegion>(i);
  }
  return SchedulingRegion::NumRegions;
}

bool DeltaCycleQueue::hasEvents(SchedulingRegion region) const {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  return !regionQueues[static_cast<size_t>(region)].empty();
}

bool DeltaCycleQueue::hasAnyEvents() const {
  for (size_t i = 0; i < static_cast<size_t>(SchedulingRegion::NumRegions); ++i)
    if (!regionQueues[i].empty())
      return true;
  return false;
}

std::vector<Event> DeltaCycleQueue::popRegionEvents(SchedulingRegion region) {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  std::vector<Event> result;
  std::swap(result, regionQueues[static_cast<size_t>(region)]);
  return result;
}

size_t DeltaCycleQueue::getEventCount(SchedulingRegion region) const {
  assert(static_cast<size_t>(region) <
             static_cast<size_t>(SchedulingRegion::NumRegions) &&
         "Invalid scheduling region");
  return regionQueues[static_cast<size_t>(region)].size();
}

void DeltaCycleQueue::clear() {
  for (size_t i = 0; i < static_cast<size_t>(SchedulingRegion::NumRegions); ++i)
    regionQueues[i].clear();
}

//===----------------------------------------------------------------------===//
// TimeWheel Implementation
//===----------------------------------------------------------------------===//

TimeWheel::TimeWheel(Config config) : config(config) {
  levels.resize(config.numLevels);
  for (size_t i = 0; i < config.numLevels; ++i) {
    levels[i].slots.resize(config.slotsPerLevel);
    for (auto &slot : levels[i].slots) {
      slot.baseTime = 0;
      slot.hasEvents = false;
    }
  }
}

TimeWheel::~TimeWheel() = default;

size_t TimeWheel::getSlotIndex(uint64_t time, size_t level) const {
  // Calculate the resolution at this level
  uint64_t resolution = config.baseResolution;
  for (size_t i = 0; i < level; ++i)
    resolution *= config.slotsPerLevel;

  // Get the slot index within this level
  return (time / resolution) % config.slotsPerLevel;
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

  // Calculate which level this event belongs to
  uint64_t levelCapacity = config.baseResolution * config.slotsPerLevel;
  size_t level = 0;
  while (level < config.numLevels - 1 && timeDelta >= levelCapacity) {
    levelCapacity *= config.slotsPerLevel;
    ++level;
  }

  // If beyond all levels, use overflow
  if (level >= config.numLevels || timeDelta >= levelCapacity) {
    overflow[targetTime].schedule(region, std::move(event));
    ++totalEvents;
    return;
  }

  // Calculate slot index
  size_t slotIndex = getSlotIndex(targetTime, level);
  auto &slot = levels[level].slots[slotIndex];

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
    // Re-schedule all events at the correct lower level, preserving delta steps
    for (auto &[deltaStep, deltaQueue] : slot.deltaQueues) {
      for (size_t regionIdx = 0;
           regionIdx < static_cast<size_t>(SchedulingRegion::NumRegions);
           ++regionIdx) {
        auto region = static_cast<SchedulingRegion>(regionIdx);
        auto events = deltaQueue.popRegionEvents(region);
        for (auto &event : events) {
          SimTime eventTime(slot.baseTime, deltaStep,
                            static_cast<uint8_t>(regionIdx));
          // Decrement total since schedule will increment it again
          --totalEvents;
          schedule(eventTime, region, std::move(event));
        }
      }
    }
  }
  slot.clear();
}

bool TimeWheel::findNextEventTime(SimTime &nextTime) {
  // First check current time's slot for any events at or after current delta
  auto &currentSlot = levels[0].slots[levels[0].currentSlot];
  if (currentSlot.hasEvents) {
    // Check if there are events at or after the current delta step
    for (auto &[deltaStep, queue] : currentSlot.deltaQueues) {
      if (deltaStep >= currentTime.deltaStep && queue.hasAnyEvents()) {
        nextTime = SimTime(currentTime.realTime, deltaStep, 0);
        return true;
      }
    }
  }

  // Search through all levels and find the minimum time with events
  // We need to find the earliest event time, not just the first slot encountered
  uint64_t minTime = UINT64_MAX;
  bool found = false;

  for (size_t level = 0; level < config.numLevels; ++level) {
    auto &lvl = levels[level];
    for (size_t i = 0; i < config.slotsPerLevel; ++i) {
      auto &slot = lvl.slots[i];
      if (slot.hasEvents && slot.baseTime >= currentTime.realTime) {
        // For slots at the current real time, we already checked above
        if (slot.baseTime == currentTime.realTime)
          continue;
        if (slot.baseTime < minTime) {
          minTime = slot.baseTime;
          found = true;
        }
      }
    }
  }

  // Check overflow - it's a sorted map so begin() is the minimum
  if (!overflow.empty()) {
    auto it = overflow.begin();
    if (it->first >= currentTime.realTime && it->first < minTime) {
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
    // Advance real time
    currentTime = SimTime(nextTime.realTime, 0, 0);

    // Update slot positions
    for (size_t level = 0; level < config.numLevels; ++level) {
      levels[level].currentSlot = getSlotIndex(currentTime.realTime, level);
    }

    // Cascade events from higher levels
    for (size_t level = config.numLevels - 1; level > 0; --level) {
      cascade(level);
    }

    // Move overflow events if they're now in range
    while (!overflow.empty() && overflow.begin()->first == currentTime.realTime) {
      auto node = overflow.extract(overflow.begin());
      auto &deltaQueue = node.mapped();
      for (size_t regionIdx = 0;
           regionIdx < static_cast<size_t>(SchedulingRegion::NumRegions);
           ++regionIdx) {
        auto region = static_cast<SchedulingRegion>(regionIdx);
        auto events = deltaQueue.popRegionEvents(region);
        for (auto &event : events) {
          --totalEvents;
          schedule(SimTime(currentTime.realTime, 0, regionIdx), region,
                   std::move(event));
        }
      }
    }
  }

  return true;
}

size_t TimeWheel::processCurrentRegion() {
  size_t slotIdx = getSlotIndex(currentTime.realTime, 0);
  auto &slot = levels[0].slots[slotIdx];

  // Get the delta queue for the current delta step
  auto it = slot.deltaQueues.find(currentTime.deltaStep);
  if (it == slot.deltaQueues.end()) {
    // No events at current delta step
    slot.hasEvents = slot.hasAnyEvents();
    return 0;
  }

  auto &deltaQueue = it->second;

  // Find the next non-empty region starting from current region
  auto region = deltaQueue.getNextNonEmptyRegion(
      static_cast<SchedulingRegion>(currentTime.region));
  if (region == SchedulingRegion::NumRegions) {
    // No events to process in current or later regions
    slot.hasEvents = slot.hasAnyEvents();
    return 0;
  }

  // Update current region to match
  currentTime.region = static_cast<uint8_t>(region);

  auto events = deltaQueue.popRegionEvents(region);
  size_t count = events.size();
  totalEvents -= count;

  for (auto &event : events)
    event.execute();

  // Advance to next region
  auto nextRegion = deltaQueue.getNextNonEmptyRegion(
      static_cast<SchedulingRegion>(currentTime.region + 1));
  if (nextRegion != SchedulingRegion::NumRegions) {
    currentTime.region = static_cast<uint8_t>(nextRegion);
  } else {
    // No more events in this delta, check if we need to start a new delta
    // This will be handled by the scheduler
    slot.hasEvents = slot.hasAnyEvents();
  }

  return count;
}

size_t TimeWheel::processCurrentDelta() {
  size_t total = 0;
  size_t slotIdx = getSlotIndex(currentTime.realTime, 0);
  auto &slot = levels[0].slots[slotIdx];

  // Get the delta queue for the current delta step
  auto it = slot.deltaQueues.find(currentTime.deltaStep);
  if (it != slot.deltaQueues.end()) {
    auto &deltaQueue = it->second;

    for (size_t regionIdx = static_cast<size_t>(currentTime.region);
         regionIdx < static_cast<size_t>(SchedulingRegion::NumRegions);
         ++regionIdx) {
      auto region = static_cast<SchedulingRegion>(regionIdx);
      auto events = deltaQueue.popRegionEvents(region);
      size_t count = events.size();
      totalEvents -= count;
      total += count;

      for (auto &event : events)
        event.execute();
    }

    // Clean up empty delta queue
    if (!deltaQueue.hasAnyEvents())
      slot.deltaQueues.erase(it);
  }

  // Reset region to start of next delta
  currentTime.region = 0;
  currentTime.deltaStep++;
  slot.hasEvents = slot.hasAnyEvents();

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

bool EventScheduler::isComplete() const { return !wheel->hasEvents(); }

void EventScheduler::reset() {
  wheel->clear();
  stats = Statistics();
}
