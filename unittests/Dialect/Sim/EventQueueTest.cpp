//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/EventQueue.h"
#include "gtest/gtest.h"
#include <vector>

using namespace circt;
using namespace circt::sim;

namespace {

//===----------------------------------------------------------------------===//
// SimTime Tests
//===----------------------------------------------------------------------===//

TEST(SimTime, DefaultConstruction) {
  SimTime t;
  EXPECT_EQ(t.realTime, 0u);
  EXPECT_EQ(t.deltaStep, 0u);
  EXPECT_EQ(t.region, 0u);
}

TEST(SimTime, ParameterizedConstruction) {
  SimTime t(1000, 5, 3);
  EXPECT_EQ(t.realTime, 1000u);
  EXPECT_EQ(t.deltaStep, 5u);
  EXPECT_EQ(t.region, 3u);
}

TEST(SimTime, Comparison) {
  SimTime t1(100, 0, 0);
  SimTime t2(100, 0, 0);
  SimTime t3(200, 0, 0);
  SimTime t4(100, 1, 0);
  SimTime t5(100, 0, 1);

  // Equality
  EXPECT_TRUE(t1 == t2);
  EXPECT_FALSE(t1 == t3);

  // Less than - different real time
  EXPECT_TRUE(t1 < t3);
  EXPECT_FALSE(t3 < t1);

  // Less than - same real time, different delta
  EXPECT_TRUE(t1 < t4);
  EXPECT_FALSE(t4 < t1);

  // Less than - same real time and delta, different region
  EXPECT_TRUE(t1 < t5);
  EXPECT_FALSE(t5 < t1);
}

TEST(SimTime, NextDelta) {
  SimTime t(1000, 5, 3);
  SimTime next = t.nextDelta();

  EXPECT_EQ(next.realTime, 1000u);
  EXPECT_EQ(next.deltaStep, 6u);
  EXPECT_EQ(next.region, 0u);
}

TEST(SimTime, NextRegion) {
  SimTime t(1000, 5, 3);
  SimTime next = t.nextRegion();

  EXPECT_EQ(next.realTime, 1000u);
  EXPECT_EQ(next.deltaStep, 5u);
  EXPECT_EQ(next.region, 4u);
}

TEST(SimTime, AdvanceTime) {
  SimTime t(1000, 5, 3);
  SimTime next = t.advanceTime(500);

  EXPECT_EQ(next.realTime, 1500u);
  EXPECT_EQ(next.deltaStep, 0u);
  EXPECT_EQ(next.region, 0u);
}

//===----------------------------------------------------------------------===//
// SchedulingRegion Tests
//===----------------------------------------------------------------------===//

TEST(SchedulingRegion, RegionNames) {
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Preponed), "Preponed");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Active), "Active");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Inactive), "Inactive");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::NBA), "NBA");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Observed), "Observed");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Reactive), "Reactive");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::ReInactive),
               "ReInactive");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::ReNBA), "ReNBA");
  EXPECT_STREQ(getSchedulingRegionName(SchedulingRegion::Postponed),
               "Postponed");
}

TEST(SchedulingRegion, RegionOrdering) {
  // Verify IEEE 1800 region ordering
  EXPECT_LT(static_cast<int>(SchedulingRegion::Preponed),
            static_cast<int>(SchedulingRegion::Active));
  EXPECT_LT(static_cast<int>(SchedulingRegion::Active),
            static_cast<int>(SchedulingRegion::Inactive));
  EXPECT_LT(static_cast<int>(SchedulingRegion::Inactive),
            static_cast<int>(SchedulingRegion::NBA));
  EXPECT_LT(static_cast<int>(SchedulingRegion::NBA),
            static_cast<int>(SchedulingRegion::Observed));
  EXPECT_LT(static_cast<int>(SchedulingRegion::Observed),
            static_cast<int>(SchedulingRegion::Reactive));
  EXPECT_LT(static_cast<int>(SchedulingRegion::Reactive),
            static_cast<int>(SchedulingRegion::ReInactive));
  EXPECT_LT(static_cast<int>(SchedulingRegion::ReInactive),
            static_cast<int>(SchedulingRegion::ReNBA));
  EXPECT_LT(static_cast<int>(SchedulingRegion::ReNBA),
            static_cast<int>(SchedulingRegion::Postponed));
}

//===----------------------------------------------------------------------===//
// Event Tests
//===----------------------------------------------------------------------===//

TEST(Event, DefaultConstruction) {
  Event e;
  EXPECT_FALSE(e.isValid());
}

TEST(Event, ExecuteCallback) {
  int counter = 0;
  Event e([&counter]() { counter++; });

  EXPECT_TRUE(e.isValid());
  EXPECT_EQ(counter, 0);

  e.execute();
  EXPECT_EQ(counter, 1);

  e.execute();
  EXPECT_EQ(counter, 2);
}

//===----------------------------------------------------------------------===//
// DeltaCycleQueue Tests
//===----------------------------------------------------------------------===//

TEST(DeltaCycleQueue, EmptyQueue) {
  DeltaCycleQueue queue;

  EXPECT_FALSE(queue.hasAnyEvents());
  EXPECT_FALSE(queue.hasEvents(SchedulingRegion::Active));
  EXPECT_EQ(queue.getEventCount(SchedulingRegion::Active), 0u);
}

TEST(DeltaCycleQueue, ScheduleAndPop) {
  DeltaCycleQueue queue;
  int counter = 0;

  queue.schedule(SchedulingRegion::Active, Event([&counter]() { counter++; }));

  EXPECT_TRUE(queue.hasAnyEvents());
  EXPECT_TRUE(queue.hasEvents(SchedulingRegion::Active));
  EXPECT_EQ(queue.getEventCount(SchedulingRegion::Active), 1u);

  auto events = queue.popRegionEvents(SchedulingRegion::Active);
  EXPECT_EQ(events.size(), 1u);
  EXPECT_FALSE(queue.hasEvents(SchedulingRegion::Active));

  events[0].execute();
  EXPECT_EQ(counter, 1);
}

TEST(DeltaCycleQueue, MultipleRegions) {
  DeltaCycleQueue queue;

  queue.schedule(SchedulingRegion::Active, Event([]() {}));
  queue.schedule(SchedulingRegion::NBA, Event([]() {}));
  queue.schedule(SchedulingRegion::NBA, Event([]() {}));
  queue.schedule(SchedulingRegion::Postponed, Event([]() {}));

  EXPECT_EQ(queue.getEventCount(SchedulingRegion::Active), 1u);
  EXPECT_EQ(queue.getEventCount(SchedulingRegion::NBA), 2u);
  EXPECT_EQ(queue.getEventCount(SchedulingRegion::Postponed), 1u);
  EXPECT_EQ(queue.getEventCount(SchedulingRegion::Inactive), 0u);
}

TEST(DeltaCycleQueue, GetNextNonEmptyRegion) {
  DeltaCycleQueue queue;

  // Empty queue
  EXPECT_EQ(queue.getNextNonEmptyRegion(SchedulingRegion::Preponed),
            SchedulingRegion::NumRegions);

  // Add to NBA only
  queue.schedule(SchedulingRegion::NBA, Event([]() {}));

  EXPECT_EQ(queue.getNextNonEmptyRegion(SchedulingRegion::Preponed),
            SchedulingRegion::NBA);
  EXPECT_EQ(queue.getNextNonEmptyRegion(SchedulingRegion::Active),
            SchedulingRegion::NBA);
  EXPECT_EQ(queue.getNextNonEmptyRegion(SchedulingRegion::NBA),
            SchedulingRegion::NBA);
  EXPECT_EQ(queue.getNextNonEmptyRegion(SchedulingRegion::Observed),
            SchedulingRegion::NumRegions);
}

TEST(DeltaCycleQueue, Clear) {
  DeltaCycleQueue queue;

  queue.schedule(SchedulingRegion::Active, Event([]() {}));
  queue.schedule(SchedulingRegion::NBA, Event([]() {}));

  EXPECT_TRUE(queue.hasAnyEvents());

  queue.clear();

  EXPECT_FALSE(queue.hasAnyEvents());
}

//===----------------------------------------------------------------------===//
// TimeWheel Tests
//===----------------------------------------------------------------------===//

TEST(TimeWheel, EmptyWheel) {
  TimeWheel wheel;

  EXPECT_FALSE(wheel.hasEvents());
  EXPECT_EQ(wheel.getEventCount(), 0u);
  EXPECT_EQ(wheel.getCurrentTime().realTime, 0u);
}

TEST(TimeWheel, ScheduleAndProcess) {
  TimeWheel wheel;
  int counter = 0;

  wheel.schedule(SimTime(0), Event([&counter]() { counter++; }));

  EXPECT_TRUE(wheel.hasEvents());
  EXPECT_EQ(wheel.getEventCount(), 1u);

  size_t processed = wheel.processCurrentRegion();
  EXPECT_EQ(processed, 1u);
  EXPECT_EQ(counter, 1);
  EXPECT_FALSE(wheel.hasEvents());
}

TEST(TimeWheel, ScheduleFutureEvent) {
  TimeWheel wheel;
  int counter = 0;

  // Schedule event 100 fs in the future
  wheel.schedule(SimTime(100), Event([&counter]() { counter++; }));

  EXPECT_TRUE(wheel.hasEvents());

  // Processing current time should do nothing
  size_t processed = wheel.processCurrentTime();
  EXPECT_EQ(processed, 0u);
  EXPECT_EQ(counter, 0);

  // Advance to next event
  EXPECT_TRUE(wheel.advanceToNextEvent());
  EXPECT_EQ(wheel.getCurrentTime().realTime, 100u);

  // Now process
  processed = wheel.processCurrentTime();
  EXPECT_EQ(processed, 1u);
  EXPECT_EQ(counter, 1);
}

TEST(TimeWheel, MultipleEventsAtSameTime) {
  TimeWheel wheel;
  std::vector<int> order;

  wheel.schedule(SimTime(100), SchedulingRegion::Active,
                 Event([&order]() { order.push_back(1); }));
  wheel.schedule(SimTime(100), SchedulingRegion::Active,
                 Event([&order]() { order.push_back(2); }));
  wheel.schedule(SimTime(100), SchedulingRegion::Active,
                 Event([&order]() { order.push_back(3); }));

  EXPECT_EQ(wheel.getEventCount(), 3u);

  wheel.advanceToNextEvent();
  wheel.processCurrentTime();

  EXPECT_EQ(order.size(), 3u);
  // Events should execute in order they were scheduled
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
  EXPECT_EQ(order[2], 3);
}

TEST(TimeWheel, RegionOrdering) {
  TimeWheel wheel;
  std::vector<std::string> order;

  // Schedule events in reverse region order
  wheel.schedule(SimTime(0), SchedulingRegion::Postponed,
                 Event([&order]() { order.push_back("Postponed"); }));
  wheel.schedule(SimTime(0), SchedulingRegion::NBA,
                 Event([&order]() { order.push_back("NBA"); }));
  wheel.schedule(SimTime(0), SchedulingRegion::Active,
                 Event([&order]() { order.push_back("Active"); }));
  wheel.schedule(SimTime(0), SchedulingRegion::Preponed,
                 Event([&order]() { order.push_back("Preponed"); }));

  // Process one delta which should process all regions
  wheel.processCurrentDelta();

  // Verify events were processed in region order
  ASSERT_EQ(order.size(), 4u);
  EXPECT_EQ(order[0], "Preponed");
  EXPECT_EQ(order[1], "Active");
  EXPECT_EQ(order[2], "NBA");
  EXPECT_EQ(order[3], "Postponed");
}

TEST(TimeWheel, Clear) {
  TimeWheel wheel;

  wheel.schedule(SimTime(100), Event([]() {}));
  wheel.schedule(SimTime(200), Event([]() {}));

  EXPECT_TRUE(wheel.hasEvents());

  wheel.clear();

  EXPECT_FALSE(wheel.hasEvents());
  EXPECT_EQ(wheel.getCurrentTime().realTime, 0u);
}

//===----------------------------------------------------------------------===//
// EventScheduler Tests
//===----------------------------------------------------------------------===//

TEST(EventScheduler, BasicScheduling) {
  EventScheduler scheduler;
  int counter = 0;

  scheduler.scheduleNow(SchedulingRegion::Active,
                        Event([&counter]() { counter++; }));

  EXPECT_FALSE(scheduler.isComplete());
  EXPECT_TRUE(scheduler.stepDelta());
  EXPECT_EQ(counter, 1);
}

TEST(EventScheduler, DelayedScheduling) {
  EventScheduler scheduler;
  int counter = 0;

  scheduler.scheduleDelay(1000, SchedulingRegion::Active,
                          Event([&counter]() { counter++; }));

  EXPECT_FALSE(scheduler.isComplete());
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);

  scheduler.runUntil(2000);

  EXPECT_EQ(counter, 1);
  EXPECT_GE(scheduler.getCurrentTime().realTime, 1000u);
}

TEST(EventScheduler, NextDeltaScheduling) {
  EventScheduler scheduler;
  int value = 0;

  // Schedule event that schedules another event in next delta
  scheduler.scheduleNow(SchedulingRegion::Active, Event([&scheduler, &value]() {
    value = 1;
    scheduler.scheduleNextDelta(SchedulingRegion::Active,
                                Event([&value]() { value = 2; }));
  }));

  // First delta
  scheduler.stepDelta();
  EXPECT_EQ(value, 1);
  EXPECT_EQ(scheduler.getCurrentTime().deltaStep, 1u);

  // Second delta
  scheduler.stepDelta();
  EXPECT_EQ(value, 2);
}

TEST(EventScheduler, Statistics) {
  EventScheduler scheduler;

  scheduler.scheduleNow(SchedulingRegion::Active, Event([]() {}));
  scheduler.scheduleNow(SchedulingRegion::Active, Event([]() {}));
  scheduler.scheduleDelay(100, SchedulingRegion::Active, Event([]() {}));

  scheduler.runUntil(1000);

  auto stats = scheduler.getStatistics();
  EXPECT_EQ(stats.eventsProcessed, 3u);
  EXPECT_GE(stats.deltaCycles, 2u);
  EXPECT_GE(stats.realTimeAdvances, 1u);
}

TEST(EventScheduler, Reset) {
  EventScheduler scheduler;

  scheduler.scheduleDelay(100, SchedulingRegion::Active, Event([]() {}));
  scheduler.runUntil(50);

  scheduler.reset();

  EXPECT_TRUE(scheduler.isComplete());
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);
  EXPECT_EQ(scheduler.getStatistics().eventsProcessed, 0u);
}

TEST(EventScheduler, RunDeltas) {
  EventScheduler scheduler;
  int counter = 0;

  // Create chain of delta events
  scheduler.scheduleNow(SchedulingRegion::Active, Event([&scheduler, &counter]() {
    counter++;
    if (counter < 5) {
      scheduler.scheduleNextDelta(SchedulingRegion::Active,
                                  Event([&scheduler, &counter]() {
                                    counter++;
                                    if (counter < 5) {
                                      scheduler.scheduleNextDelta(
                                          SchedulingRegion::Active,
                                          Event([&counter]() { counter++; }));
                                    }
                                  }));
    }
  }));

  // Run only 2 deltas
  size_t deltas = scheduler.runDeltas(2);

  EXPECT_EQ(deltas, 2u);
  EXPECT_EQ(counter, 2);
}

} // namespace
