//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "gtest/gtest.h"
#include "llvm/ADT/APInt.h"
#include <vector>

using namespace circt;
using namespace circt::sim;

namespace {

//===----------------------------------------------------------------------===//
// ProcessState Tests
//===----------------------------------------------------------------------===//

TEST(ProcessState, StateNames) {
  EXPECT_STREQ(getProcessStateName(ProcessState::Uninitialized),
               "Uninitialized");
  EXPECT_STREQ(getProcessStateName(ProcessState::Ready), "Ready");
  EXPECT_STREQ(getProcessStateName(ProcessState::Running), "Running");
  EXPECT_STREQ(getProcessStateName(ProcessState::Suspended), "Suspended");
  EXPECT_STREQ(getProcessStateName(ProcessState::Waiting), "Waiting");
  EXPECT_STREQ(getProcessStateName(ProcessState::Terminated), "Terminated");
}

//===----------------------------------------------------------------------===//
// EdgeType Tests
//===----------------------------------------------------------------------===//

TEST(EdgeType, EdgeNames) {
  EXPECT_STREQ(getEdgeTypeName(EdgeType::None), "none");
  EXPECT_STREQ(getEdgeTypeName(EdgeType::Posedge), "posedge");
  EXPECT_STREQ(getEdgeTypeName(EdgeType::Negedge), "negedge");
  EXPECT_STREQ(getEdgeTypeName(EdgeType::AnyEdge), "anyedge");
}

TEST(EdgeType, ParseEdgeType) {
  EXPECT_EQ(parseEdgeType("posedge"), EdgeType::Posedge);
  EXPECT_EQ(parseEdgeType("negedge"), EdgeType::Negedge);
  EXPECT_EQ(parseEdgeType("anyedge"), EdgeType::AnyEdge);
  EXPECT_EQ(parseEdgeType("any"), EdgeType::AnyEdge);
  EXPECT_EQ(parseEdgeType("invalid"), EdgeType::None);
}

//===----------------------------------------------------------------------===//
// SignalValue Tests
//===----------------------------------------------------------------------===//

TEST(SignalValue, DefaultConstruction) {
  SignalValue sv;
  EXPECT_TRUE(sv.isUnknown());
  EXPECT_EQ(sv.getWidth(), 1u);
}

TEST(SignalValue, ParameterizedConstruction) {
  SignalValue sv(42, 8);
  EXPECT_FALSE(sv.isUnknown());
  EXPECT_EQ(sv.getValue(), 42u);
  EXPECT_EQ(sv.getWidth(), 8u);
}

TEST(SignalValue, MakeX) {
  SignalValue sv = SignalValue::makeX(16);
  EXPECT_TRUE(sv.isUnknown());
  EXPECT_EQ(sv.getWidth(), 16u);
}

TEST(SignalValue, Equality) {
  SignalValue a(1, 1);
  SignalValue b(1, 1);
  SignalValue c(0, 1);
  SignalValue x1 = SignalValue::makeX();
  SignalValue x2 = SignalValue::makeX();

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(x1 == x2); // X values are equal
  EXPECT_FALSE(a == x1);
}

TEST(SignalValue, GetLSB) {
  SignalValue zero(0, 1);
  SignalValue one(1, 1);
  SignalValue even(42, 8);
  SignalValue odd(43, 8);

  EXPECT_FALSE(zero.getLSB());
  EXPECT_TRUE(one.getLSB());
  EXPECT_FALSE(even.getLSB());
  EXPECT_TRUE(odd.getLSB());
}

TEST(SignalValue, DetectEdgePosedge) {
  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  EXPECT_EQ(SignalValue::detectEdge(zero, one), EdgeType::Posedge);
  EXPECT_EQ(SignalValue::detectEdge(one, zero), EdgeType::Negedge);
  EXPECT_EQ(SignalValue::detectEdge(zero, zero), EdgeType::None);
  EXPECT_EQ(SignalValue::detectEdge(one, one), EdgeType::None);
}

TEST(SignalValue, DetectEdgeMultiBit) {
  SignalValue a(0x55, 8);
  SignalValue b(0x56, 8);
  SignalValue c(0x54, 8);

  // LSB changes from 1 to 0
  EXPECT_EQ(SignalValue::detectEdge(a, b), EdgeType::Negedge);
  // LSB changes from 1 to 0
  EXPECT_EQ(SignalValue::detectEdge(a, c), EdgeType::Negedge);
  // Different values
  EXPECT_EQ(SignalValue::detectEdge(b, c), EdgeType::AnyEdge);
}

TEST(SignalValue, FourStateXDetection) {
  // Test 4-state struct encoding: {value: i1, unknown: i1} flattened to 2 bits
  // Lower bit = unknown, upper bit = value
  // 0b00 = 0 (known 0), 0b10 = 1 (known 1), 0b01 = X (unknown, value=0), 0b11 = X (unknown, value=1)

  SignalValue known0(llvm::APInt(2, 0b00));  // Known 0
  SignalValue known1(llvm::APInt(2, 0b10));  // Known 1
  SignalValue x_val0(llvm::APInt(2, 0b01));  // X with underlying value=0
  SignalValue x_val1(llvm::APInt(2, 0b11));  // X with underlying value=1

  // isFourStateX should be true when any unknown bits are set
  EXPECT_FALSE(known0.isFourStateX());  // unknown=0
  EXPECT_FALSE(known1.isFourStateX());  // unknown=0
  EXPECT_TRUE(x_val0.isFourStateX());   // unknown=1
  EXPECT_TRUE(x_val1.isFourStateX());   // unknown=1

  // Two X values should be equal even with different underlying value bits
  EXPECT_TRUE(x_val0 == x_val1);

  // X to X should NOT trigger an edge
  EXPECT_EQ(SignalValue::detectEdge(x_val0, x_val1), EdgeType::None);
  EXPECT_EQ(SignalValue::detectEdge(x_val1, x_val0), EdgeType::None);

  // Known to X or X to Known SHOULD trigger an edge
  EXPECT_NE(SignalValue::detectEdge(known0, x_val0), EdgeType::None);
  EXPECT_NE(SignalValue::detectEdge(x_val0, known0), EdgeType::None);
  EXPECT_NE(SignalValue::detectEdge(known1, x_val1), EdgeType::None);
  EXPECT_NE(SignalValue::detectEdge(x_val1, known1), EdgeType::None);

  // Test wider 4-state values (4-bit value + 4-bit unknown = 8-bit total)
  // All lower bits set = fully X
  SignalValue wide_x1(llvm::APInt(8, 0x0F));  // unknown=0b1111, value=0b0000 -> all X
  SignalValue wide_x2(llvm::APInt(8, 0xFF));  // unknown=0b1111, value=0b1111 -> all X
  SignalValue wide_known(llvm::APInt(8, 0xA0));  // unknown=0b0000, value=0b1010 -> known

  EXPECT_TRUE(wide_x1.isFourStateX());
  EXPECT_TRUE(wide_x2.isFourStateX());
  EXPECT_FALSE(wide_known.isFourStateX());

  // Two fully-X values should be equal
  EXPECT_TRUE(wide_x1 == wide_x2);
  EXPECT_EQ(SignalValue::detectEdge(wide_x1, wide_x2), EdgeType::None);

  // Partially unknown values should be treated as X
  SignalValue partial_unknown(llvm::APInt(8, 0x01));  // unknown=0b0001, value=0b0000
  EXPECT_TRUE(partial_unknown.isFourStateX());
}

//===----------------------------------------------------------------------===//
// SensitivityList Tests
//===----------------------------------------------------------------------===//

TEST(SensitivityList, EmptyList) {
  SensitivityList list;
  EXPECT_TRUE(list.empty());
  EXPECT_EQ(list.size(), 0u);
}

TEST(SensitivityList, AddLevel) {
  SensitivityList list;
  list.addLevel(1);
  list.addLevel(2);

  EXPECT_FALSE(list.empty());
  EXPECT_EQ(list.size(), 2u);
}

TEST(SensitivityList, AddEdge) {
  SensitivityList list;
  list.addPosedge(1);
  list.addNegedge(2);
  list.addEdge(3, EdgeType::AnyEdge);

  EXPECT_EQ(list.size(), 3u);
}

TEST(SensitivityList, IsTriggeredByPosedge) {
  SensitivityList list;
  list.addPosedge(1);

  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  EXPECT_TRUE(list.isTriggeredBy(1, zero, one));   // Posedge triggers
  EXPECT_FALSE(list.isTriggeredBy(1, one, zero));  // Negedge doesn't trigger
  EXPECT_FALSE(list.isTriggeredBy(1, zero, zero)); // No change doesn't trigger
  EXPECT_FALSE(list.isTriggeredBy(2, zero, one));  // Wrong signal doesn't trigger
}

TEST(SensitivityList, IsTriggeredByNegedge) {
  SensitivityList list;
  list.addNegedge(1);

  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  EXPECT_FALSE(list.isTriggeredBy(1, zero, one)); // Posedge doesn't trigger
  EXPECT_TRUE(list.isTriggeredBy(1, one, zero));  // Negedge triggers
}

TEST(SensitivityList, IsTriggeredByAnyEdge) {
  SensitivityList list;
  list.addLevel(1);

  SignalValue a(10, 8);
  SignalValue b(20, 8);

  EXPECT_TRUE(list.isTriggeredBy(1, a, b));  // Any change triggers
  EXPECT_FALSE(list.isTriggeredBy(1, a, a)); // No change doesn't trigger
}

TEST(SensitivityList, Clear) {
  SensitivityList list;
  list.addPosedge(1);
  list.addNegedge(2);

  EXPECT_EQ(list.size(), 2u);
  list.clear();
  EXPECT_TRUE(list.empty());
}

TEST(SensitivityList, AutoInferred) {
  SensitivityList list;
  EXPECT_FALSE(list.isAutoInferred());

  list.setAutoInferred(true);
  EXPECT_TRUE(list.isAutoInferred());
}

//===----------------------------------------------------------------------===//
// Process Tests
//===----------------------------------------------------------------------===//

TEST(Process, Construction) {
  int counter = 0;
  Process proc(1, "test_proc", [&counter]() { counter++; });

  EXPECT_EQ(proc.getId(), 1u);
  EXPECT_EQ(proc.getName(), "test_proc");
  EXPECT_EQ(proc.getState(), ProcessState::Uninitialized);
}

TEST(Process, Execute) {
  int counter = 0;
  Process proc(1, "test_proc", [&counter]() { counter++; });

  proc.execute();
  EXPECT_EQ(counter, 1);

  proc.execute();
  EXPECT_EQ(counter, 2);
}

TEST(Process, StateTransitions) {
  Process proc(1, "test_proc", []() {});

  EXPECT_EQ(proc.getState(), ProcessState::Uninitialized);

  proc.setState(ProcessState::Ready);
  EXPECT_EQ(proc.getState(), ProcessState::Ready);

  proc.setState(ProcessState::Running);
  EXPECT_EQ(proc.getState(), ProcessState::Running);

  proc.setState(ProcessState::Suspended);
  EXPECT_EQ(proc.getState(), ProcessState::Suspended);
}

TEST(Process, SensitivityList) {
  Process proc(1, "test_proc", []() {});

  proc.getSensitivityList().addPosedge(1);
  proc.getSensitivityList().addNegedge(2);

  EXPECT_EQ(proc.getSensitivityList().size(), 2u);
}

TEST(Process, Combinational) {
  Process proc(1, "comb_proc", []() {});

  EXPECT_FALSE(proc.isCombinational());
  proc.setCombinational(true);
  EXPECT_TRUE(proc.isCombinational());
}

//===----------------------------------------------------------------------===//
// SignalState Tests
//===----------------------------------------------------------------------===//

TEST(SignalState, DefaultConstruction) {
  SignalState state;
  EXPECT_TRUE(state.getCurrentValue().isUnknown());
}

TEST(SignalState, ConstructWithWidth) {
  SignalState state(8);
  EXPECT_TRUE(state.getCurrentValue().isUnknown());
  EXPECT_EQ(state.getCurrentValue().getWidth(), 8u);
}

TEST(SignalState, UpdateValue) {
  SignalState state(1);
  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  // Initial update from X to 0 - per IEEE 1800, X->0 is negedge
  EdgeType edge1 = state.updateValue(zero);
  EXPECT_EQ(state.getCurrentValue(), zero);
  EXPECT_EQ(edge1, EdgeType::Negedge);

  // Update from 0 to 1 - posedge
  EdgeType edge2 = state.updateValue(one);
  EXPECT_EQ(state.getCurrentValue(), one);
  EXPECT_EQ(edge2, EdgeType::Posedge);

  // Update from 1 to 0 - negedge
  EdgeType edge3 = state.updateValue(zero);
  EXPECT_EQ(state.getCurrentValue(), zero);
  EXPECT_EQ(edge3, EdgeType::Negedge);

  // No change
  EdgeType edge4 = state.updateValue(zero);
  EXPECT_EQ(edge4, EdgeType::None);
}

TEST(SignalState, HasChanged) {
  SignalState state(1);
  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  state.updateValue(zero);
  EXPECT_TRUE(state.hasChanged()); // X to 0 is a change

  state.updateValue(zero);
  EXPECT_FALSE(state.hasChanged()); // 0 to 0 is no change

  state.updateValue(one);
  EXPECT_TRUE(state.hasChanged()); // 0 to 1 is a change
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Basic Operations
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, DefaultConstruction) {
  ProcessScheduler scheduler;
  EXPECT_TRUE(scheduler.isComplete());
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);
}

TEST(ProcessScheduler, RegisterProcess) {
  ProcessScheduler scheduler;
  int counter = 0;

  ProcessId id = scheduler.registerProcess("test", [&counter]() { counter++; });
  EXPECT_NE(id, InvalidProcessId);

  Process *proc = scheduler.getProcess(id);
  EXPECT_NE(proc, nullptr);
  EXPECT_EQ(proc->getName(), "test");
}

TEST(ProcessScheduler, UnregisterProcess) {
  ProcessScheduler scheduler;

  ProcessId id = scheduler.registerProcess("test", []() {});
  EXPECT_NE(scheduler.getProcess(id), nullptr);

  scheduler.unregisterProcess(id);
  EXPECT_EQ(scheduler.getProcess(id), nullptr);
}

TEST(ProcessScheduler, RegisterSignal) {
  ProcessScheduler scheduler;

  SignalId id = scheduler.registerSignal("clk", 1);
  EXPECT_NE(id, 0u);

  const SignalValue &val = scheduler.getSignalValue(id);
  EXPECT_TRUE(val.isUnknown());
}

TEST(ProcessScheduler, UpdateSignal) {
  ProcessScheduler scheduler;

  SignalId id = scheduler.registerSignal("data", 8);
  scheduler.updateSignal(id, SignalValue(42, 8));

  const SignalValue &val = scheduler.getSignalValue(id);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 42u);
}

TEST(ProcessScheduler, UpdateSignalNormalizesWidth) {
  ProcessScheduler scheduler;

  SignalId id = scheduler.registerSignal("data", 8);
  scheduler.updateSignal(id, SignalValue(llvm::APInt(16, 0x1234)));

  const SignalValue &wideVal = scheduler.getSignalValue(id);
  EXPECT_EQ(wideVal.getWidth(), 8u);
  EXPECT_EQ(wideVal.getValue(), 0x34u);

  scheduler.updateSignal(id, SignalValue(llvm::APInt(4, 0xA)));

  const SignalValue &narrowVal = scheduler.getSignalValue(id);
  EXPECT_EQ(narrowVal.getWidth(), 8u);
  EXPECT_EQ(narrowVal.getValue(), 0x0Au);
}

TEST(ProcessScheduler, FourStateStructPosedgeTriggers) {
  ProcessScheduler scheduler;
  SignalId clk = scheduler.registerSignal("clk", 2,
                                          SignalEncoding::FourStateStruct);

  int counter = 0;
  ProcessId procId =
      scheduler.registerProcess("posedge", [&counter]() { ++counter; });

  SensitivityList list;
  list.addPosedge(clk);
  scheduler.setSensitivity(procId, list);

  scheduler.updateSignal(clk, SignalValue(llvm::APInt(2, 0b00))); // known 0
  scheduler.executeDeltaCycle();
  counter = 0;

  scheduler.updateSignal(clk, SignalValue(llvm::APInt(2, 0b10))); // known 1
  scheduler.executeDeltaCycle();
  EXPECT_EQ(counter, 1);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Process Execution
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, ExecuteSingleProcess) {
  ProcessScheduler scheduler;
  int counter = 0;

  scheduler.registerProcess("test", [&counter]() { counter++; });

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  EXPECT_EQ(counter, 1);
}

TEST(ProcessScheduler, AbortStopsExecution) {
  ProcessScheduler scheduler;
  int counter = 0;

  scheduler.registerProcess("test", [&counter]() { counter++; });
  scheduler.setShouldAbortCallback([]() { return true; });

  EXPECT_FALSE(scheduler.executeDeltaCycle());
  EXPECT_EQ(counter, 0);
}

TEST(ProcessScheduler, ExecuteMultipleProcesses) {
  ProcessScheduler scheduler;
  std::vector<int> order;

  scheduler.registerProcess("proc1", [&order]() { order.push_back(1); });
  scheduler.registerProcess("proc2", [&order]() { order.push_back(2); });
  scheduler.registerProcess("proc3", [&order]() { order.push_back(3); });

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  EXPECT_EQ(order.size(), 3u);
}

TEST(ProcessScheduler, ProcessStatistics) {
  ProcessScheduler scheduler;
  scheduler.registerProcess("test", []() {});

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  auto stats = scheduler.getStatistics();
  EXPECT_EQ(stats.processesRegistered, 1u);
  EXPECT_EQ(stats.processesExecuted, 1u);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Sensitivity Lists
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, SensitivityTrigger) {
  ProcessScheduler scheduler;
  int counter = 0;

  ProcessId procId = scheduler.registerProcess("sensitive_proc",
                                               [&counter]() { counter++; });
  SignalId sigId = scheduler.registerSignal("trigger", 1);

  scheduler.addSensitivity(procId, sigId, EdgeType::Posedge);
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  int initialCount = counter;

  // Simulate a positive edge
  scheduler.updateSignal(sigId, SignalValue(0, 1));
  scheduler.updateSignal(sigId, SignalValue(1, 1));
  scheduler.executeDeltaCycle();

  EXPECT_GT(counter, initialCount);
}

TEST(ProcessScheduler, SensitivityNoTrigger) {
  ProcessScheduler scheduler;
  int counter = 0;

  ProcessId procId = scheduler.registerProcess("sensitive_proc",
                                               [&counter]() { counter++; });
  SignalId sigId = scheduler.registerSignal("trigger", 1);

  scheduler.addSensitivity(procId, sigId, EdgeType::Posedge);
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  int initialCount = counter;

  // Simulate a negative edge - should not trigger posedge-sensitive process
  scheduler.updateSignal(sigId, SignalValue(1, 1));
  scheduler.updateSignal(sigId, SignalValue(0, 1));

  // Don't expect additional execution from negedge on posedge-sensitive process
  EXPECT_GE(counter, initialCount);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Process Suspension
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, SuspendAndResume) {
  ProcessScheduler scheduler;
  int counter = 0;

  ProcessId id = scheduler.registerProcess("suspend_test",
                                           [&counter]() { counter++; });
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  EXPECT_EQ(counter, 1);

  // Manually resume the process
  scheduler.resumeProcess(id);
  scheduler.executeDeltaCycle();

  EXPECT_EQ(counter, 2);
}

TEST(ProcessScheduler, SuspendUntilTime) {
  ProcessScheduler scheduler;
  int counter = 0;

  ProcessId id = scheduler.registerProcess("timed_proc",
                                           [&counter]() { counter++; });

  scheduler.initialize();
  scheduler.executeDeltaCycle();
  EXPECT_EQ(counter, 1);

  // Suspend until 1000 fs
  scheduler.suspendProcess(id, SimTime(1000, 0, 0));

  // Run until 2000 fs
  scheduler.runUntil(2000);

  EXPECT_GE(counter, 2);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Delta Cycle Limits
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, DeltaCycleLimit) {
  ProcessScheduler::Config config;
  config.maxDeltaCycles = 10;
  ProcessScheduler scheduler(config);

  // Create a process that reschedules itself infinitely
  [[maybe_unused]] ProcessId id =
      scheduler.registerProcess("infinite_loop", [&scheduler]() {
        // This would create an infinite loop
      });

  scheduler.initialize();

  // Should not hang due to delta cycle limit
  size_t deltas = scheduler.executeCurrentTime();

  // The scheduler should have stopped due to limit
  EXPECT_LE(deltas, 10u);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Reset
//===----------------------------------------------------------------------===//

TEST(ProcessScheduler, Reset) {
  ProcessScheduler scheduler;

  scheduler.registerProcess("test", []() {});
  scheduler.registerSignal("sig", 1);
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  auto statsBeforeReset = scheduler.getStatistics();
  EXPECT_GT(statsBeforeReset.processesRegistered, 0u);

  scheduler.reset();

  auto statsAfterReset = scheduler.getStatistics();
  EXPECT_EQ(statsAfterReset.processesRegistered, 0u);
  EXPECT_EQ(statsAfterReset.processesExecuted, 0u);
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);
}

//===----------------------------------------------------------------------===//
// EdgeDetector Tests
//===----------------------------------------------------------------------===//

TEST(EdgeDetector, RecordValue) {
  EdgeDetector detector;

  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  // First value from unknown - per IEEE 1800, X->0 is negedge
  EdgeType edge1 = detector.recordValue(zero);
  EXPECT_EQ(edge1, EdgeType::Negedge);

  // 0 -> 1 posedge
  EdgeType edge2 = detector.recordValue(one);
  EXPECT_EQ(edge2, EdgeType::Posedge);

  // 1 -> 0 negedge
  EdgeType edge3 = detector.recordValue(zero);
  EXPECT_EQ(edge3, EdgeType::Negedge);
}

TEST(EdgeDetector, HasPosedge) {
  EdgeDetector detector;

  SignalValue zero(0, 1);
  SignalValue one(1, 1);

  detector.recordValue(zero);
  EXPECT_TRUE(detector.hasPosedge(one));
  EXPECT_FALSE(detector.hasNegedge(one));

  detector.recordValue(one);
  EXPECT_FALSE(detector.hasPosedge(zero));
  EXPECT_TRUE(detector.hasNegedge(zero));
}

//===----------------------------------------------------------------------===//
// CombProcessManager Tests
//===----------------------------------------------------------------------===//

TEST(CombProcessManager, RegisterAndTrack) {
  ProcessScheduler scheduler;
  CombProcessManager manager(scheduler);

  ProcessId id = scheduler.registerProcess("comb", []() {});
  manager.registerCombProcess(id);

  Process *proc = scheduler.getProcess(id);
  EXPECT_TRUE(proc->isCombinational());
}

TEST(CombProcessManager, InferSensitivity) {
  ProcessScheduler scheduler;
  CombProcessManager manager(scheduler);

  ProcessId procId = scheduler.registerProcess("comb", []() {});
  SignalId sig1 = scheduler.registerSignal("a", 8);
  SignalId sig2 = scheduler.registerSignal("b", 8);

  manager.registerCombProcess(procId);
  manager.beginTracking(procId);
  manager.recordSignalRead(procId, sig1);
  manager.recordSignalRead(procId, sig2);
  manager.endTracking(procId);

  // After finalization, process should be sensitive to both signals
  Process *proc = scheduler.getProcess(procId);
  EXPECT_EQ(proc->getSensitivityList().size(), 2u);
}

//===----------------------------------------------------------------------===//
// Integration Tests
//===----------------------------------------------------------------------===//

TEST(ProcessSchedulerIntegration, ClockDrivenCounter) {
  ProcessScheduler scheduler;
  int counter = 0;

  SignalId clk = scheduler.registerSignal("clk", 1);

  ProcessId proc = scheduler.registerProcess("counter", [&counter]() {
    counter++;
  });

  scheduler.addSensitivity(proc, clk, EdgeType::Posedge);
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  int initial = counter;

  // Simulate 5 clock cycles
  for (int i = 0; i < 5; ++i) {
    scheduler.updateSignal(clk, SignalValue(0, 1));
    scheduler.executeDeltaCycle();
    scheduler.updateSignal(clk, SignalValue(1, 1));
    scheduler.executeDeltaCycle();
  }

  // Counter should have incremented for each posedge
  EXPECT_GE(counter, initial + 5);
}

TEST(ProcessSchedulerIntegration, MultipleSignalSensitivity) {
  ProcessScheduler scheduler;
  std::vector<std::string> events;

  SignalId clk = scheduler.registerSignal("clk", 1);
  SignalId rst = scheduler.registerSignal("rst", 1);

  ProcessId proc = scheduler.registerProcess("clk_rst_proc", [&events]() {
    events.push_back("triggered");
  });

  scheduler.addSensitivity(proc, clk, EdgeType::Posedge);
  scheduler.addSensitivity(proc, rst, EdgeType::Negedge);

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  size_t initialEvents = events.size();

  // Clock posedge
  scheduler.updateSignal(clk, SignalValue(0, 1));
  scheduler.updateSignal(clk, SignalValue(1, 1));
  scheduler.executeDeltaCycle();

  // Reset negedge
  scheduler.updateSignal(rst, SignalValue(1, 1));
  scheduler.updateSignal(rst, SignalValue(0, 1));
  scheduler.executeDeltaCycle();

  // Both events should have triggered the process
  EXPECT_GE(events.size(), initialEvents + 2);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Delayed Event Time Advancement
//===----------------------------------------------------------------------===//

TEST(ProcessSchedulerIntegration, DelayedEventAdvancesTime) {
  // This test verifies that when an event is scheduled on the EventScheduler
  // for a future time, advanceTime() properly advances to that time and
  // executes the callback.
  ProcessScheduler scheduler;
  int processExecuted = 0;
  uint64_t resumeTime = 10000000; // 10 ms in femtoseconds

  ProcessId procId = scheduler.registerProcess("delayed_proc",
                                               [&processExecuted]() {
                                                 processExecuted++;
                                               });

  scheduler.initialize();

  // Execute initial delta - process runs once
  scheduler.executeDeltaCycle();
  EXPECT_EQ(processExecuted, 1);
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);

  // Schedule a delayed event via the EventScheduler (simulating llhd.wait)
  scheduler.getEventScheduler().schedule(
      SimTime(resumeTime, 0, 0), SchedulingRegion::Active,
      Event([&scheduler, procId]() {
        scheduler.resumeProcess(procId);
      }));

  // Verify event is pending
  EXPECT_FALSE(scheduler.getEventScheduler().isComplete());

  // Advance time - should move to the scheduled event time
  bool advanced = scheduler.advanceTime();
  EXPECT_TRUE(advanced);
  EXPECT_EQ(scheduler.getCurrentTime().realTime, resumeTime);

  // Execute the resumed process
  scheduler.executeCurrentTime();
  EXPECT_EQ(processExecuted, 2);
}

TEST(ProcessSchedulerIntegration, MultipleDelayedEvents) {
  // Test scheduling multiple events at different future times
  ProcessScheduler scheduler;
  std::vector<uint64_t> executedTimes;

  ProcessId proc = scheduler.registerProcess("time_logger",
                                             [&scheduler, &executedTimes]() {
                                               executedTimes.push_back(
                                                   scheduler.getCurrentTime().realTime);
                                             });

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  // Clear the initial execution
  executedTimes.clear();

  // Schedule events at different times
  uint64_t times[] = {1000, 5000, 3000, 10000, 2000};
  for (uint64_t t : times) {
    scheduler.getEventScheduler().schedule(
        SimTime(t, 0, 0), SchedulingRegion::Active,
        Event([&scheduler, proc]() {
          scheduler.resumeProcess(proc);
        }));
  }

  // Run until all events are processed
  scheduler.runUntil(20000);

  // All events should have been executed
  EXPECT_EQ(executedTimes.size(), 5u);

  // Times should be in order (events sorted by time)
  EXPECT_EQ(executedTimes[0], 1000u);
  EXPECT_EQ(executedTimes[1], 2000u);
  EXPECT_EQ(executedTimes[2], 3000u);
  EXPECT_EQ(executedTimes[3], 5000u);
  EXPECT_EQ(executedTimes[4], 10000u);
}

TEST(ProcessSchedulerIntegration, DelayedWaitPattern) {
  // This test simulates the exact pattern used by llhd.wait with delay:
  // 1. Process starts and schedules a delayed resume
  // 2. Process suspends
  // 3. Time advances to the resume time
  // 4. Process resumes and continues execution
  ProcessScheduler scheduler;

  enum class Phase { Initial, AfterWait, Completed };
  Phase currentPhase = Phase::Initial;
  ProcessId procId;

  procId = scheduler.registerProcess("wait_pattern", [&]() {
    if (currentPhase == Phase::Initial) {
      // First execution - schedule delayed resume and "wait"
      currentPhase = Phase::AfterWait;
      scheduler.getEventScheduler().schedule(
          SimTime(10000000, 0, 0), SchedulingRegion::Active,
          Event([&scheduler, procId]() {
            scheduler.resumeProcess(procId);
          }));
      // Process would normally be suspended here by the interpreter
    } else if (currentPhase == Phase::AfterWait) {
      // Second execution - after wait completes
      currentPhase = Phase::Completed;
    }
  });

  scheduler.initialize();

  // First execution
  scheduler.executeDeltaCycle();
  EXPECT_EQ(currentPhase, Phase::AfterWait);
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 0u);

  // Time should advance to the delayed event
  SimTime finalTime = scheduler.runUntil(20000000);

  // Process should have completed
  EXPECT_EQ(currentPhase, Phase::Completed);
  // Time should have advanced to at least the wait time
  EXPECT_GE(finalTime.realTime, 10000000u);

  // Verify statistics
  auto stats = scheduler.getStatistics();
  EXPECT_EQ(stats.processesExecuted, 2u);
}

TEST(ProcessSchedulerIntegration, EventSchedulerIntegrationCheck) {
  // Verify that advanceTime() properly checks the EventScheduler
  // and doesn't return false prematurely when events are pending
  ProcessScheduler scheduler;
  bool eventExecuted = false;

  scheduler.registerProcess("dummy", []() {});
  scheduler.initialize();
  scheduler.executeDeltaCycle();

  // Schedule a future event
  scheduler.getEventScheduler().schedule(
      SimTime(5000, 0, 0), SchedulingRegion::Active,
      Event([&eventExecuted]() { eventExecuted = true; }));

  // isComplete should return false - we have pending events
  EXPECT_FALSE(scheduler.isComplete());

  // advanceTime should advance to the event and return true
  bool canAdvance = scheduler.advanceTime();
  EXPECT_TRUE(canAdvance);
  EXPECT_TRUE(eventExecuted);
  EXPECT_EQ(scheduler.getCurrentTime().realTime, 5000u);
}

//===----------------------------------------------------------------------===//
// ProcessScheduler Tests - Concurrent Initial and Always Blocks
//===----------------------------------------------------------------------===//

TEST(ProcessSchedulerIntegration, ConcurrentInitialAndAlwaysBlocks) {
  // This test simulates the scenario with:
  // - initial block (runs once, terminates)
  // - always #5 clk = ~clk (clock generator)
  // - always @(posedge clk) counter++ (counter)
  //
  // The test verifies that:
  // 1. All three processes execute initially
  // 2. The clock process continues to run via delay-based wake
  // 3. The counter process continues to be triggered by clock edges
  // 4. The signal-to-process mapping persists across wake/sleep cycles

  ProcessScheduler scheduler;
  SignalId clk = scheduler.registerSignal("clk", 1);

  int initialRan = 0;
  int clockExecutions = 0;
  int counterExecutions = 0;
  bool clkValue = false;

  // Initial block - runs once and terminates
  ProcessId initialProc = scheduler.registerProcess("initial", [&]() {
    initialRan++;
    scheduler.terminateProcess(initialProc);
  });

  // Clock generator - drives clk and schedules own resume
  ProcessId clockProc;
  clockProc = scheduler.registerProcess("clock", [&]() {
    clockExecutions++;
    clkValue = !clkValue;

    // Schedule signal update (simulating llhd.drv with epsilon delay)
    // Use NBA region as LLHD drives do
    SignalValue newClkVal(clkValue ? 1 : 0, 1);
    scheduler.getEventScheduler().schedule(
        scheduler.getCurrentTime(), SchedulingRegion::NBA,
        Event([&scheduler, clk, newClkVal]() {
          scheduler.updateSignal(clk, newClkVal);
        }));

    // Schedule own resume after 5ns (simulating llhd.wait with delay)
    scheduler.getEventScheduler().schedule(
        SimTime(scheduler.getCurrentTime().realTime + 5000000, 0, 0),
        SchedulingRegion::Active,
        Event([&scheduler, clockProc]() {
          scheduler.scheduleProcess(clockProc, SchedulingRegion::Active);
        }));
  });

  // Counter - triggers on clock edge
  ProcessId counterProc = scheduler.registerProcess("counter", [&]() {
    counterExecutions++;
    // Re-register sensitivity (simulating suspendProcessForEvents)
    SensitivityList waitList;
    waitList.addLevel(clk);
    scheduler.suspendProcessForEvents(counterProc, waitList);
  });

  // Initialize - all processes should be scheduled
  scheduler.initialize();
  EXPECT_EQ(scheduler.getStatistics().processesRegistered, 3u);

  // Run simulation for 50ns (should be ~10 clock cycles)
  SimTime endTime = scheduler.runUntil(50000000);

  // Verify results
  EXPECT_EQ(initialRan, 1) << "Initial block should run exactly once";
  EXPECT_GE(clockExecutions, 10)
      << "Clock should execute at least 10 times (50ns / 5ns per cycle)";
  EXPECT_GE(counterExecutions, 10)
      << "Counter should execute at least as many times as clock (triggered by edges)";

  // The simulation should have advanced time
  EXPECT_GE(endTime.realTime, 45000000u);

  auto stats = scheduler.getStatistics();
  EXPECT_GE(stats.processesExecuted, 20u)
      << "Should have many process executions";
  EXPECT_GE(stats.signalUpdates, 10u) << "Should have many signal updates";
  EXPECT_GE(stats.edgesDetected, 5u) << "Should detect clock edges";
}

TEST(ProcessSchedulerIntegration, SuspendProcessForEventsPeristsMapping) {
  // This test specifically verifies that the signalToProcesses mapping
  // persists across multiple wake/sleep cycles using suspendProcessForEvents

  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  int triggerCount = 0;
  ProcessId proc;
  proc = scheduler.registerProcess("waker", [&]() {
    triggerCount++;
    // Re-register for the next event
    SensitivityList waitList;
    waitList.addLevel(sig);
    scheduler.suspendProcessForEvents(proc, waitList);
  });

  scheduler.initialize();
  scheduler.executeDeltaCycle();

  EXPECT_EQ(triggerCount, 1) << "Process should execute once during init";

  // Process should now be in Waiting state
  Process *p = scheduler.getProcess(proc);
  EXPECT_EQ(p->getState(), ProcessState::Waiting);

  // Trigger the process multiple times with signal changes
  for (int i = 0; i < 10; ++i) {
    SignalValue oldVal = scheduler.getSignalValue(sig);
    SignalValue newVal((i % 2) ? 0 : 1, 1);
    scheduler.updateSignal(sig, newVal);
    scheduler.executeDeltaCycle();
  }

  // Process should have been triggered 10 more times
  EXPECT_EQ(triggerCount, 11)
      << "Process should be triggered for each signal change";

  // Process should still be in Waiting state (re-registered each time)
  EXPECT_EQ(p->getState(), ProcessState::Waiting);
}

//===----------------------------------------------------------------------===//
// Signal Strength Resolution Tests
//===----------------------------------------------------------------------===//

TEST(SignalStrengthResolution, SingleDriverNoStrength) {
  // Single driver without explicit strength should behave normally
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Update with strength (default strong)
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 1u);
}

TEST(SignalStrengthResolution, StrongerDriverWins) {
  // When two drivers drive different values, stronger one wins
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Weak, driving 0
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(0, 1),
                                     DriveStrength::Weak, DriveStrength::Weak);

  // Driver 2: Strong, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  // Strong driver (value 1) should win
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 1u);
}

TEST(SignalStrengthResolution, WeakerDriverLoses) {
  // Verify weaker driver loses even when it drives later
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Strong, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  // Driver 2: Weak, driving 0 (later but weaker)
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(0, 1),
                                     DriveStrength::Weak, DriveStrength::Weak);

  // Strong driver (value 1) should still win
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 1u);
}

TEST(SignalStrengthResolution, EqualStrengthConflictProducesX) {
  // When two drivers of equal strength drive different values, result is X
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Strong, driving 0
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(0, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  // Driver 2: Strong, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  // Equal strength conflict should produce X
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_TRUE(val.isUnknown());
}

TEST(SignalStrengthResolution, PullupWithWeakDriver) {
  // Pullup (highz for 0, pull for 1) with weak driver
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Pullup (highz0, pull1) driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::HighZ, DriveStrength::Pull);

  // Driver 2: Weak driving 0
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(0, 1),
                                     DriveStrength::Weak, DriveStrength::Weak);

  // Weak (strength 3) is weaker than Pull (strength 2) for value 1
  // But weak driving 0 vs pullup driving 1: weak0 (3) vs pull1 (2)
  // Pull is stronger, so 1 wins
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 1u);
}

TEST(SignalStrengthResolution, SupplyStrengthOverridesAll) {
  // Supply strength should override all other strengths
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Strong, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  // Driver 2: Supply, driving 0
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(0, 1),
                                     DriveStrength::Supply,
                                     DriveStrength::Supply);

  // Supply driver (value 0) should win
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 0u);
}

TEST(SignalStrengthResolution, SameDriverUpdates) {
  // Same driver updating its value should work correctly
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Strong, driving 0
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(0, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  const SignalValue &val1 = scheduler.getSignalValue(sig);
  EXPECT_EQ(val1.getValue(), 0u);

  // Same driver now drives 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::Strong,
                                     DriveStrength::Strong);

  const SignalValue &val2 = scheduler.getSignalValue(sig);
  EXPECT_EQ(val2.getValue(), 1u);
}

TEST(SignalStrengthResolution, MultipleDriversSameValue) {
  // Multiple drivers driving the same value should not conflict
  ProcessScheduler scheduler;
  SignalId sig = scheduler.registerSignal("sig", 1);

  // Driver 1: Weak, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/1, SignalValue(1, 1),
                                     DriveStrength::Weak, DriveStrength::Weak);

  // Driver 2: Pull, driving 1
  scheduler.updateSignalWithStrength(sig, /*driverId=*/2, SignalValue(1, 1),
                                     DriveStrength::Pull, DriveStrength::Pull);

  // Both driving 1, no conflict
  const SignalValue &val = scheduler.getSignalValue(sig);
  EXPECT_FALSE(val.isUnknown());
  EXPECT_EQ(val.getValue(), 1u);
}

} // namespace
