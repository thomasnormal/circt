//===- DebugTest.cpp - Tests for CIRCT Debug Infrastructure ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-debug/Debug.h"
#include "gtest/gtest.h"

using namespace circt::debug;

//===----------------------------------------------------------------------===//
// SignalValue Tests
//===----------------------------------------------------------------------===//

TEST(SignalValueTest, Construction) {
  SignalValue v1(8);
  EXPECT_EQ(v1.getWidth(), 8u);
  EXPECT_TRUE(v1.isFullyDefined());
  EXPECT_EQ(v1.toBinaryString(), "00000000");

  SignalValue v2(8, 0xAB);
  EXPECT_EQ(v2.getWidth(), 8u);
  EXPECT_EQ(v2.toBinaryString(), "10101011");
  EXPECT_EQ(v2.toHexString(), "ab");
}

TEST(SignalValueTest, BitAccess) {
  SignalValue v(8, 0);

  v.setBit(0, LogicValue::One);
  v.setBit(2, LogicValue::Unknown);
  v.setBit(4, LogicValue::HighZ);

  EXPECT_EQ(v.getBit(0), LogicValue::One);
  EXPECT_EQ(v.getBit(1), LogicValue::Zero);
  EXPECT_EQ(v.getBit(2), LogicValue::Unknown);
  EXPECT_EQ(v.getBit(3), LogicValue::Zero);
  EXPECT_EQ(v.getBit(4), LogicValue::HighZ);

  EXPECT_TRUE(v.hasUnknown());
  EXPECT_TRUE(v.hasHighZ());
  EXPECT_FALSE(v.isFullyDefined());
}

TEST(SignalValueTest, StringConversion) {
  SignalValue v(8, 0xAB);
  EXPECT_EQ(v.toBinaryString(), "10101011");
  EXPECT_EQ(v.toHexString(), "ab");

  // With unknowns
  SignalValue v2(4);
  v2.setBit(0, LogicValue::One);
  v2.setBit(1, LogicValue::Zero);
  v2.setBit(2, LogicValue::Unknown);
  v2.setBit(3, LogicValue::HighZ);
  EXPECT_EQ(v2.toBinaryString(), "zx01");
}

TEST(SignalValueTest, FromString) {
  // Binary
  auto v1 = SignalValue::fromString("0b1010", 4);
  ASSERT_TRUE(v1.has_value());
  EXPECT_EQ(v1->toBinaryString(), "1010");

  // Hex
  auto v2 = SignalValue::fromString("0xFF", 8);
  ASSERT_TRUE(v2.has_value());
  EXPECT_EQ(v2->toHexString(), "ff");

  // Verilog style
  auto v3 = SignalValue::fromString("'h5A", 8);
  ASSERT_TRUE(v3.has_value());
  EXPECT_EQ(v3->toHexString(), "5a");

  // With X
  auto v4 = SignalValue::fromString("0b10x1", 4);
  ASSERT_TRUE(v4.has_value());
  EXPECT_EQ(v4->getBit(1), LogicValue::Unknown);
}

TEST(SignalValueTest, Comparison) {
  SignalValue v1(8, 0xAB);
  SignalValue v2(8, 0xAB);
  SignalValue v3(8, 0xCD);

  EXPECT_TRUE(v1 == v2);
  EXPECT_FALSE(v1 == v3);
  EXPECT_TRUE(v1 != v3);
}

TEST(SignalValueTest, ToAPInt) {
  SignalValue v(16, 0x1234);
  llvm::APInt api = v.toAPInt();
  EXPECT_EQ(api.getZExtValue(), 0x1234u);
}

//===----------------------------------------------------------------------===//
// SimTime Tests
//===----------------------------------------------------------------------===//

TEST(SimTimeTest, Construction) {
  SimTime t1;
  EXPECT_EQ(t1.value, 0u);

  SimTime t2(100, SimTime::NS);
  EXPECT_EQ(t2.value, 100u);
  EXPECT_EQ(t2.toString(), "100ns");
}

TEST(SimTimeTest, Conversion) {
  SimTime ns(1000, SimTime::NS);
  EXPECT_DOUBLE_EQ(ns.toNanoseconds(), 1000.0);

  SimTime us(1, SimTime::US);
  EXPECT_DOUBLE_EQ(us.toNanoseconds(), 1000.0);

  SimTime ps(1000000, SimTime::PS);
  EXPECT_DOUBLE_EQ(ps.toNanoseconds(), 1000.0);
}

TEST(SimTimeTest, Comparison) {
  SimTime t1(100, SimTime::NS);
  SimTime t2(200, SimTime::NS);
  SimTime t3(0.1, SimTime::US);

  EXPECT_TRUE(t1 < t2);
  EXPECT_TRUE(t1 <= t2);
  EXPECT_TRUE(t1 == SimTime(100, SimTime::NS));
}

TEST(SimTimeTest, Addition) {
  SimTime t1(100, SimTime::NS);
  SimTime t2(50, SimTime::NS);
  SimTime t3 = t1 + t2;

  EXPECT_EQ(t3.value, 150u);
  EXPECT_EQ(t3.unit, SimTime::NS);
}

//===----------------------------------------------------------------------===//
// Scope Tests
//===----------------------------------------------------------------------===//

TEST(ScopeTest, Hierarchy) {
  auto root = std::make_unique<Scope>("top");
  auto child1 = std::make_unique<Scope>("mod1", root.get());
  auto child2 = std::make_unique<Scope>("mod2", root.get());

  EXPECT_EQ(root->getName(), "top");
  EXPECT_EQ(root->getFullPath(), "top");
  EXPECT_EQ(root->getParent(), nullptr);

  root->addChild(std::move(child1));
  root->addChild(std::move(child2));

  EXPECT_EQ(root->getChildren().size(), 2u);

  auto *found = root->findChild("mod1");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->getName(), "mod1");
  EXPECT_EQ(found->getFullPath(), "top.mod1");
  EXPECT_EQ(found->getParent(), root.get());
}

TEST(ScopeTest, Signals) {
  Scope scope("test");

  SignalInfo sig1;
  sig1.name = "clk";
  sig1.fullPath = "test.clk";
  sig1.type = SignalType::Input;
  sig1.width = 1;

  SignalInfo sig2;
  sig2.name = "data";
  sig2.fullPath = "test.data";
  sig2.type = SignalType::Wire;
  sig2.width = 32;

  scope.addSignal(sig1);
  scope.addSignal(sig2);

  EXPECT_EQ(scope.getSignals().size(), 2u);

  auto *found = scope.findSignal("clk");
  ASSERT_NE(found, nullptr);
  EXPECT_EQ(found->name, "clk");
  EXPECT_EQ(found->width, 1u);
}

//===----------------------------------------------------------------------===//
// SimState Tests
//===----------------------------------------------------------------------===//

TEST(SimStateTest, TimeAndCycle) {
  SimState state;

  EXPECT_EQ(state.getCycle(), 0u);
  EXPECT_EQ(state.getTime().value, 0u);

  state.advanceCycle(5);
  EXPECT_EQ(state.getCycle(), 5u);

  state.advanceTime(SimTime(100, SimTime::NS));
  EXPECT_EQ(state.getTime().value, 100u);
}

TEST(SimStateTest, SignalValues) {
  SimState state;

  state.setSignalValue("test.clk", SignalValue(1, 0));
  state.setSignalValue("test.data", SignalValue(8, 0xAB));

  EXPECT_TRUE(state.hasSignal("test.clk"));
  EXPECT_TRUE(state.hasSignal("test.data"));
  EXPECT_FALSE(state.hasSignal("test.nonexistent"));

  auto clk = state.getSignalValue("test.clk");
  EXPECT_EQ(clk.getWidth(), 1u);
  EXPECT_EQ(clk.getBit(0), LogicValue::Zero);

  auto data = state.getSignalValue("test.data");
  EXPECT_EQ(data.toAPInt().getZExtValue(), 0xABu);
}

//===----------------------------------------------------------------------===//
// Breakpoint Tests
//===----------------------------------------------------------------------===//

TEST(BreakpointTest, LineBreakpoint) {
  LineBreakpoint bp(1, "test.v", 42);

  EXPECT_EQ(bp.getId(), 1u);
  EXPECT_EQ(bp.getType(), Breakpoint::Type::Line);
  EXPECT_EQ(bp.getFile(), "test.v");
  EXPECT_EQ(bp.getLine(), 42u);
  EXPECT_TRUE(bp.isEnabled());
  EXPECT_EQ(bp.getDescription(), "at test.v:42");

  SimState state;
  EXPECT_FALSE(bp.shouldBreak(state)); // No location set

  state.setCurrentLocation("test.v", 42);
  EXPECT_TRUE(bp.shouldBreak(state));

  state.setCurrentLocation("test.v", 43);
  EXPECT_FALSE(bp.shouldBreak(state));

  bp.setEnabled(false);
  state.setCurrentLocation("test.v", 42);
  EXPECT_FALSE(bp.shouldBreak(state));
}

TEST(BreakpointTest, CycleBreakpoint) {
  CycleBreakpoint bp(1, 100);

  EXPECT_EQ(bp.getTargetCycle(), 100u);
  EXPECT_EQ(bp.getDescription(), "at cycle 100");

  SimState state;
  EXPECT_FALSE(bp.shouldBreak(state));

  state.setCycle(100);
  EXPECT_TRUE(bp.shouldBreak(state));

  state.setCycle(101);
  EXPECT_FALSE(bp.shouldBreak(state));
}

TEST(BreakpointTest, TimeBreakpoint) {
  TimeBreakpoint bp(1, SimTime(500, SimTime::NS));

  EXPECT_EQ(bp.getTargetTime().value, 500u);

  SimState state;
  EXPECT_FALSE(bp.shouldBreak(state));

  state.setTime(SimTime(500, SimTime::NS));
  EXPECT_TRUE(bp.shouldBreak(state));
}

//===----------------------------------------------------------------------===//
// BreakpointManager Tests
//===----------------------------------------------------------------------===//

TEST(BreakpointManagerTest, AddRemove) {
  BreakpointManager mgr;

  unsigned id1 = mgr.addLineBreakpoint("test.v", 10);
  unsigned id2 = mgr.addCycleBreakpoint(100);
  unsigned id3 = mgr.addSignalBreakpoint("clk");

  EXPECT_EQ(mgr.getBreakpoints().size(), 3u);

  EXPECT_TRUE(mgr.removeBreakpoint(id2));
  EXPECT_EQ(mgr.getBreakpoints().size(), 2u);

  EXPECT_FALSE(mgr.removeBreakpoint(id2)); // Already removed

  mgr.removeAllBreakpoints();
  EXPECT_EQ(mgr.getBreakpoints().size(), 0u);
}

TEST(BreakpointManagerTest, EnableDisable) {
  BreakpointManager mgr;

  unsigned id = mgr.addCycleBreakpoint(50);
  auto *bp = mgr.getBreakpoint(id);
  ASSERT_NE(bp, nullptr);
  EXPECT_TRUE(bp->isEnabled());

  mgr.enableBreakpoint(id, false);
  EXPECT_FALSE(bp->isEnabled());

  mgr.enableBreakpoint(id, true);
  EXPECT_TRUE(bp->isEnabled());
}

TEST(BreakpointManagerTest, ShouldBreak) {
  BreakpointManager mgr;

  mgr.addCycleBreakpoint(10);
  mgr.addCycleBreakpoint(20);

  SimState state;

  state.setCycle(5);
  EXPECT_FALSE(mgr.shouldBreak(state));

  state.setCycle(10);
  EXPECT_TRUE(mgr.shouldBreak(state));

  auto triggered = mgr.getTriggeredBreakpoints(state);
  EXPECT_EQ(triggered.size(), 1u);
}

//===----------------------------------------------------------------------===//
// Watchpoint Tests
//===----------------------------------------------------------------------===//

TEST(WatchpointTest, Basic) {
  Watchpoint wp(1, "test.data");

  EXPECT_EQ(wp.getId(), 1u);
  EXPECT_EQ(wp.getSignal(), "test.data");
  EXPECT_TRUE(wp.isEnabled());
  EXPECT_EQ(wp.getHistory().size(), 0u);
}

TEST(WatchpointTest, Recording) {
  Watchpoint wp(1, "test.data");

  SimState state;
  state.setSignalValue("test.data", SignalValue(8, 0x00));

  wp.checkAndRecord(state);
  EXPECT_EQ(wp.getHistory().size(), 1u);

  // Same value - no record
  wp.checkAndRecord(state);
  EXPECT_EQ(wp.getHistory().size(), 1u);

  // Different value - record
  state.setSignalValue("test.data", SignalValue(8, 0xFF));
  state.setTime(SimTime(100, SimTime::NS));
  wp.checkAndRecord(state);
  EXPECT_EQ(wp.getHistory().size(), 2u);

  EXPECT_EQ(wp.getHistory()[1].time.value, 100u);
}

//===----------------------------------------------------------------------===//
// ExpressionEvaluator Tests
//===----------------------------------------------------------------------===//

TEST(ExpressionEvaluatorTest, SimpleValues) {
  SimState state;
  state.setSignalValue("test.data", SignalValue(8, 0xAB));

  ExpressionEvaluator eval(state);

  // Signal lookup
  auto result = eval.evaluate("test.data");
  EXPECT_TRUE(result.succeeded);
  EXPECT_EQ(result.value->toAPInt().getZExtValue(), 0xABu);

  // Constant
  auto constResult = eval.evaluate("0xFF");
  EXPECT_TRUE(constResult.succeeded);
  EXPECT_EQ(constResult.value->toAPInt().getZExtValue(), 0xFFu);

  // Unknown signal
  auto unknownResult = eval.evaluate("nonexistent");
  EXPECT_FALSE(unknownResult.succeeded);
}

TEST(ExpressionEvaluatorTest, Comparisons) {
  SimState state;
  state.setSignalValue("test.a", SignalValue(8, 100));
  state.setSignalValue("test.b", SignalValue(8, 50));

  ExpressionEvaluator eval(state);

  EXPECT_TRUE(eval.isTrue("test.a > test.b"));
  EXPECT_FALSE(eval.isTrue("test.a < test.b"));
  EXPECT_TRUE(eval.isTrue("test.a == 100"));
  EXPECT_TRUE(eval.isTrue("test.b != 100"));
}
