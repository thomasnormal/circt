//===- I1ValueSimplifierTest.cpp - i1 simplifier tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/I1ValueSimplifier.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

TEST(I1ValueSimplifierTest, IcmpEqFalseInverts) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;
  ports.push_back(
      {builder.getStringAttr("clk"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto clk = top.getBodyBlock()->getArgument(0);
  auto zero = hw::ConstantOp::create(builder, loc, i1, 0);
  auto eq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq, clk,
                                 zero)
                .getResult();

  auto simplified = simplifyI1Value(eq);
  EXPECT_TRUE(simplified.value);
  EXPECT_EQ(simplified.value, clk);
  EXPECT_TRUE(simplified.invert);
}

TEST(I1ValueSimplifierTest, NeutralOpsCollapse) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;
  ports.push_back(
      {builder.getStringAttr("clk"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto clk = top.getBodyBlock()->getArgument(0);
  auto one = hw::ConstantOp::create(builder, loc, i1, 1);
  auto zero = hw::ConstantOp::create(builder, loc, i1, 0);

  auto andTrue = comb::AndOp::create(builder, loc, clk, one).getResult();
  auto orFalse = comb::OrOp::create(builder, loc, clk, zero).getResult();
  auto orTrue = comb::OrOp::create(builder, loc, clk, one).getResult();

  auto simplifiedAnd = simplifyI1Value(andTrue);
  EXPECT_TRUE(simplifiedAnd.value);
  EXPECT_EQ(simplifiedAnd.value, clk);
  EXPECT_FALSE(simplifiedAnd.invert);

  auto simplifiedOr = simplifyI1Value(orFalse);
  EXPECT_TRUE(simplifiedOr.value);
  EXPECT_EQ(simplifiedOr.value, clk);
  EXPECT_FALSE(simplifiedOr.invert);

  auto simplifiedConst = simplifyI1Value(orTrue);
  EXPECT_FALSE(simplifiedConst.value);
}

TEST(I1ValueSimplifierTest, FromClockToClockUnwrap) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;
  ports.push_back(
      {builder.getStringAttr("clk"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto clk = top.getBodyBlock()->getArgument(0);
  auto toClock = seq::ToClockOp::create(builder, loc, clk);
  auto fromClock = seq::FromClockOp::create(builder, loc, toClock);

  auto simplified = simplifyI1Value(fromClock.getResult());
  EXPECT_TRUE(simplified.value);
  EXPECT_EQ(simplified.value, clk);
  EXPECT_FALSE(simplified.invert);
}

TEST(I1ValueSimplifierTest, MuxSimplification) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect, seq::SeqDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;
  ports.push_back(
      {builder.getStringAttr("clk"), i1, hw::ModulePort::Direction::Input});
  ports.push_back(
      {builder.getStringAttr("alt"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto clk = top.getBodyBlock()->getArgument(0);
  auto alt = top.getBodyBlock()->getArgument(1);
  auto one = hw::ConstantOp::create(builder, loc, i1, 1);
  auto zero = hw::ConstantOp::create(builder, loc, i1, 0);

  auto muxTrue = comb::MuxOp::create(builder, loc, one, clk, alt).getResult();
  auto muxFalse = comb::MuxOp::create(builder, loc, zero, clk, alt).getResult();
  auto muxSame = comb::MuxOp::create(builder, loc, clk, alt, alt).getResult();

  auto simplifiedTrue = simplifyI1Value(muxTrue);
  EXPECT_TRUE(simplifiedTrue.value);
  EXPECT_EQ(simplifiedTrue.value, clk);

  auto simplifiedFalse = simplifyI1Value(muxFalse);
  EXPECT_TRUE(simplifiedFalse.value);
  EXPECT_EQ(simplifiedFalse.value, alt);

  auto simplifiedSame = simplifyI1Value(muxSame);
  EXPECT_TRUE(simplifiedSame.value);
  EXPECT_EQ(simplifiedSame.value, alt);
}

} // namespace
