//===- BoolConditionTest.cpp - BoolCondition unit tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/BoolCondition.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

TEST(BoolConditionTest, ConstantDetection) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto top = hw::HWModuleOp::create(builder, builder.getStringAttr("Top"),
                                    ArrayRef<hw::PortInfo>{});
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto one = hw::ConstantOp::create(builder, llvm::APInt(1, 1));
  auto zero = hw::ConstantOp::create(builder, llvm::APInt(1, 0));

  BoolCondition condOne(one.getResult());
  BoolCondition condZero(zero.getResult());
  EXPECT_TRUE(condOne.isTrue());
  EXPECT_TRUE(condZero.isFalse());

  auto materialized = condOne.materialize(builder, loc);
  auto constOp = materialized.getDefiningOp<hw::ConstantOp>();
  ASSERT_TRUE(constOp);
  EXPECT_TRUE(constOp.getValue().isAllOnes());
}

TEST(BoolConditionTest, ValueOperations) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  SmallVector<hw::PortInfo> ports;
  ports.push_back(
      {builder.getStringAttr("a"), i1, hw::ModulePort::Direction::Input});
  ports.push_back(
      {builder.getStringAttr("b"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto argA = top.getBodyBlock()->getArgument(0);
  auto argB = top.getBodyBlock()->getArgument(1);

  BoolCondition condA(argA);
  BoolCondition condB(argB);

  auto orCond = condA.orWith(condB, builder);
  EXPECT_TRUE(orCond.getValue().getDefiningOp<comb::OrOp>());

  auto andCond = condA.andWith(condB, builder);
  EXPECT_TRUE(andCond.getValue().getDefiningOp<comb::AndOp>());

  auto invCond = condA.inverted(builder);
  auto *invOp = invCond.getValue().getDefiningOp();
  EXPECT_TRUE(invOp);
  EXPECT_TRUE(isa<comb::XorOp>(invOp));
}

} // namespace
