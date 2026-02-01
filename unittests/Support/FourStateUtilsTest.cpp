//===- FourStateUtilsTest.cpp - 4-state helper tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/FourStateUtils.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

static bool isConstI1(Value value, bool expected) {
  if (auto cst = value.getDefiningOp<hw::ConstantOp>()) {
    if (auto intTy = dyn_cast<IntegerType>(cst.getType());
        intTy && intTy.getWidth() == 1)
      return cst.getValue().isAllOnes() == expected;
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValue()))
      return boolAttr.getValue() == expected;
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
          intTy && intTy.getWidth() == 1)
        return intAttr.getValue().isAllOnes() == expected;
    }
  }
  return false;
}

static Value makeFourState(ImplicitLocOpBuilder &builder, hw::StructType type,
                           bool value, bool unknown) {
  auto i1 = builder.getI1Type();
  auto val = hw::ConstantOp::create(builder, builder.getLoc(), i1,
                                    value ? 1 : 0)
                 .getResult();
  auto unk = hw::ConstantOp::create(builder, builder.getLoc(), i1,
                                    unknown ? 1 : 0)
                 .getResult();
  return hw::StructCreateOp::create(builder, builder.getLoc(), type,
                                    ValueRange{val, unk})
      .getResult();
}

static hw::StructType getFourStateType(MLIRContext &context,
                                       ImplicitLocOpBuilder &builder,
                                       Type elementType) {
  SmallVector<hw::StructType::FieldInfo, 2> fields;
  fields.push_back({builder.getStringAttr("value"), elementType});
  fields.push_back({builder.getStringAttr("unknown"), elementType});
  return hw::StructType::get(&context, fields);
}

TEST(FourStateUtilsTest, ResolveFourStatePairNoConflict) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto lhs = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);
  auto rhs = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);

  auto resolved = resolveFourStatePair(builder, loc, lhs, rhs);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(0), true));
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), false));
}

TEST(FourStateUtilsTest, ResolveFourStatePairConflictSetsUnknown) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto lhs = makeFourState(builder, structTy, /*value=*/false,
                           /*unknown=*/false);
  auto rhs = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);

  auto resolved = resolveFourStatePair(builder, loc, lhs, rhs);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), true));
}

TEST(FourStateUtilsTest, ResolveFourStatePairRejectsMismatchedTypes) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto i2 = builder.getIntegerType(2);
  auto structTyA = getFourStateType(context, builder, i1);
  auto structTyB = getFourStateType(context, builder, i2);

  auto lhs = makeFourState(builder, structTyA, /*value=*/true,
                           /*unknown=*/false);
  auto rhsVal = hw::ConstantOp::create(builder, i2, 0);
  auto rhsUnk = hw::ConstantOp::create(builder, i2, 0);
  auto rhs = hw::StructCreateOp::create(builder, loc, structTyB,
                                        ValueRange{rhsVal.getResult(),
                                                   rhsUnk.getResult()})
                 .getResult();

  auto resolved = resolveFourStatePair(builder, loc, lhs, rhs);
  EXPECT_FALSE(resolved);
}

TEST(FourStateUtilsTest, ResolveFourStateValuesWithEnableSelectsEnabled) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto lhs = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);
  auto rhs = makeFourState(builder, structTy, /*value=*/false,
                           /*unknown=*/false);
  auto enTrue = hw::ConstantOp::create(builder, loc, i1, 1).getResult();
  auto enFalse = hw::ConstantOp::create(builder, loc, i1, 0).getResult();

  SmallVector<Value, 2> values{lhs, rhs};
  SmallVector<Value, 2> enables{enTrue, enFalse};
  auto resolved =
      resolveFourStateValuesWithEnable(builder, loc, values, enables);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(0), true));
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), false));
}

TEST(FourStateUtilsTest, ResolveFourStateValuesWithEnableAllDisabledUnknown) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto lhs = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);
  auto rhs = makeFourState(builder, structTy, /*value=*/false,
                           /*unknown=*/false);
  auto enFalse = hw::ConstantOp::create(builder, loc, i1, 0).getResult();

  SmallVector<Value, 2> values{lhs, rhs};
  SmallVector<Value, 2> enables{enFalse, enFalse};
  auto resolved =
      resolveFourStateValuesWithEnable(builder, loc, values, enables);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), true));
}

TEST(FourStateUtilsTest, ResolveFourStateValuesWithStrengthStrongWins) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto zero = makeFourState(builder, structTy, /*value=*/false,
                            /*unknown=*/false);
  auto one = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);
  auto enTrue = hw::ConstantOp::create(builder, loc, i1, 1).getResult();

  SmallVector<Value, 2> values{zero, one};
  SmallVector<Value, 2> enables{enTrue, enTrue};
  SmallVector<unsigned, 2> strengths{1u, 3u};
  SmallVector<unsigned, 2> widths{1u, 3u};
  auto resolved = resolveFourStateValuesWithStrength(builder, loc, values,
                                                     enables, strengths, widths);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(0), false));
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), false));
}

TEST(FourStateUtilsTest, ResolveFourStateValuesWithStrengthConflictUnknown) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto structTy = getFourStateType(context, builder, i1);

  auto zero = makeFourState(builder, structTy, /*value=*/false,
                            /*unknown=*/false);
  auto one = makeFourState(builder, structTy, /*value=*/true,
                           /*unknown=*/false);
  auto enTrue = hw::ConstantOp::create(builder, loc, i1, 1).getResult();

  SmallVector<Value, 2> values{zero, one};
  SmallVector<Value, 2> enables{enTrue, enTrue};
  SmallVector<unsigned, 2> strengths{1u, 1u};
  SmallVector<unsigned, 2> widths{1u, 1u};
  auto resolved = resolveFourStateValuesWithStrength(builder, loc, values,
                                                     enables, strengths, widths);
  ASSERT_TRUE(resolved);

  auto structCreate = resolved.getDefiningOp<hw::StructCreateOp>();
  ASSERT_TRUE(structCreate);
  ASSERT_EQ(structCreate.getNumOperands(), 2u);
  EXPECT_TRUE(isConstI1(structCreate.getOperand(1), true));
}

} // namespace
