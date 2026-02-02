//===- TwoStateUtilsTest.cpp - 2-state helper tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/TwoStateUtils.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

static bool isConstInt(Value value, unsigned width, uint64_t expected) {
  if (auto cst = value.getDefiningOp<hw::ConstantOp>()) {
    if (auto intTy = dyn_cast<IntegerType>(cst.getType());
        intTy && intTy.getWidth() == width)
      return cst.getValue() == APInt(width, expected);
  }
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue())) {
      if (auto intTy = dyn_cast<IntegerType>(intAttr.getType());
          intTy && intTy.getWidth() == width)
        return intAttr.getValue() == APInt(width, expected);
    }
  }
  return false;
}

TEST(TwoStateUtilsTest, ResolveTwoStateValuesWithStrengthStrongWins) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto i1 = builder.getI1Type();
  auto zero = hw::ConstantOp::create(builder, loc, i1, 0).getResult();
  auto one = hw::ConstantOp::create(builder, loc, i1, 1).getResult();
  auto enable = hw::ConstantOp::create(builder, loc, i1, 1).getResult();
  auto unknown = hw::ConstantOp::create(builder, loc, i1, 1).getResult();

  SmallVector<Value, 2> values({zero, one});
  SmallVector<Value, 2> enables({enable, enable});
  SmallVector<unsigned, 2> strength0({1u, 3u});
  SmallVector<unsigned, 2> strength1({1u, 3u});

  auto resolved = resolveTwoStateValuesWithStrength(
      builder, loc, values, enables, strength0, strength1, unknown);
  ASSERT_TRUE(resolved);
  EXPECT_TRUE(isConstInt(resolved, 1, 0));
}

TEST(TwoStateUtilsTest, ResolveTwoStateValuesWithStrengthConflictUsesUnknown) {
  MLIRContext context;
  context.loadDialect<hw::HWDialect, comb::CombDialect>();

  auto loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto zero = hw::ConstantOp::create(builder, loc, APInt(2, 0)).getResult();
  auto ones = hw::ConstantOp::create(builder, loc, APInt(2, 3)).getResult();
  auto enable =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1).getResult();
  auto unknown = hw::ConstantOp::create(builder, loc, APInt(2, 2)).getResult();

  SmallVector<Value, 2> values({zero, ones});
  SmallVector<Value, 2> enables({enable, enable});
  SmallVector<unsigned, 2> strength0({1u, 1u});
  SmallVector<unsigned, 2> strength1({1u, 1u});

  auto resolved = resolveTwoStateValuesWithStrength(
      builder, loc, values, enables, strength0, strength1, unknown);
  ASSERT_TRUE(resolved);
  EXPECT_TRUE(isConstInt(resolved, 2, 2));
}

} // namespace
