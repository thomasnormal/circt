//===- CommutativeValueEquivalenceTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/CommutativeValueEquivalence.h"
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

TEST(CommutativeValueEquivalenceTest, XorCommutation) {
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

  auto xorAB = comb::XorOp::create(builder, loc, argA, argB).getResult();
  auto xorBA = comb::XorOp::create(builder, loc, argB, argA).getResult();

  CommutativeValueEquivalence equiv;
  EXPECT_TRUE(equiv.isEquivalent(xorAB, xorBA));
}

TEST(CommutativeValueEquivalenceTest, ICmpCommutation) {
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

  auto cmpAB = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                    argA, argB)
                   .getResult();
  auto cmpBA = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                    argB, argA)
                   .getResult();

  CommutativeValueEquivalence equiv;
  EXPECT_TRUE(equiv.isEquivalent(cmpAB, cmpBA));
}

TEST(CommutativeValueEquivalenceTest, NonEquivalent) {
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
  ports.push_back(
      {builder.getStringAttr("c"), i1, hw::ModulePort::Direction::Input});
  auto top =
      hw::HWModuleOp::create(builder, builder.getStringAttr("Top"), ports);
  builder.setInsertionPointToStart(top.getBodyBlock());

  auto argA = top.getBodyBlock()->getArgument(0);
  auto argB = top.getBodyBlock()->getArgument(1);
  auto argC = top.getBodyBlock()->getArgument(2);

  auto xorAB = comb::XorOp::create(builder, loc, argA, argB).getResult();
  auto xorAC = comb::XorOp::create(builder, loc, argA, argC).getResult();

  CommutativeValueEquivalence equiv;
  EXPECT_FALSE(equiv.isEquivalent(xorAB, xorAC));
}

} // namespace
