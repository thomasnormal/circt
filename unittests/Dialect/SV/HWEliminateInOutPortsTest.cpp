//===- HWEliminateInOutPortsTest.cpp - SV pass unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;

namespace {

TEST(HWEliminateInOutPortsTest, MergeDynamicWriters) {
  static constexpr const char *kIR = R"mlir(
module {
  hw.module @top(inout %io : !hw.array<4x!hw.struct<value: i1, unknown: i1>>,
                 in %sel0 : i2, in %sel1 : i2) {
    %v0 = hw.constant 1 : i1
    %u0 = hw.constant 0 : i1
    %v1 = hw.constant 0 : i1
    %u1 = hw.constant 1 : i1
    %s0 = hw.struct_create (%v0, %u0) : !hw.struct<value: i1, unknown: i1>
    %s1 = hw.struct_create (%v1, %u1) : !hw.struct<value: i1, unknown: i1>
    %elem0 = sv.array_index_inout %io[%sel0] : !hw.inout<array<4xstruct<value: i1, unknown: i1>>>, i2
    %elem1 = sv.array_index_inout %io[%sel1] : !hw.inout<array<4xstruct<value: i1, unknown: i1>>>, i2
    sv.assign %elem0, %s0 : !hw.struct<value: i1, unknown: i1>
    sv.assign %elem1, %s1 : !hw.struct<value: i1, unknown: i1>
    hw.output
  }
}
)mlir";

  MLIRContext context;
  context.loadDialect<hw::HWDialect, sv::SVDialect>();
  auto module = parseSourceString<ModuleOp>(kIR, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  sv::HWEliminateInOutPortsOptions options;
  options.resolveReadWrite = true;
  pm.addPass(sv::createHWEliminateInOutPortsPass(options));
  ASSERT_TRUE(succeeded(pm.run(module.get())));

  auto top = module->lookupSymbol<hw::HWModuleOp>("top");
  ASSERT_TRUE(top);

  unsigned inoutCount = 0;
  unsigned readInputs = 0;
  unsigned writeOutputs = 0;
  for (auto &port : top.getPortList()) {
    if (port.dir == hw::ModulePort::Direction::InOut)
      ++inoutCount;
    if (port.dir == hw::ModulePort::Direction::Input &&
        port.name.getValue().starts_with("io_rd_dyn"))
      ++readInputs;
    if (port.dir == hw::ModulePort::Direction::Output &&
        port.name.getValue().starts_with("io_wr_dyn"))
      ++writeOutputs;
  }

  EXPECT_EQ(inoutCount, 0u);
  EXPECT_EQ(readInputs, 1u);
  EXPECT_EQ(writeOutputs, 1u);

  auto output = cast<hw::OutputOp>(top.getBodyBlock()->getTerminator());
  EXPECT_EQ(output.getNumOperands(), 1u);

  EXPECT_TRUE(top.getBodyBlock()->getOps<sv::AssignOp>().empty());
  EXPECT_TRUE(top.getBodyBlock()->getOps<sv::ArrayIndexInOutOp>().empty());
}

TEST(HWEliminateInOutPortsTest, ResolveReadWrite2State) {
  static constexpr const char *kIR = R"mlir(
module {
  hw.module @top(inout %io : i1, out o : i1) {
    %c0 = hw.constant 0 : i1
    sv.assign %io, %c0 : i1
    %read = sv.read_inout %io : !hw.inout<i1>
    hw.output %read : i1
  }
}
)mlir";

  MLIRContext context;
  context.loadDialect<hw::HWDialect, sv::SVDialect, verif::VerifDialect>();
  auto module = parseSourceString<ModuleOp>(kIR, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  sv::HWEliminateInOutPortsOptions options;
  options.resolveReadWrite = true;
  pm.addPass(sv::createHWEliminateInOutPortsPass(options));
  ASSERT_TRUE(succeeded(pm.run(module.get())));

  auto top = module->lookupSymbol<hw::HWModuleOp>("top");
  ASSERT_TRUE(top);

  unsigned inoutCount = 0;
  unsigned readInputs = 0;
  unsigned unknownInputs = 0;
  unsigned writeOutputs = 0;
  for (auto &port : top.getPortList()) {
    if (port.dir == hw::ModulePort::Direction::InOut)
      ++inoutCount;
    if (port.dir == hw::ModulePort::Direction::Input &&
        port.name.getValue().starts_with("io_rd"))
      ++readInputs;
    if (port.dir == hw::ModulePort::Direction::Input &&
        port.name.getValue().starts_with("io_unknown"))
      ++unknownInputs;
    if (port.dir == hw::ModulePort::Direction::Output &&
        port.name.getValue().starts_with("io_wr"))
      ++writeOutputs;
  }

  EXPECT_EQ(inoutCount, 0u);
  EXPECT_EQ(readInputs, 1u);
  EXPECT_EQ(unknownInputs, 1u);
  EXPECT_EQ(writeOutputs, 1u);

  EXPECT_TRUE(top.getBodyBlock()->getOps<sv::AssignOp>().empty());
  EXPECT_TRUE(top.getBodyBlock()->getOps<sv::ReadInOutOp>().empty());
}

} // namespace
