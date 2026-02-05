//===- SMTDeadCodeEliminationTest.cpp - SMT DCE tests ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Passes.h"

#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(SMTDeadCodeEliminationTest, DropsUnusedDeclarations) {
  MLIRContext context;
  context.loadDialect<smt::SMTDialect>();

  const char *ir = R"mlir(
    module {
      smt.solver() : () -> () {
        %a = smt.declare_fun "a" : !smt.bool
        %b = smt.declare_fun : !smt.bool
        %true = smt.constant true
        smt.assert %true
        smt.yield
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createSMTDeadCodeEliminationPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  int numDeclares = 0;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "smt.declare_fun")
      ++numDeclares;
  });
  EXPECT_EQ(numDeclares, 0);
}

TEST(SMTDeadCodeEliminationTest, KeepsNestedAssertions) {
  MLIRContext context;
  context.loadDialect<smt::SMTDialect, arith::ArithDialect, scf::SCFDialect>();

  const char *ir = R"mlir(
    module {
      smt.solver() : () -> () {
        %cond = arith.constant true
        scf.if %cond {
          %true = smt.constant true
          smt.assert %true
          scf.yield
        }
        smt.yield
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &context);
  ASSERT_TRUE(module);

  PassManager pm(&context);
  pm.addPass(circt::createSMTDeadCodeEliminationPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));

  int numAsserts = 0;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "smt.assert")
      ++numAsserts;
  });
  EXPECT_EQ(numAsserts, 1);
}
