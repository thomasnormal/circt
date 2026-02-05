//===- SMTDeadCodeElimination.cpp - Prune unused SMT ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace circt {
namespace {

static bool isSMTDialectOp(Operation *op) {
  if (auto *dialect = op->getDialect())
    return dialect->getNamespace() == "smt";
  return false;
}

static bool isSMTRootOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "smt.assert" || name == "smt.check" || name == "smt.reset" ||
         name == "smt.push" || name == "smt.pop" || name == "smt.set_logic" ||
         name == "smt.yield";
}

struct SMTDeadCodeEliminationPass
    : public PassWrapper<SMTDeadCodeEliminationPass,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "smt.solver")
        return;
      runOnSolver(op);
    });
  }

  void runOnSolver(Operation *solverOp) {
    // Erase unused SMT ops bottom-up.
    //
    // Important: SMT statement operations (assert/check/push/pop/...) may be
    // nested under non-SMT control flow (e.g. `scf.if`). Only remove operations
    // that are unused *and* trivially dead, plus unused `smt.declare_fun`
    // (which has effects but is safe to drop if its result is unused).
    solverOp->walk<WalkOrder::PostOrder, mlir::ReverseIterator>(
        [&](Operation *op) {
      if (op == solverOp)
        return;
      if (!isSMTDialectOp(op))
        return;
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (isSMTRootOp(op))
        return;
      if (!op->use_empty())
        return;

      if (wouldOpBeTriviallyDead(op) ||
          op->getName().getStringRef() == "smt.declare_fun") {
        op->erase();
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createSMTDeadCodeEliminationPass() {
  return std::make_unique<SMTDeadCodeEliminationPass>();
}

} // namespace circt
