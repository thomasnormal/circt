//===- DeleteLocalVar.cpp - Delete local temporary variables --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimplifyProcedures pass.
// Use to insert a local "shadow" variable in always blocks for every
// module-level variable that the process modifies.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace circt {
namespace moore {
#define GEN_PASS_DEF_SIMPLIFYPROCEDURES
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace moore
} // namespace circt

using namespace circt;
using namespace moore;

namespace {
struct SimplifyProceduresPass
    : public circt::moore::impl::SimplifyProceduresBase<
          SimplifyProceduresPass> {
  void runOnOperation() override;
};
} // namespace

std::unique_ptr<mlir::Pass> circt::moore::createSimplifyProceduresPass() {
  return std::make_unique<SimplifyProceduresPass>();
}

void SimplifyProceduresPass::runOnOperation() {
  getOperation()->walk([&](ProcedureOp procedureOp) {
    bool hasCallBoundary = false;
    bool hasSuspensionPoint = false;
    procedureOp.walk([&](Operation *nestedOp) {
      if (isa<mlir::func::CallOp, mlir::func::CallIndirectOp>(nestedOp))
        hasCallBoundary = true;
      if (isa<WaitEventOp, WaitConditionOp, WaitDelayOp, WaitForkOp>(nestedOp))
        hasSuspensionPoint = true;
    });
    // Procedure-local shadowing of module globals is only safe within a
    // single activation. If the procedure can suspend (wait/event/delay), the
    // cached local mirror can go stale across wakeups and break blocking
    // assignment semantics. Keep direct global reads/writes in that case.
    if (hasCallBoundary || hasSuspensionPoint)
      return;

    mlir::OpBuilder builder(&getContext());
    DenseSet<Value> processedGlobals;

    // Use to collect blocking assignments that have been replaced by a "shadow"
    // variable.
    procedureOp.walk([&](Operation *op) {
      SmallVector<std::tuple<BlockingAssignOp, Value, Value>> assignOps;
      auto &nestedOp = *op;
      // Only create a "shadow" varaible for the global variable used by other
      // operations in the procedure body.
      if (isa<ReadOp>(nestedOp) &&
          isa<SVModuleOp>(
              nestedOp.getOperand(0).getDefiningOp()->getParentOp())) {
        if (!processedGlobals.insert(nestedOp.getOperand(0)).second)
          return;

        // Collect the users of the global variable that is mentioned above.
        DenseSet<Operation *> users;
        for (auto *user : nestedOp.getOperand(0).getUsers())
          // Ensuring don't handle the users existing in another procedure body.
          if (procedureOp->isAncestor(user))
            users.insert(user);

        // Only create a shadow variable for globals that are modified in this
        // procedure. Read-only globals must remain direct reads.
        bool hasBlockingAssignUser = llvm::any_of(users, [](Operation *user) {
          return isa<BlockingAssignOp>(user);
        });
        if (!hasBlockingAssignUser)
          return;

        // Because the operand of moore.event_wait is net.
        if (auto varOp = llvm::dyn_cast_or_null<VariableOp>(
                nestedOp.getOperand(0).getDefiningOp())) {
          auto resultType = varOp.getResult().getType();
          builder.setInsertionPointToStart(&procedureOp.getBody().front());
          auto readOp = ReadOp::create(
              builder, nestedOp.getLoc(),
              cast<RefType>(resultType).getNestedType(), varOp.getResult());
          auto newVarOp = VariableOp::create(builder, nestedOp.getLoc(),
                                             resultType, StringAttr{}, Value{});
          BlockingAssignOp::create(builder, nestedOp.getLoc(), newVarOp,
                                   readOp);
          builder.clearInsertionPoint();

          // Replace the users of the global variable with a corresponding
          // "shadow" variable.
          for (auto *user : users) {
            // Preserve values read inside explicit event controls. Rewriting
            // those reads to the local shadow variable would make wait_event
            // observe a procedure-local value instead of the module signal.
            if (user->getParentOfType<WaitEventOp>())
              continue;
            user->replaceUsesOfWith(varOp.getResult(), newVarOp);
            if (auto assignOp = dyn_cast<BlockingAssignOp>(user))
              assignOps.push_back({assignOp, newVarOp, varOp});
          }
        }
      }

      // Ensure the global variable has the correct value. So needing to create
      // a blocking assign for the global variable when the "shadow" variable
      // has a new value.
      for (auto [assignOp, localVar, var] : assignOps) {
        builder.setInsertionPointAfter(assignOp);
        auto readOp = ReadOp::create(builder, assignOp.getLoc(), localVar);
        BlockingAssignOp::create(builder, assignOp.getLoc(), var, readOp);
        builder.clearInsertionPoint();
      }
    });
  });
}
