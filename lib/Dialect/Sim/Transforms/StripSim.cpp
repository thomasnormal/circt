//===- StripSim.cpp - Strip simulation-only ops --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes operations from the Sim dialect, which are not meaningful
// for non-simulation flows (e.g. BMC and LEC).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_STRIPSIM
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace circt;
using namespace sim;

namespace {
struct StripSimPass : impl::StripSimBase<StripSimPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, hw::HWDialect, llhd::LLHDDialect,
                    seq::SeqDialect, verif::VerifDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<mlir::Operation *> toErase;
    llvm::SmallPtrSet<mlir::Operation *, 32> eraseSet;
    auto markErase = [&](mlir::Operation *op) {
      if (!op)
        return;
      if (!eraseSet.insert(op).second)
        return;
      toErase.push_back(op);
    };
    auto invertI1 = [](mlir::Location loc, mlir::OpBuilder &builder,
                       mlir::Value value) -> mlir::Value {
      auto one = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      return comb::XorOp::create(builder, loc, value, one);
    };
    auto buildBlockGuard = [&](mlir::Block *block, mlir::Location loc,
                               mlir::OpBuilder &builder) -> mlir::Value {
      struct GuardTerm {
        mlir::Value cond;
        bool invert = false;
        bool isConstTrue = false;
      };
      llvm::SmallVector<GuardTerm> terms;
      for (mlir::Block *pred : block->getPredecessors()) {
        auto *terminator = pred->getTerminator();
        if (auto br = dyn_cast<mlir::cf::BranchOp>(terminator)) {
          if (br.getDest() != block)
            continue;
          terms.push_back(GuardTerm{/*cond=*/{}, /*invert=*/false,
                                   /*isConstTrue=*/true});
          continue;
        }
        if (auto condBr = dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
          if (condBr.getTrueDest() == block) {
            terms.push_back(
                GuardTerm{condBr.getCondition(), /*invert=*/false});
            continue;
          }
          if (condBr.getFalseDest() == block) {
            terms.push_back(
                GuardTerm{condBr.getCondition(), /*invert=*/true});
            continue;
          }
        }
        return {};
      }
      if (terms.empty())
        return {};
      llvm::SmallVector<mlir::Value> guards;
      guards.reserve(terms.size());
      for (const auto &term : terms) {
        if (term.isConstTrue) {
          auto one =
              hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
          guards.push_back(one);
          continue;
        }
        auto guard = term.cond;
        if (term.invert)
          guard = invertI1(loc, builder, guard);
        guards.push_back(guard);
      }
      if (guards.size() == 1)
        return guards.front();
      return comb::OrOp::create(builder, loc, guards, false);
    };
    module.walk([&](mlir::Operation *op) {
      if (auto currentTime = dyn_cast<llhd::CurrentTimeOp>(op)) {
        mlir::OpBuilder builder(currentTime);
        auto zeroTime = llhd::ConstantTimeOp::create(
            builder, currentTime.getLoc(), /*time=*/0, /*timeUnit=*/"ns",
            /*delta=*/0, /*epsilon=*/0);
        currentTime.replaceAllUsesWith(zeroTime.getResult());
        markErase(currentTime);
        return;
      }

      if (!isa_and_nonnull<SimDialect>(op->getDialect()))
        return;

      bool inProcess = op->getParentOfType<llhd::ProcessOp>() != nullptr;
      auto emitFailAssert = [&](mlir::Location loc, mlir::OpBuilder &builder,
                                mlir::Value enable) {
        auto falseVal =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
        verif::AssertOp::create(builder, loc, falseVal, enable,
                                mlir::StringAttr());
      };
      auto emitClockedFailAssert = [&](mlir::Location loc,
                                       mlir::OpBuilder &builder,
                                       mlir::Value clock,
                                       mlir::Value enable) {
        auto falseVal =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
        auto clockI1 = seq::FromClockOp::create(builder, loc, clock);
        verif::ClockedAssertOp::create(
            builder, loc, falseVal, verif::ClockEdge::Pos, clockI1, enable,
            mlir::StringAttr());
      };
      auto emitFinishEvent = [&](mlir::Location loc, mlir::OpBuilder &builder,
                                 mlir::Value event) {
        auto finishOp = verif::AssumeOp::create(builder, loc, event, mlir::Value(),
                                                mlir::StringAttr());
        finishOp->setAttr("bmc.finish", builder.getUnitAttr());
      };
      auto getOrTrue = [&](mlir::Location loc, mlir::OpBuilder &builder,
                           mlir::Value value) -> mlir::Value {
        if (value)
          return value;
        return hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      };

      if (auto pauseOp = dyn_cast<sim::PauseOp>(op)) {
        if (inProcess) {
          markErase(op);
          return;
        }
        mlir::OpBuilder builder(pauseOp);
        auto guard =
            buildBlockGuard(pauseOp->getBlock(), pauseOp.getLoc(), builder);
        emitFailAssert(pauseOp.getLoc(), builder, guard);
      } else if (auto termOp = dyn_cast<sim::TerminateOp>(op)) {
        if (inProcess) {
          markErase(op);
          return;
        }
        mlir::OpBuilder builder(termOp);
        auto guard =
            buildBlockGuard(termOp->getBlock(), termOp.getLoc(), builder);
        if (!termOp.getSuccess()) {
          emitFailAssert(termOp.getLoc(), builder, guard);
        } else {
          emitFinishEvent(termOp.getLoc(), builder,
                          getOrTrue(termOp.getLoc(), builder, guard));
        }
      } else if (auto pauseOp = dyn_cast<sim::ClockedPauseOp>(op)) {
        if (inProcess) {
          markErase(op);
          return;
        }
        mlir::OpBuilder builder(pauseOp);
        mlir::Value enable = pauseOp.getCondition();
        if (auto guard =
                buildBlockGuard(pauseOp->getBlock(), pauseOp.getLoc(), builder))
          enable = comb::AndOp::create(builder, pauseOp.getLoc(), guard, enable);
        emitClockedFailAssert(pauseOp.getLoc(), builder, pauseOp.getClock(),
                              enable);
      } else if (auto termOp = dyn_cast<sim::ClockedTerminateOp>(op)) {
        if (inProcess) {
          markErase(op);
          return;
        }
        mlir::OpBuilder builder(termOp);
        mlir::Value enable = termOp.getCondition();
        if (auto guard =
                buildBlockGuard(termOp->getBlock(), termOp.getLoc(), builder))
          enable = comb::AndOp::create(builder, termOp.getLoc(), guard, enable);
        if (!termOp.getSuccess()) {
          emitClockedFailAssert(termOp.getLoc(), builder, termOp.getClock(),
                                enable);
        } else {
          auto clockI1 =
              seq::FromClockOp::create(builder, termOp.getLoc(), termOp.getClock());
          auto event =
              comb::AndOp::create(builder, termOp.getLoc(), clockI1, enable);
          emitFinishEvent(termOp.getLoc(), builder, event);
        }
      }
      if (llvm::any_of(op->getUsers(), [](mlir::Operation *user) {
            return user->getName().getDialectNamespace() != "sim";
          }))
        return;
      markErase(op);
    });

    llvm::SmallVector<mlir::Operation *> filtered;
    filtered.reserve(toErase.size());
    for (mlir::Operation *op : toErase) {
      bool ancestorMarked = false;
      for (auto *parent = op->getParentOp(); parent;
           parent = parent->getParentOp()) {
        if (eraseSet.contains(parent)) {
          ancestorMarked = true;
          break;
        }
      }
      if (!ancestorMarked)
        filtered.push_back(op);
    }

    for (mlir::Operation *op : llvm::reverse(filtered))
      op->erase();
  }
};
} // namespace
