//===- LowerToBMC.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToBMCPass : public circt::impl::LowerToBMCBase<LowerToBMCPass> {
  using LowerToBMCBase::LowerToBMCBase;
  void runOnOperation() override;
};
} // namespace

static LogicalResult
inlineLlhdCombinationalOps(verif::BoundedModelCheckingOp bmcOp) {
  SmallVector<llhd::CombinationalOp> combinationalOps;
  bmcOp.getCircuit().walk([&](llhd::CombinationalOp op) {
    combinationalOps.push_back(op);
  });

  for (auto op : combinationalOps) {
    if (!op.getBody().hasOneBlock()) {
      op.emitError("llhd.combinational with control flow must be flattened "
                   "before lower-to-bmc");
      return failure();
    }
    auto *parentBlock = op->getBlock();
    auto &bodyBlock = op.getBody().front();
    auto yieldOp = dyn_cast<llhd::YieldOp>(bodyBlock.getTerminator());
    if (!yieldOp) {
      op.emitError("llhd.combinational missing llhd.yield terminator");
      return failure();
    }

    SmallVector<Value> yieldedValues(yieldOp.getYieldOperands().begin(),
                                     yieldOp.getYieldOperands().end());
    yieldOp.erase();

    // Move the body ops into the parent block before the combinational op.
    parentBlock->getOperations().splice(op->getIterator(),
                                        bodyBlock.getOperations());

    // Replace results with the yielded values.
    for (auto [result, value] : llvm::zip(op.getResults(), yieldedValues))
      result.replaceAllUsesWith(value);

    op.erase();
  }

  for (Block &block : bmcOp.getCircuit()) {
    if (!sortTopologically(&block)) {
      bmcOp.emitError(
          "could not resolve cycles after inlining llhd.combinational");
      return failure();
    }
  }
  return success();
}

namespace {
struct ClockInputEquivalence {
  bool isEquivalent(Value lhs, Value rhs) {
    if (lhs == rhs)
      return true;
    if (lhs.getType() != rhs.getType())
      return false;

    auto key = std::make_pair(lhs, rhs);
    if (auto it = memo.find(key); it != memo.end())
      return it->second;

    // Default to non-equivalent to break potential cycles.
    memo[key] = false;

    auto *lhsOp = lhs.getDefiningOp();
    auto *rhsOp = rhs.getDefiningOp();
    if (!lhsOp || !rhsOp)
      return false;

    if (lhsOp == rhsOp)
      return false;

    if (lhsOp->getNumRegions() != 0 || rhsOp->getNumRegions() != 0)
      return false;

    if (lhsOp->getNumResults() != 1 || rhsOp->getNumResults() != 1)
      return false;

    if (!isMemoryEffectFree(lhsOp) || !isMemoryEffectFree(rhsOp))
      return false;

    auto eq = OperationEquivalence::isEquivalentTo(
        lhsOp, rhsOp,
        [&](Value a, Value b) {
          return isEquivalent(a, b) ? success() : failure();
        },
        /*markEquivalent=*/nullptr,
        OperationEquivalence::IgnoreLocations |
            OperationEquivalence::IgnoreDiscardableAttrs);

    memo[key] = eq;
    return eq;
  }

  DenseMap<std::pair<Value, Value>, bool> memo;
};
} // namespace

static LogicalResult lowerLlhdForBMC(verif::BoundedModelCheckingOp bmcOp) {
  // Hoist simple top-level drives so probes see current combinational inputs.
  if (!bmcOp.getCircuit().empty()) {
    Block &circuitBlock = bmcOp.getCircuit().front();
    SmallVector<llhd::DriveOp> hoistDrives;
    for (auto drive : circuitBlock.getOps<llhd::DriveOp>()) {
      if (drive.getEnable())
        continue;
      if (!isa<BlockArgument>(drive.getValue()))
        continue;
      if (!drive.getTime().getDefiningOp<llhd::ConstantTimeOp>())
        continue;
      auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
      if (!sigOp || sigOp->getBlock() != &circuitBlock)
        continue;
      hoistDrives.push_back(drive);
    }
    for (auto drive : hoistDrives) {
      auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
      if (sigOp)
        drive->moveAfter(sigOp);
    }
  }

  // Replace llhd.sig with a plain SSA value from its init. Probes are replaced
  // with that SSA value, and drives update the SSA through a mux on the enable.
  DenseMap<Value, Value> signalValue;
  DenseMap<Value, Value> signalEnable;
  SmallVector<llhd::SignalOp> signalOps;

  auto updateSignal = [&](llhd::SignalOp op, Value newVal, Value enable) {
    Value curVal = signalValue.lookup(op.getResult());
    if (!curVal)
      return;
    if (auto probe = newVal.getDefiningOp<llhd::ProbeOp>()) {
      if (probe.getSignal() == op.getResult())
        newVal = curVal;
    }
    OpBuilder builder(op);
    Value mergedVal = newVal;
    if (enable) {
      mergedVal = comb::MuxOp::create(builder, op.getLoc(), enable, newVal,
                                      curVal);
    }
    signalValue[op.getResult()] = mergedVal;
    signalEnable[op.getResult()] = enable;
  };

  auto processRegion =
      [&](Region &region, bool reorderDrives,
          auto &&processRegionRef) -> void {
    for (Block &block : region) {
      if (reorderDrives) {
        SmallVector<llhd::DriveOp> drives;
        SmallVector<llhd::ProbeOp> probes;
        SmallVector<std::pair<Region *, bool>, 4> nestedRegions;
        for (Operation &op : llvm::make_early_inc_range(block)) {
          if (auto ctime = dyn_cast<llhd::ConstantTimeOp>(op)) {
            OpBuilder builder(ctime);
            auto zero = hw::ConstantOp::create(builder, ctime.getLoc(),
                                               builder.getI1Type(), 0);
            ctime.replaceAllUsesWith(zero.getResult());
            ctime.erase();
            continue;
          }
          if (auto sig = dyn_cast<llhd::SignalOp>(op)) {
            signalOps.push_back(sig);
            signalValue[sig.getResult()] = sig.getInit();
            signalEnable[sig.getResult()] = nullptr;
            continue;
          }
          if (auto drive = dyn_cast<llhd::DriveOp>(op)) {
            drives.push_back(drive);
            continue;
          }
          if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
            probes.push_back(probe);
            continue;
          }
          bool childReorder = reorderDrives;
          if (isa<llhd::ProcessOp>(op))
            childReorder = false;
          for (Region &nested : op.getRegions())
            nestedRegions.push_back({&nested, childReorder});
        }

        for (auto drive : drives) {
          auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
          if (sigOp) {
            Value enable = drive.getEnable();
            updateSignal(sigOp, drive.getValue(), enable);
          }
          drive.erase();
        }

        for (auto probe : probes) {
          auto sigOp = probe.getSignal().getDefiningOp<llhd::SignalOp>();
          if (sigOp) {
            Value curVal = signalValue.lookup(sigOp.getResult());
            if (curVal) {
              if (curVal == probe.getResult()) {
                if (probe.use_empty())
                  probe.erase();
                continue;
              }
              probe.replaceAllUsesWith(curVal);
              for (auto &it : signalValue) {
                if (it.second == probe.getResult())
                  it.second = curVal;
              }
              for (auto &it : signalEnable) {
                if (it.second == probe.getResult())
                  it.second = curVal;
              }
              probe.erase();
              continue;
            }
          }
        }

        // Re-sort after replacements to restore dominance.
        mlir::sortTopologically(&block);

        for (auto [nested, childReorder] : nestedRegions)
          processRegionRef(*nested, childReorder, processRegionRef);
        continue;
      }

      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto ctime = dyn_cast<llhd::ConstantTimeOp>(op)) {
          OpBuilder builder(ctime);
          auto zero = hw::ConstantOp::create(builder, ctime.getLoc(),
                                             builder.getI1Type(), 0);
          ctime.replaceAllUsesWith(zero.getResult());
          ctime.erase();
          continue;
        }
        if (auto sig = dyn_cast<llhd::SignalOp>(op)) {
          signalOps.push_back(sig);
          signalValue[sig.getResult()] = sig.getInit();
          signalEnable[sig.getResult()] = nullptr;
          continue;
        }
        if (auto drive = dyn_cast<llhd::DriveOp>(op)) {
          auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
          if (sigOp) {
            Value enable = drive.getEnable();
            updateSignal(sigOp, drive.getValue(), enable);
          }
          drive.erase();
          continue;
        }
        if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
          auto sigOp = probe.getSignal().getDefiningOp<llhd::SignalOp>();
          if (sigOp) {
            Value curVal = signalValue.lookup(sigOp.getResult());
            if (curVal) {
              if (curVal == probe.getResult()) {
                if (probe.use_empty())
                  probe.erase();
                continue;
              }
              probe.replaceAllUsesWith(curVal);
              probe.erase();
              continue;
            }
          }
        }
        for (Region &nested : op.getRegions())
          processRegionRef(nested, /*reorderDrives=*/false, processRegionRef);
      }
    }
  };

  processRegion(bmcOp.getCircuit(), /*reorderDrives=*/true, processRegion);

  for (auto op : signalOps) {
    if (!op.use_empty())
      continue;
    op.erase();
  }

  return success();
}

void LowerToBMCPass::runOnOperation() {
  Namespace names;
  // Fetch the 'hw.module' operation to model check.
  auto moduleOp = getOperation();
  auto hwModule = moduleOp.lookupSymbol<hw::HWModuleOp>(topModule);
  if (!hwModule) {
    moduleOp.emitError("hw.module named '") << topModule << "' not found";
    return signalPassFailure();
  }

  if (!sortTopologically(&hwModule.getBodyRegion().front())) {
    hwModule->emitError("could not resolve cycles in module");
    return signalPassFailure();
  }

  if (bound < ignoreAssertionsUntil) {
    hwModule->emitError(
        "number of ignored cycles must be less than or equal to bound");
    return signalPassFailure();
  }

  // Create necessary function declarations and globals
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp->getLoc();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(builder, moduleOp, "printf", ptrTy, voidTy, true);
  if (failed(printfFunc)) {
    moduleOp->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Replace the top-module with a function performing the BMC
  auto entryFunc = func::FuncOp::create(builder, loc, topModule,
                                        builder.getFunctionType({}, {}));
  builder.createBlock(&entryFunc.getBody());

  {
    OpBuilder::InsertionGuard guard(builder);
    auto *terminator = hwModule.getBody().front().getTerminator();
    builder.setInsertionPoint(terminator);
    verif::YieldOp::create(builder, loc, terminator->getOperands());
    terminator->erase();
  }

  auto numRegs = hwModule->getAttrOfType<IntegerAttr>("num_regs");
  auto initialValues = hwModule->getAttrOfType<ArrayAttr>("initial_values");
  if (numRegs && initialValues) {
    for (auto value : initialValues) {
      if (!isa<IntegerAttr, UnitAttr>(value)) {
        hwModule->emitError("initial_values attribute must contain only "
                            "integer or unit attributes");
        return signalPassFailure();
      }
    }
  } else {
    hwModule->emitOpError("no num_regs or initial_values attribute found - "
                          "please run externalize "
                          "registers pass first");
    return signalPassFailure();
  }

  auto collectClockArgs = [&]() {
    SmallVector<BlockArgument, 4> clocks;
    auto &entryBlock = hwModule.getBody().front();
    auto inputTypes = hwModule.getInputTypes();
    for (auto [idx, input] : llvm::enumerate(inputTypes)) {
      if (isa<seq::ClockType>(input))
        clocks.push_back(entryBlock.getArgument(idx));
    }
    return clocks;
  };

  bool structClockFound = false;
  SmallVector<BlockArgument, 4> explicitClocks;
  {
    auto &entryBlock = hwModule.getBody().front();
    auto inputTypes = hwModule.getInputTypes();
    for (auto [idx, input] : llvm::enumerate(inputTypes)) {
      if (isa<seq::ClockType>(input)) {
        explicitClocks.push_back(entryBlock.getArgument(idx));
        continue;
      }
      if (auto hwStruct = dyn_cast<hw::StructType>(input)) {
        for (auto field : hwStruct.getElements()) {
          if (isa<seq::ClockType>(field.type)) {
            structClockFound = true;
            break;
          }
        }
      }
    }
  }

  if (structClockFound) {
    if (!allowMultiClock) {
      hwModule.emitError("designs with multiple clocks not yet supported");
      return signalPassFailure();
    }
    hwModule.emitError(
        "clock inputs inside struct types are not yet supported");
    return signalPassFailure();
  }

  if (!allowMultiClock && explicitClocks.size() > 1) {
    hwModule.emitError("designs with multiple clocks not yet supported");
    return signalPassFailure();
  }

  bool hasExplicitClockInput = !explicitClocks.empty();
  bool hasClk = hasExplicitClockInput;
  if (!hasExplicitClockInput) {
    SmallVector<seq::ToClockOp> toClockOps;
    hwModule.walk([&](seq::ToClockOp toClockOp) {
      toClockOps.push_back(toClockOp);
    });
    if (!toClockOps.empty()) {
      SmallVector<Value> clockInputs;
      ClockInputEquivalence equivalence;
      clockInputs.reserve(toClockOps.size());
      for (auto toClockOp : toClockOps) {
        auto input = toClockOp.getInput();
        // Graph regions can use values before they are defined, which prevents
        // CSE from collapsing equivalent clock expressions. Deduplicate
        // structurally equivalent inputs here to avoid spurious multi-clock
        // errors.
        if (llvm::any_of(clockInputs, [&](Value existing) {
              return equivalence.isEquivalent(input, existing);
            }))
          continue;
        clockInputs.push_back(input);
      }

      auto lookupClockInputIndex = [&](Value input) -> std::optional<size_t> {
        for (auto [idx, existing] : llvm::enumerate(clockInputs)) {
          if (equivalence.isEquivalent(input, existing))
            return idx;
        }
        return std::nullopt;
      };

      if (!allowMultiClock && clockInputs.size() > 1) {
        hwModule.emitError("designs with multiple clocks not yet supported");
        return signalPassFailure();
      }

      auto clockTy = seq::ClockType::get(ctx);
      SmallVector<BlockArgument, 4> newClocks(clockInputs.size());
      for (size_t idx = clockInputs.size(); idx-- > 0;) {
        auto name = idx == 0 ? "bmc_clock"
                             : ("bmc_clock_" + Twine(idx)).str();
        newClocks[idx] = hwModule.prependInput(name, clockTy).second;
      }

      // Constrain each derived clock input to match its BMC clock.
      for (auto [idx, clockInput] : llvm::enumerate(clockInputs)) {
        OpBuilder::InsertionGuard guard(builder);
        if (auto *def = clockInput.getDefiningOp()) {
          builder.setInsertionPointAfter(def);
        } else {
          builder.setInsertionPointToStart(clockInput.getParentBlock());
        }
        auto fromClk =
            seq::FromClockOp::create(builder, loc, newClocks[idx]);
        auto eq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                       fromClk, clockInput);
        verif::AssumeOp::create(builder, loc, eq, Value(), StringAttr());
      }

      for (auto toClockOp : toClockOps) {
        auto idx = lookupClockInputIndex(toClockOp.getInput());
        if (!idx) {
          toClockOp.emitError("failed to map derived clock input");
          return signalPassFailure();
        }
        toClockOp.replaceAllUsesWith(newClocks[*idx]);
        toClockOp.erase();
      }
      hasClk = true;
    }
  }
  // Also check for i1 inputs that are converted to clocks via ToClockOp
  if (!hasClk) {
    hwModule.walk([&](seq::ToClockOp toClockOp) {
      if (auto blockArg = dyn_cast<BlockArgument>(toClockOp.getInput())) {
        if (blockArg.getOwner() == &hwModule.getBody().front()) {
          hasClk = true;
        }
      }
    });
  }

  auto clockArgs = collectClockArgs();
  unsigned clockCount = clockArgs.size();
  bool multiClock = allowMultiClock && clockCount > 1;
  unsigned clockScale = multiClock ? clockCount : 1;

  // Scale bounds for multi-clock interleaving so each clock observes the
  // requested number of cycles.
  unsigned effectiveBound =
      risingClocksOnly ? bound * clockScale : 2 * bound * clockScale;

  verif::BoundedModelCheckingOp bmcOp =
      verif::BoundedModelCheckingOp::create(
          builder, loc, effectiveBound,
          cast<IntegerAttr>(numRegs).getValue().getZExtValue(), initialValues);
  auto inputNames = hwModule.getInputNames();
  if (!inputNames.empty())
    bmcOp->setAttr("bmc_input_names", builder.getArrayAttr(inputNames));
  if (auto regClocks = hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks"))
    bmcOp->setAttr("bmc_reg_clocks", regClocks);
  if (ignoreAssertionsUntil) {
    unsigned scaledIgnore =
        risingClocksOnly ? ignoreAssertionsUntil * clockScale
                         : 2 * ignoreAssertionsUntil * clockScale;
    bmcOp->setAttr("ignore_asserts_until",
                   builder.getI32IntegerAttr(scaledIgnore));
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    // Initialize clock to 0 if it exists, otherwise just yield nothing
    // We initialize to 1 if we're in rising clocks only mode
    auto *initBlock = builder.createBlock(&bmcOp.getInit());
    builder.setInsertionPointToStart(initBlock);
    if (hasClk) {
      auto initVal = hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                            risingClocksOnly ? 1 : 0);
      SmallVector<Value> yields;
      yields.reserve(clockCount + 1);
      for (unsigned idx = 0; idx < clockCount; ++idx)
        yields.push_back(seq::ToClockOp::create(builder, loc, initVal));
      if (multiClock && !risingClocksOnly) {
        auto phaseInit =
            hw::ConstantOp::create(builder, loc, builder.getI32Type(), 0);
        yields.push_back(phaseInit);
      }
      verif::YieldOp::create(builder, loc, yields);
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }

    // Toggle clock in loop region if it exists, otherwise just yield nothing
    auto *loopBlock = builder.createBlock(&bmcOp.getLoop());
    builder.setInsertionPointToStart(loopBlock);
    if (hasClk) {
      for (unsigned idx = 0; idx < clockCount; ++idx)
        loopBlock->addArgument(seq::ClockType::get(ctx), loc);
      BlockArgument phaseArg;
      if (multiClock && !risingClocksOnly)
        phaseArg = loopBlock->addArgument(builder.getI32Type(), loc);

      if (risingClocksOnly) {
        // In rising clocks only mode we don't need to toggle the clock
        SmallVector<Value> yields;
        yields.reserve(clockCount);
        for (unsigned idx = 0; idx < clockCount; ++idx)
          yields.push_back(loopBlock->getArgument(idx));
        verif::YieldOp::create(builder, loc, yields);
      } else {
        auto cNeg1 =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), -1);
        SmallVector<Value> yields;
        yields.reserve(clockCount + 1);
        for (unsigned idx = 0; idx < clockCount; ++idx) {
          auto fromClk = seq::FromClockOp::create(builder, loc,
                                                  loopBlock->getArgument(idx));
          auto nClk = comb::XorOp::create(builder, loc, fromClk, cNeg1);
          Value nextVal = nClk;
          if (multiClock) {
            auto idxConst = hw::ConstantOp::create(builder, loc,
                                                   builder.getI32Type(), idx);
            auto eq = comb::ICmpOp::create(builder, loc,
                                           comb::ICmpPredicate::eq, phaseArg,
                                           idxConst);
            nextVal = comb::MuxOp::create(builder, loc, fromClk.getType(), eq,
                                          nClk, fromClk);
          }
          yields.push_back(seq::ToClockOp::create(builder, loc, nextVal));
        }
        if (multiClock) {
          auto one = hw::ConstantOp::create(builder, loc, builder.getI32Type(),
                                            1);
          auto zero = hw::ConstantOp::create(builder, loc, builder.getI32Type(),
                                             0);
          auto max = hw::ConstantOp::create(
              builder, loc, builder.getI32Type(), clockCount - 1);
          auto eqMax = comb::ICmpOp::create(builder, loc,
                                            comb::ICmpPredicate::eq, phaseArg,
                                            max);
          auto nextPhase =
              comb::AddOp::create(builder, loc, phaseArg, one, false);
          auto phase =
              comb::MuxOp::create(builder, loc, builder.getI32Type(), eqMax,
                                  zero, nextPhase);
          yields.push_back(phase);
        }
        verif::YieldOp::create(builder, loc, yields);
      }
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }
  }
  bmcOp.getCircuit().takeBody(hwModule.getBody());
  hwModule->erase();

  // Convert ref-typed circuit outputs to value-typed yields by probing them.
  bmcOp.getCircuit().walk([&](verif::YieldOp yieldOp) {
    bool needsRewrite = false;
    for (auto operand : yieldOp.getOperands()) {
      if (isa<llhd::RefType>(operand.getType())) {
        needsRewrite = true;
        break;
      }
    }
    if (!needsRewrite)
      return;
    OpBuilder builder(yieldOp);
    SmallVector<Value> newOperands;
    newOperands.reserve(yieldOp.getNumOperands());
    for (auto operand : yieldOp.getOperands()) {
      if (isa<llhd::RefType>(operand.getType())) {
        auto probe =
            llhd::ProbeOp::create(builder, yieldOp.getLoc(), operand);
        newOperands.push_back(probe.getResult());
      } else {
        newOperands.push_back(operand);
      }
    }
    yieldOp->setOperands(newOperands);
  });

  if (failed(lowerLlhdForBMC(bmcOp)))
    return signalPassFailure();
  if (failed(inlineLlhdCombinationalOps(bmcOp)))
    return signalPassFailure();

  // Define global string constants to print on success/failure
  auto createUniqueStringGlobal = [&](StringRef str) -> FailureOr<Value> {
    Location loc = moduleOp.getLoc();

    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        "resultString",
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
    SymbolTable symTable(moduleOp);
    if (failed(symTable.renameToUnique(global, {&symTable}))) {
      return mlir::failure();
    }

    return success(
        LLVM::AddressOfOp::create(builder, loc, global)->getResult(0));
  };

  auto successStrAddr =
      createUniqueStringGlobal("Bound reached with no violations!\n");
  auto failureStrAddr =
      createUniqueStringGlobal("Assertion can be violated!\n");

  if (failed(successStrAddr) || failed(failureStrAddr)) {
    moduleOp->emitOpError("could not create result message strings");
    return signalPassFailure();
  }

  auto formatString =
      LLVM::SelectOp::create(builder, loc, bmcOp.getResult(),
                             successStrAddr.value(), failureStrAddr.value());

  LLVM::CallOp::create(builder, loc, printfFunc.value(),
                       ValueRange{formatString});
  func::ReturnOp::create(builder, loc);

  if (insertMainFunc) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    Type i32Ty = builder.getI32Type();
    auto mainFunc = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    func::CallOp::create(builder, loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
    func::ReturnOp::create(builder, loc, constZero);
  }
}
