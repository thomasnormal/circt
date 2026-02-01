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
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/CommutativeValueEquivalence.h"
#include "circt/Support/I1ValueSimplifier.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

static LogicalResult lowerLlhdForBMC(verif::BoundedModelCheckingOp bmcOp) {
  // Hoist simple top-level drives so probes see current combinational inputs.
  if (!bmcOp.getCircuit().empty()) {
    Block &circuitBlock = bmcOp.getCircuit().front();
    auto isHoistableValue = [](Value value) {
      return isa<BlockArgument>(value) ||
             value.getDefiningOp<hw::ConstantOp>() ||
             value.getDefiningOp<arith::ConstantOp>() ||
             value.getDefiningOp<hw::AggregateConstantOp>();
    };
    DenseMap<Operation *, SmallVector<llhd::DriveOp>> drivesBySignal;
    for (auto drive : circuitBlock.getOps<llhd::DriveOp>()) {
      if (drive.getEnable())
        continue;
      if (!drive.getTime().getDefiningOp<llhd::ConstantTimeOp>())
        continue;
      auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
      if (!sigOp || sigOp->getBlock() != &circuitBlock)
        continue;
      drivesBySignal[sigOp].push_back(drive);
    }
    SmallVector<llhd::DriveOp> hoistDrives;
    for (auto &entry : drivesBySignal) {
      if (entry.second.size() != 1)
        continue;
      auto drive = entry.second.front();
      if (!isHoistableValue(drive.getValue()))
        continue;
      hoistDrives.push_back(drive);
    }
    DenseMap<Operation *, Operation *> lastAfter;
    for (auto drive : hoistDrives) {
      auto sigOp = drive.getSignal().getDefiningOp<llhd::SignalOp>();
      if (!sigOp)
        continue;
      Operation *insertAfter = sigOp;
      if (auto it = lastAfter.find(sigOp); it != lastAfter.end())
        insertAfter = it->second;
      drive->moveAfter(insertAfter);
      lastAfter[sigOp] = drive;
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
  auto reportFunc = LLVM::lookupOrCreateFn(builder, moduleOp,
                                           "circt_bmc_report_result",
                                           {builder.getI1Type()}, voidTy);
  if (failed(reportFunc)) {
    moduleOp->emitError("failed to lookup or create circt_bmc_report_result");
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
  SmallVector<Attribute> clockSourceAttrs;
  SmallVector<Attribute> clockKeyAttrs;
  if (!hasExplicitClockInput) {
    SmallVector<seq::ToClockOp> toClockOps;
    SmallVector<ltl::ClockOp> ltlClockOps;
    hwModule.walk([&](seq::ToClockOp toClockOp) {
      toClockOps.push_back(toClockOp);
    });
    hwModule.walk([&](ltl::ClockOp clockOp) { ltlClockOps.push_back(clockOp); });
    struct ClockInputInfo {
      Value value;
      Value canonical;
      bool invert = false;
      Value fourStateBase;
    };
    SmallVector<ClockInputInfo> clockInputs;
    CommutativeValueEquivalence equivalence;
    clockInputs.reserve(toClockOps.size() + ltlClockOps.size());
    auto canonicalizeClockValue = [&](Value value, bool &invert) -> Value {
      auto simplified = simplifyI1Value(value);
      invert = simplified.invert;
      Value canonical = simplified.value ? simplified.value : value;
      if (auto andOp = canonical.getDefiningOp<comb::AndOp>()) {
        if (andOp.getNumOperands() == 2) {
          if (auto gated =
                  matchFourStateClockGate(andOp.getOperand(0),
                                          andOp.getOperand(1)))
            canonical = gated;
          else if (auto gated =
                       matchFourStateClockGate(andOp.getOperand(1),
                                               andOp.getOperand(0)))
            canonical = gated;
        }
      }
      return canonical;
    };
    auto getFourStateClockBase = [&](Value input) -> Value {
      if (!input)
        return Value();
      if (auto extract = input.getDefiningOp<hw::StructExtractOp>()) {
        if (auto fieldAttr = extract.getFieldNameAttr()) {
          if (fieldAttr.getValue() == "value" &&
              isFourStateStructType(extract.getInput().getType()))
            return extract.getInput();
        }
      }
      if (auto andOp = input.getDefiningOp<comb::AndOp>()) {
        if (andOp.getNumOperands() == 2) {
          if (auto gated = matchFourStateClockGate(andOp.getOperand(0),
                                                   andOp.getOperand(1))) {
            return getStructFieldBase(gated, "value");
          }
          if (auto gated = matchFourStateClockGate(andOp.getOperand(1),
                                                   andOp.getOperand(0))) {
            return getStructFieldBase(gated, "value");
          }
        }
      }
      return Value();
    };

    auto maybeAddClockInput = [&](Value input) {
      if (!input || input.getType() != builder.getI1Type())
        return;
      bool invert = false;
      Value canonical = canonicalizeClockValue(input, invert);
      Value base = getFourStateClockBase(canonical);
      if (!base)
        base = getFourStateClockBase(input);
      // Graph regions can use values before they are defined, which prevents
      // CSE from collapsing equivalent clock expressions. Deduplicate
      // structurally equivalent inputs here to avoid spurious multi-clock
      // errors.
      if (llvm::any_of(clockInputs, [&](const ClockInputInfo &existing) {
            return invert == existing.invert &&
                   (equivalence.isEquivalent(canonical, existing.canonical) ||
                    (base && existing.fourStateBase &&
                     base == existing.fourStateBase));
          }))
        return;
      clockInputs.push_back({input, canonical, invert, base});
    };
    for (auto toClockOp : toClockOps)
      maybeAddClockInput(toClockOp.getInput());
    for (auto clockOp : ltlClockOps)
      maybeAddClockInput(clockOp.getClock());
    if (clockInputs.empty()) {
      if (auto regClocks =
              hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks")) {
        DenseSet<StringRef> seenNames;
        auto inputNames = hwModule.getInputNames();
        Block &entryBlock = hwModule.getBody().front();
        Value constTrue;
        auto ensureConstTrue = [&]() -> Value {
          if (constTrue)
            return constTrue;
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&entryBlock);
          constTrue =
              hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
          return constTrue;
        };
        for (auto attr : regClocks) {
          auto nameAttr = dyn_cast<StringAttr>(attr);
          if (!nameAttr || nameAttr.getValue().empty())
            continue;
          if (!seenNames.insert(nameAttr.getValue()).second)
            continue;
          std::optional<unsigned> inputIdx;
          for (auto [idx, name] : llvm::enumerate(inputNames)) {
            auto str = dyn_cast<StringAttr>(name);
            if (str && str.getValue() == nameAttr.getValue()) {
              inputIdx = idx;
              break;
            }
          }
          if (!inputIdx)
            continue;
          Value arg = entryBlock.getArgument(*inputIdx);
          Value clockVal;
          if (auto structTy = dyn_cast<hw::StructType>(arg.getType())) {
            auto valueIdx = structTy.getFieldIndex("value");
            auto unknownIdx = structTy.getFieldIndex("unknown");
            if (!valueIdx || !unknownIdx)
              continue;
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&entryBlock);
            Value value =
                hw::StructExtractOp::create(builder, loc, arg, "value");
            Value unknown =
                hw::StructExtractOp::create(builder, loc, arg, "unknown");
            Value notUnknown = comb::XorOp::create(
                                   builder, loc, unknown, ensureConstTrue())
                                   .getResult();
            clockVal = comb::AndOp::create(builder, loc, value, notUnknown)
                           .getResult();
          } else if (auto intTy = dyn_cast<IntegerType>(arg.getType());
                     intTy && intTy.getWidth() == 1) {
            clockVal = arg;
          } else {
            continue;
          }
          maybeAddClockInput(clockVal);
        }
      }
    }

    if (!clockInputs.empty()) {
      auto traceRoot = [&](Value value, BlockArgument &root) -> bool {
        return traceI1ValueRoot(value, root);
      };

      auto resolveClockInputName =
          [&](const ClockInputInfo &input) -> std::string {
        BlockArgument root;
        if (!traceRoot(input.canonical, root) &&
            !traceRoot(input.value, root))
          return {};
        if (root.getOwner() != hwModule.getBodyBlock())
          return {};
        auto inputNames = hwModule.getInputNames();
        if (root.getArgNumber() >= inputNames.size())
          return {};
        if (auto nameAttr =
                dyn_cast<StringAttr>(inputNames[root.getArgNumber()]))
          return nameAttr.getValue().str();
        return {};
      };

      auto makeClockKey = [&](const ClockInputInfo &input) -> StringAttr {
        BlockArgument root;
        if (!(traceRoot(input.canonical, root) ||
              traceRoot(input.value, root)))
          return StringAttr{};
        if (root.getOwner() != hwModule.getBodyBlock())
          return StringAttr{};
        std::string key =
            ("arg" + Twine(root.getArgNumber())).str();
        if (input.invert)
          key.append(":inv");
        return builder.getStringAttr(key);
      };

      {
        DenseMap<std::pair<BlockArgument, unsigned>, size_t> rootToIndex;
        SmallVector<ClockInputInfo> deduped;
        deduped.reserve(clockInputs.size());
        for (const auto &input : clockInputs) {
          BlockArgument root;
          if ((traceRoot(input.canonical, root) ||
               traceRoot(input.value, root)) &&
              root && root.getOwner() == hwModule.getBodyBlock()) {
            auto key = std::make_pair(root, input.invert ? 1u : 0u);
            if (rootToIndex.contains(key))
              continue;
            rootToIndex.try_emplace(key, deduped.size());
          }
          deduped.push_back(input);
        }
        clockInputs.swap(deduped);
      }

      {
        if (clockInputs.size() > 1) {
          BlockArgument firstRoot;
          bool firstInvert = clockInputs.front().invert;
          if (traceRoot(clockInputs.front().canonical, firstRoot) ||
              traceRoot(clockInputs.front().value, firstRoot)) {
            bool allSame = true;
            for (const auto &input : clockInputs) {
              if (input.invert != firstInvert) {
                allSame = false;
                break;
              }
              BlockArgument root;
              if (!(traceRoot(input.canonical, root) ||
                    traceRoot(input.value, root)) ||
                  root != firstRoot) {
                allSame = false;
                break;
              }
            }
            if (allSame)
              clockInputs.resize(1);
          }
        }
      }

      auto lookupClockInputIndex = [&](Value input) -> std::optional<size_t> {
        bool invert = false;
        Value canonical = canonicalizeClockValue(input, invert);
        Value base = getFourStateClockBase(canonical);
        if (!base)
          base = getFourStateClockBase(input);
        for (auto [idx, existing] : llvm::enumerate(clockInputs)) {
          if (invert != existing.invert)
            continue;
          if (equivalence.isEquivalent(canonical, existing.canonical) ||
              (base && existing.fourStateBase &&
               base == existing.fourStateBase))
            return idx;
        }
        return std::nullopt;
      };

      if (!allowMultiClock && clockInputs.size() > 1) {
        hwModule.emitError("designs with multiple clocks not yet supported");
        return signalPassFailure();
      }

      SmallVector<std::string> clockNames;
      clockNames.reserve(clockInputs.size());
      for (const auto &input : clockInputs)
        clockNames.push_back(resolveClockInputName(input));

      DenseSet<StringRef> usedClockNames;
      auto chooseClockName = [&](size_t idx) -> std::string {
        std::string name;
        if (idx < clockNames.size())
          name = clockNames[idx];
        if (name.empty())
          name = idx == 0 ? "bmc_clock" : ("bmc_clock_" + Twine(idx)).str();
        if (usedClockNames.contains(name)) {
          name = ("bmc_clock_" + Twine(idx)).str();
          unsigned suffix = static_cast<unsigned>(idx);
          while (usedClockNames.contains(name))
            name = ("bmc_clock_" + Twine(++suffix)).str();
        }
        usedClockNames.insert(name);
        return name;
      };

      auto clockTy = seq::ClockType::get(ctx);
      SmallVector<BlockArgument, 4> newClocks(clockInputs.size());
      for (size_t idx = clockInputs.size(); idx-- > 0;) {
        auto name = chooseClockName(idx);
        newClocks[idx] = hwModule.prependInput(name, clockTy).second;
      }
      auto inputNamesAfter = hwModule.getInputNames();
      SmallVector<StringAttr> actualClockNames;
      actualClockNames.reserve(clockInputs.size());
      for (auto arg : newClocks) {
        auto argNum = arg.getArgNumber();
        if (argNum < inputNamesAfter.size()) {
          if (auto nameAttr = dyn_cast<StringAttr>(inputNamesAfter[argNum]))
            actualClockNames.push_back(nameAttr);
          else
            actualClockNames.push_back(StringAttr{});
        } else {
          actualClockNames.push_back(StringAttr{});
        }
      }
      DenseMap<StringAttr, StringAttr> clockNameRemap;
      for (size_t idx = 0; idx < clockInputs.size(); ++idx) {
        if (idx >= clockNames.size())
          continue;
        if (clockNames[idx].empty())
          continue;
        auto origName = builder.getStringAttr(clockNames[idx]);
        auto actualName = idx < actualClockNames.size() ? actualClockNames[idx]
                                                        : StringAttr{};
        if (!actualName || actualName.getValue().empty())
          continue;
        if (origName != actualName)
          clockNameRemap.try_emplace(origName, actualName);
      }

      DenseSet<StringRef> validClockNames;
      for (auto nameAttr : actualClockNames) {
        if (nameAttr && !nameAttr.getValue().empty())
          validClockNames.insert(nameAttr.getValue());
      }

      clockKeyAttrs.clear();
      clockKeyAttrs.reserve(clockInputs.size());
      for (const auto &input : clockInputs)
        clockKeyAttrs.push_back(makeClockKey(input));

      if (!clockInputs.empty()) {
        DenseSet<unsigned> seenSourceArgs;
        for (auto [idx, clockInput] : llvm::enumerate(clockInputs)) {
          BlockArgument root;
          if (!traceRoot(clockInput.canonical, root) &&
              !traceRoot(clockInput.value, root))
            continue;
          if (root.getOwner() != hwModule.getBodyBlock())
            continue;
          unsigned argIndex = root.getArgNumber();
          if (!seenSourceArgs.insert(argIndex).second)
            continue;
          bool invert = clockInput.invert;
          auto dict = builder.getDictionaryAttr(
              {builder.getNamedAttr(
                   "arg_index", builder.getI32IntegerAttr(argIndex)),
               builder.getNamedAttr(
                   "clock_pos",
                   builder.getI32IntegerAttr(static_cast<unsigned>(idx))),
               builder.getNamedAttr("invert",
                                    builder.getBoolAttr(invert))});
          clockSourceAttrs.push_back(dict);
        }
      }

      // Constrain each derived clock input to match its BMC clock.
      for (auto [idx, clockInput] : llvm::enumerate(clockInputs)) {
        OpBuilder::InsertionGuard guard(builder);
        if (auto *def = clockInput.value.getDefiningOp()) {
          builder.setInsertionPointAfter(def);
        } else {
          builder.setInsertionPointToStart(clockInput.value.getParentBlock());
        }
        auto fromClk =
            seq::FromClockOp::create(builder, loc, newClocks[idx]);
        auto eq = comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::eq,
                                       fromClk, clockInput.value);
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

      // Rewrite ltl.clock operands to reference the corresponding BMC clock
      // input. This avoids treating structurally equivalent clock expressions
      // as unrelated to the inserted clock ports.
      for (auto clockOp : ltlClockOps) {
        auto idx = lookupClockInputIndex(clockOp.getClock());
        if (!idx) {
          clockOp.emitError("failed to map clocked property input");
          return signalPassFailure();
        }
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(clockOp);
        auto fromClk =
            seq::FromClockOp::create(builder, loc, newClocks[*idx]);
        auto rebuilt = ltl::ClockOp::create(builder, loc, clockOp.getInput(),
                                            clockOp.getEdge(), fromClk);
        clockOp.replaceAllUsesWith(rebuilt.getResult());
        clockOp.erase();
      }
      if (!clockNameRemap.empty()) {
        if (auto regClocks =
                hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks")) {
          SmallVector<Attribute> remapped;
          remapped.reserve(regClocks.size());
          for (auto attr : regClocks) {
            auto nameAttr = dyn_cast<StringAttr>(attr);
            if (!nameAttr) {
              remapped.push_back(attr);
              continue;
            }
            if (auto it = clockNameRemap.find(nameAttr);
                it != clockNameRemap.end())
              remapped.push_back(it->second);
            else
              remapped.push_back(nameAttr);
          }
          hwModule->setAttr("bmc_reg_clocks",
                            ArrayAttr::get(ctx, remapped));
        }
        hwModule.walk([&](Operation *op) {
          if (!isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp>(op))
            return;
          auto nameAttr = op->getAttrOfType<StringAttr>("bmc.clock");
          if (!nameAttr)
            return;
          auto it = clockNameRemap.find(nameAttr);
          if (it == clockNameRemap.end())
            return;
          op->setAttr("bmc.clock", it->second);
        });
      }
      // If there is a single derived BMC clock input, remap any explicit
      // bmc.clock attributes that don't match an inserted clock name to the
      // new clock. This avoids rejecting clocked properties when the clock
      // expression cannot be traced back to a named module input.
      if (newClocks.size() == 1 && !actualClockNames.empty()) {
        auto defaultClockName = actualClockNames.front();
        if (defaultClockName && !defaultClockName.getValue().empty()) {
          auto remapIfUnknown = [&](StringAttr nameAttr) -> StringAttr {
            if (!nameAttr || nameAttr.getValue().empty())
              return nameAttr;
            if (validClockNames.contains(nameAttr.getValue()))
              return nameAttr;
            return defaultClockName;
          };
          if (auto regClocks =
                  hwModule->getAttrOfType<ArrayAttr>("bmc_reg_clocks")) {
            SmallVector<Attribute> remapped;
            remapped.reserve(regClocks.size());
            for (auto attr : regClocks) {
              auto nameAttr = dyn_cast<StringAttr>(attr);
              if (!nameAttr) {
                remapped.push_back(attr);
                continue;
              }
              auto updated = remapIfUnknown(nameAttr);
              remapped.push_back(updated ? updated : nameAttr);
            }
            hwModule->setAttr("bmc_reg_clocks",
                              ArrayAttr::get(ctx, remapped));
          }
          hwModule.walk([&](Operation *op) {
            if (!isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp>(op))
              return;
            auto nameAttr = op->getAttrOfType<StringAttr>("bmc.clock");
            if (!nameAttr) {
              // If a clocked check lacks a resolved clock name, map it to the
              // single derived BMC clock input to avoid rejecting it later.
              if (op->getAttr("bmc.clock_edge"))
                op->setAttr("bmc.clock", defaultClockName);
              return;
            }
            auto updated = remapIfUnknown(nameAttr);
            if (updated && updated != nameAttr)
              op->setAttr("bmc.clock", updated);
          });
        }
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
  if (!clockSourceAttrs.empty())
    bmcOp->setAttr("bmc_clock_sources",
                   builder.getArrayAttr(clockSourceAttrs));
  if (!clockKeyAttrs.empty())
    bmcOp->setAttr("bmc_clock_keys", builder.getArrayAttr(clockKeyAttrs));
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
  // Convert comb.mux producing LLVM types to llvm.select so SMT lowering
  // doesn't reject non-SMT-compatible muxes (e.g. UVM string helpers).
  SmallVector<comb::MuxOp> muxesToReplace;
  moduleOp.walk([&](comb::MuxOp muxOp) {
    bool inBmc = muxOp->getParentOfType<verif::BoundedModelCheckingOp>();
    if (inBmc) {
      if (isa<LLVM::LLVMStructType, LLVM::LLVMPointerType>(muxOp.getType()))
        muxesToReplace.push_back(muxOp);
      return;
    }
    if (muxOp->getParentOfType<hw::HWModuleOp>())
      return;
    if (LLVM::isCompatibleType(muxOp.getType()))
      muxesToReplace.push_back(muxOp);
  });
  for (auto muxOp : muxesToReplace) {
    OpBuilder muxBuilder(muxOp);
    auto select = LLVM::SelectOp::create(
        muxBuilder, muxOp.getLoc(), muxOp.getCond(), muxOp.getTrueValue(),
        muxOp.getFalseValue());
    muxOp.replaceAllUsesWith(select.getResult());
    muxOp.erase();
  }

  // Lower LLHD signal operations in non-HW contexts to LLVM memory ops so
  // downstream SMT/LLVM lowering doesn't trip on mixed LLHD usage.
  SmallVector<llhd::SignalOp> signalsToLower;
  moduleOp.walk([&](llhd::SignalOp sigOp) {
    if (sigOp->getParentOfType<hw::HWModuleOp>())
      return;
    if (sigOp->getParentOfType<llhd::ProcessOp>())
      return;
    signalsToLower.push_back(sigOp);
  });

  auto getEntryBlock = [](Operation *op) -> Block * {
    if (auto func = op->getParentOfType<func::FuncOp>())
      return func.getBody().empty() ? nullptr : &func.getBody().front();
    if (auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>())
      return llvmFunc.getBody().empty() ? nullptr : &llvmFunc.getBody().front();
    return nullptr;
  };

  for (auto sigOp : signalsToLower) {
    auto refType = cast<llhd::RefType>(sigOp.getResult().getType());
    Type nestedType = refType.getNestedType();
    if (!LLVM::isCompatibleType(nestedType))
      continue;

    bool hasUnsupportedUse = false;
    bool hasEnable = false;
    SmallVector<llhd::DriveOp> drives;
    SmallVector<llhd::ProbeOp> probes;
    for (auto *user : sigOp.getResult().getUsers()) {
      if (auto drive = dyn_cast<llhd::DriveOp>(user)) {
        drives.push_back(drive);
        if (drive.getEnable())
          hasEnable = true;
      } else if (auto probe = dyn_cast<llhd::ProbeOp>(user)) {
        probes.push_back(probe);
      } else {
        hasUnsupportedUse = true;
        break;
      }
    }
    if (hasUnsupportedUse || hasEnable)
      continue;

    Block *entryBlock = getEntryBlock(sigOp.getOperation());
    if (!entryBlock || sigOp->getBlock() != entryBlock)
      continue;

    OpBuilder entryBuilder(entryBlock, entryBlock->begin());
    auto ptrTy = LLVM::LLVMPointerType::get(&getContext());
    auto one = LLVM::ConstantOp::create(entryBuilder, sigOp.getLoc(),
                                        entryBuilder.getI64Type(), 1);
    auto alloca = LLVM::AllocaOp::create(entryBuilder, sigOp.getLoc(), ptrTy,
                                         nestedType, one);
    OpBuilder initBuilder(sigOp);
    initBuilder.setInsertionPointAfter(sigOp);
    LLVM::StoreOp::create(initBuilder, sigOp.getLoc(), sigOp.getInit(),
                          alloca);

    for (auto drive : drives) {
      OpBuilder driveBuilder(drive);
      LLVM::StoreOp::create(driveBuilder, drive.getLoc(), drive.getValue(),
                            alloca);
      drive.erase();
    }

    for (auto probe : probes) {
      OpBuilder probeBuilder(probe);
      auto load = LLVM::LoadOp::create(probeBuilder, probe.getLoc(),
                                       nestedType, alloca);
      probe.replaceAllUsesWith(load.getResult());
      probe.erase();
    }

    if (sigOp.use_empty())
      sigOp.erase();
  }

  SmallVector<llhd::DriveOp> refDrives;
  SmallVector<llhd::ProbeOp> refProbes;
  moduleOp.walk([&](Operation *op) {
    if (op->getParentOfType<hw::HWModuleOp>())
      return;
    if (op->getParentOfType<llhd::ProcessOp>())
      return;
    if (auto drive = dyn_cast<llhd::DriveOp>(op))
      refDrives.push_back(drive);
    if (auto probe = dyn_cast<llhd::ProbeOp>(op))
      refProbes.push_back(probe);
  });

  for (auto drive : refDrives) {
    if (drive.getEnable())
      continue;
    auto castOp =
        drive.getSignal().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1)
      continue;
    if (!isa<LLVM::LLVMPointerType>(castOp.getOperand(0).getType()))
      continue;
    if (!isa<llhd::RefType>(castOp.getResultTypes().front()))
      continue;
    OpBuilder builder(drive);
    LLVM::StoreOp::create(builder, drive.getLoc(), drive.getValue(),
                          castOp.getOperand(0));
    drive.erase();
    if (castOp.use_empty())
      castOp.erase();
  }

  for (auto probe : refProbes) {
    auto castOp =
        probe.getSignal().getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp || castOp.getNumOperands() != 1 ||
        castOp.getNumResults() != 1)
      continue;
    if (!isa<LLVM::LLVMPointerType>(castOp.getOperand(0).getType()))
      continue;
    if (!isa<llhd::RefType>(castOp.getResultTypes().front()))
      continue;
    OpBuilder builder(probe);
    auto load =
        LLVM::LoadOp::create(builder, probe.getLoc(), probe.getType(),
                             castOp.getOperand(0));
    probe.replaceAllUsesWith(load.getResult());
    probe.erase();
    if (castOp.use_empty())
      castOp.erase();
  }

  SmallVector<llhd::TimeToIntOp> timeToIntOps;
  moduleOp.walk([&](llhd::TimeToIntOp op) {
    if (op->getParentOfType<hw::HWModuleOp>())
      return;
    if (op->getParentOfType<llhd::ProcessOp>())
      return;
    timeToIntOps.push_back(op);
  });
  for (auto op : timeToIntOps) {
    auto currentTime = op.getInput().getDefiningOp<llhd::CurrentTimeOp>();
    if (!currentTime)
      continue;
    OpBuilder builder(op);
    auto zero = LLVM::ConstantOp::create(builder, op.getLoc(), op.getType(), 0);
    op.replaceAllUsesWith(zero.getResult());
    op.erase();
    if (currentTime.use_empty())
      currentTime.erase();
  }

  SmallVector<llhd::ConstantTimeOp> constantTimes;
  moduleOp.walk([&](llhd::ConstantTimeOp op) {
    if (op->getParentOfType<hw::HWModuleOp>())
      return;
    if (op->getParentOfType<llhd::ProcessOp>())
      return;
    if (op.use_empty())
      constantTimes.push_back(op);
  });
  for (auto op : constantTimes)
    op.erase();

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
  LLVM::CallOp::create(builder, loc, reportFunc.value(),
                       ValueRange{bmcOp.getResult()});
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
