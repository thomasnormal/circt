//===- StripLLHDProcesses.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"
#include <utility>

using namespace mlir;
using namespace circt;
using namespace circt::hw;
using namespace circt::llhd;
using namespace igraph;

namespace circt {
#define GEN_PASS_DEF_STRIPLLHDPROCESSES
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct BlockGuardInfo {
  Value condition;
  bool negate = false;
  Block *block = nullptr;
};

static std::optional<BlockGuardInfo> getBlockGuard(Block *block) {
  if (!llvm::hasSingleElement(block->getPredecessors()))
    return std::nullopt;
  Block *pred = *block->getPredecessors().begin();
  auto *terminator = pred->getTerminator();
  if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
    if (condBr.getTrueDest() == block)
      return BlockGuardInfo{condBr.getCondition(), /*negate=*/false, pred};
    if (condBr.getFalseDest() == block)
      return BlockGuardInfo{condBr.getCondition(), /*negate=*/true, pred};
  }
  return std::nullopt;
}

static bool isValueFromAbove(Value value, ProcessOp process) {
  if (auto *region = value.getParentRegion())
    return region->isProperAncestor(&process.getBody());
  return false;
}

static bool isDefinedInsideProcess(Value value, ProcessOp process) {
  if (auto *region = value.getParentRegion())
    return process.getBody().isAncestor(region);
  return false;
}

static std::optional<SmallVector<Value>>
findBlockArgMapping(Block *block, ProcessOp process) {
  if (block->getNumArguments() == 0 && block->getPredecessors().empty())
    return SmallVector<Value>{};
  auto checkOperands = [&](ValueRange operands)
      -> std::optional<SmallVector<Value>> {
    if (operands.size() != block->getNumArguments())
      return std::nullopt;
    for (Value operand : operands) {
      if (!isValueFromAbove(operand, process))
        return std::nullopt;
    }
    return SmallVector<Value>(operands.begin(), operands.end());
  };

  for (Block *pred : block->getPredecessors()) {
    auto *terminator = pred->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
      if (br.getDest() == block) {
        if (auto mapping = checkOperands(br.getDestOperands()))
          return mapping;
      }
      continue;
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(terminator)) {
      if (condBr.getTrueDest() == block) {
        if (auto mapping = checkOperands(condBr.getTrueDestOperands()))
          return mapping;
      }
      if (condBr.getFalseDest() == block) {
        if (auto mapping = checkOperands(condBr.getFalseDestOperands()))
          return mapping;
      }
      continue;
    }
  }
  return std::nullopt;
}

struct StripLLHDProcessesPass
    : public circt::impl::StripLLHDProcessesBase<StripLLHDProcessesPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect>();
  }

  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
    DenseSet<Operation *> handled;
    DenseMap<StringAttr, SmallVector<Type>> addedInputs;
    DenseMap<StringAttr, SmallVector<StringAttr>> addedInputNames;

    for (auto *startNode : instanceGraph) {
      if (handled.count(startNode->getModule().getOperation()))
        continue;

      for (InstanceGraphNode *node : llvm::post_order(startNode)) {
        if (!handled.insert(node->getModule().getOperation()).second)
          continue;

        auto hwModule =
            dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
        if (!hwModule)
          continue;

        Namespace ns;
        for (auto nameAttr : hwModule.getModuleType().getInputNames())
          ns.add(cast<StringAttr>(nameAttr).getValue());
        for (auto nameAttr : hwModule.getModuleType().getOutputNames())
          ns.add(cast<StringAttr>(nameAttr).getValue());
        unsigned insertIndex = hwModule.getNumInputPorts();
        if (auto numRegsAttr =
                hwModule->getAttrOfType<IntegerAttr>("num_regs"))
          insertIndex = hwModule.getNumInputPorts() - numRegsAttr.getInt();

        auto &moduleAddedInputs = addedInputs[hwModule.getSymNameAttr()];
        auto &moduleAddedInputNames =
            addedInputNames[hwModule.getSymNameAttr()];

        auto addInputPort =
            [&](StringAttr desiredName,
                Type type) -> std::pair<StringAttr, Value> {
          auto newInput =
              hwModule.insertInput(insertIndex++, desiredName, type);
          moduleAddedInputs.push_back(type);
          moduleAddedInputNames.push_back(newInput.first);
          ns.add(newInput.first.getValue());
          return {newInput.first, newInput.second};
        };

        SmallVector<ProcessOp> processes;
        hwModule.walk(
            [&](ProcessOp process) { processes.push_back(process); });
        DenseMap<Value, Value> signalInputs;
        int64_t abstractedProcessResultCount = 0;
        auto getSignalName = [](Value signal) -> StringRef {
          if (auto sig = signal.getDefiningOp<SignalOp>()) {
            if (auto nameAttr = sig.getNameAttr())
              return nameAttr.getValue();
          }
          return "llhd_signal";
        };
        Value zeroTime;
        auto getZeroTime = [&]() -> Value {
          if (zeroTime)
            return zeroTime;
          OpBuilder builder(&hwModule.getBody().front(),
                            hwModule.getBody().front().begin());
          zeroTime = ConstantTimeOp::create(
              builder, hwModule.getLoc(), /*time=*/0, /*timeUnit=*/"ns",
              /*delta=*/0, /*epsilon=*/1);
          return zeroTime;
        };

        for (auto process : processes) {
          bool hasWait = false;
          process.walk([&](WaitOp) { hasWait = true; });

          SmallVector<Operation *> assertLikeOps;
          process.walk([&](Operation *op) {
            if (isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp,
                    verif::ClockedAssertOp, verif::ClockedAssumeOp,
                    verif::ClockedCoverOp>(op))
              assertLikeOps.push_back(op);
          });

          for (Operation *op : assertLikeOps) {
            auto *block = op->getBlock();
            auto mappingValues = findBlockArgMapping(block, process);
            if (!mappingValues)
              continue;
            auto guardInfo = getBlockGuard(block);

            Block *guardBlock = nullptr;
            if (guardInfo) {
              if (auto *defOp = guardInfo->condition.getDefiningOp())
                guardBlock = defOp->getBlock();
              else if (auto guardArg =
                           dyn_cast<BlockArgument>(guardInfo->condition))
                guardBlock = guardArg.getOwner();
            }

            BackwardSliceOptions sliceOptions;
            sliceOptions.inclusive = true;
            sliceOptions.omitBlockArguments = true;
            sliceOptions.omitUsesFromAbove = true;

            SetVector<Operation *> slice;
            auto addSliceForValue = [&](Value value) {
              if (!value)
                return;
              (void)getBackwardSlice(value, &slice, sliceOptions);
            };

            if (auto assertOp = dyn_cast<verif::AssertOp>(op)) {
              addSliceForValue(assertOp.getProperty());
              addSliceForValue(assertOp.getEnable());
            } else if (auto assumeOp = dyn_cast<verif::AssumeOp>(op)) {
              addSliceForValue(assumeOp.getProperty());
              addSliceForValue(assumeOp.getEnable());
            } else if (auto coverOp = dyn_cast<verif::CoverOp>(op)) {
              addSliceForValue(coverOp.getProperty());
              addSliceForValue(coverOp.getEnable());
            } else if (auto clockedAssertOp =
                           dyn_cast<verif::ClockedAssertOp>(op)) {
              addSliceForValue(clockedAssertOp.getProperty());
              addSliceForValue(clockedAssertOp.getEnable());
              addSliceForValue(clockedAssertOp.getClock());
            } else if (auto clockedAssumeOp =
                           dyn_cast<verif::ClockedAssumeOp>(op)) {
              addSliceForValue(clockedAssumeOp.getProperty());
              addSliceForValue(clockedAssumeOp.getEnable());
              addSliceForValue(clockedAssumeOp.getClock());
            } else if (auto clockedCoverOp =
                           dyn_cast<verif::ClockedCoverOp>(op)) {
              addSliceForValue(clockedCoverOp.getProperty());
              addSliceForValue(clockedCoverOp.getEnable());
              addSliceForValue(clockedCoverOp.getClock());
            }
            if (guardInfo)
              addSliceForValue(guardInfo->condition);

            IRMapping mapping;
            SmallVector<Value> insertAfterValues;
            llvm::SmallPtrSet<Block *, 4> mappedBlocks;
            auto mapBlockArgs = [&](Block *argBlock,
                                    ArrayRef<Value> mappedValues) {
              for (auto [arg, value] :
                   llvm::zip(argBlock->getArguments(), mappedValues))
                mapping.map(arg, value);
              insertAfterValues.append(mappedValues.begin(), mappedValues.end());
              mappedBlocks.insert(argBlock);
            };
            mapBlockArgs(block, *mappingValues);
            if (guardBlock && guardBlock != block &&
                guardBlock->getNumArguments() != 0) {
              auto guardMappingValues = findBlockArgMapping(guardBlock, process);
              if (!guardMappingValues)
                continue;
              mapBlockArgs(guardBlock, *guardMappingValues);
            }
            bool canHoist = true;
            auto ensureBlockMapping = [&](Block *argBlock) {
              if (mappedBlocks.contains(argBlock))
                return true;
              auto extraMapping = findBlockArgMapping(argBlock, process);
              if (!extraMapping)
                return false;
              mapBlockArgs(argBlock, *extraMapping);
              return true;
            };
            if (guardInfo && guardInfo->condition &&
                isa<BlockArgument>(guardInfo->condition)) {
              auto guardArg = cast<BlockArgument>(guardInfo->condition);
              if (!ensureBlockMapping(guardArg.getOwner()))
                continue;
            }
            for (Operation *sliceOp : slice) {
              if (sliceOp->getParentOfType<ProcessOp>() != process)
                continue;
              for (Value operand : sliceOp->getOperands()) {
                auto blockArg = dyn_cast<BlockArgument>(operand);
                if (!blockArg || !isDefinedInsideProcess(blockArg, process))
                  continue;
                if (!ensureBlockMapping(blockArg.getOwner())) {
                  canHoist = false;
                  break;
                }
              }
              if (!canHoist)
                break;
            }
            if (!canHoist)
              continue;

            Operation *insertAfter = process.getOperation();
            for (Value value : insertAfterValues) {
              if (auto *defOp = value.getDefiningOp()) {
                if (defOp->getBlock() == insertAfter->getBlock() &&
                    insertAfter->isBeforeInBlock(defOp))
                  insertAfter = defOp;
              }
            }

            OpBuilder builder(insertAfter);
            builder.setInsertionPointAfter(insertAfter);
            for (Operation *sliceOp : slice) {
              if (sliceOp->getParentOfType<ProcessOp>() != process)
                continue;
              if (isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp>(sliceOp))
                continue;
              builder.clone(*sliceOp, mapping);
            }

            if (auto assertOp = dyn_cast<verif::AssertOp>(op)) {
              Value property = mapping.lookupOrDefault(assertOp.getProperty());
              Value enable = mapping.lookupOrDefault(assertOp.getEnable());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(assertOp.getProperty(), process) &&
                  property == assertOp.getProperty())
                continue;
              if (assertOp.getEnable() &&
                  isDefinedInsideProcess(assertOp.getEnable(), process) &&
                  enable == assertOp.getEnable())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              auto newOp = verif::AssertOp::create(
                  builder, op->getLoc(), property, enable,
                  assertOp.getLabelAttr());
              newOp->setAttrs(assertOp->getAttrs());
            } else if (auto assumeOp = dyn_cast<verif::AssumeOp>(op)) {
              Value property = mapping.lookupOrDefault(assumeOp.getProperty());
              Value enable = mapping.lookupOrDefault(assumeOp.getEnable());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(assumeOp.getProperty(), process) &&
                  property == assumeOp.getProperty())
                continue;
              if (assumeOp.getEnable() &&
                  isDefinedInsideProcess(assumeOp.getEnable(), process) &&
                  enable == assumeOp.getEnable())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              auto newOp = verif::AssumeOp::create(
                  builder, op->getLoc(), property, enable,
                  assumeOp.getLabelAttr());
              newOp->setAttrs(assumeOp->getAttrs());
            } else if (auto coverOp = dyn_cast<verif::CoverOp>(op)) {
              Value property = mapping.lookupOrDefault(coverOp.getProperty());
              Value enable = mapping.lookupOrDefault(coverOp.getEnable());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(coverOp.getProperty(), process) &&
                  property == coverOp.getProperty())
                continue;
              if (coverOp.getEnable() &&
                  isDefinedInsideProcess(coverOp.getEnable(), process) &&
                  enable == coverOp.getEnable())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              auto newOp = verif::CoverOp::create(
                  builder, op->getLoc(), property, enable,
                  coverOp.getLabelAttr());
              newOp->setAttrs(coverOp->getAttrs());
            } else if (auto clockedAssertOp =
                           dyn_cast<verif::ClockedAssertOp>(op)) {
              Value property =
                  mapping.lookupOrDefault(clockedAssertOp.getProperty());
              Value enable =
                  mapping.lookupOrDefault(clockedAssertOp.getEnable());
              Value clock = mapping.lookupOrDefault(clockedAssertOp.getClock());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(clockedAssertOp.getProperty(),
                                         process) &&
                  property == clockedAssertOp.getProperty())
                continue;
              if (clockedAssertOp.getEnable() &&
                  isDefinedInsideProcess(clockedAssertOp.getEnable(),
                                         process) &&
                  enable == clockedAssertOp.getEnable())
                continue;
              if (isDefinedInsideProcess(clockedAssertOp.getClock(), process) &&
                  clock == clockedAssertOp.getClock())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              verif::ClockedAssertOp::create(
                  builder, op->getLoc(), property, clockedAssertOp.getEdge(),
                  clock, enable, clockedAssertOp.getLabelAttr());
            } else if (auto clockedAssumeOp =
                           dyn_cast<verif::ClockedAssumeOp>(op)) {
              Value property =
                  mapping.lookupOrDefault(clockedAssumeOp.getProperty());
              Value enable =
                  mapping.lookupOrDefault(clockedAssumeOp.getEnable());
              Value clock = mapping.lookupOrDefault(clockedAssumeOp.getClock());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(clockedAssumeOp.getProperty(),
                                         process) &&
                  property == clockedAssumeOp.getProperty())
                continue;
              if (clockedAssumeOp.getEnable() &&
                  isDefinedInsideProcess(clockedAssumeOp.getEnable(),
                                         process) &&
                  enable == clockedAssumeOp.getEnable())
                continue;
              if (isDefinedInsideProcess(clockedAssumeOp.getClock(), process) &&
                  clock == clockedAssumeOp.getClock())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              verif::ClockedAssumeOp::create(
                  builder, op->getLoc(), property, clockedAssumeOp.getEdge(),
                  clock, enable, clockedAssumeOp.getLabelAttr());
            } else if (auto clockedCoverOp =
                           dyn_cast<verif::ClockedCoverOp>(op)) {
              Value property =
                  mapping.lookupOrDefault(clockedCoverOp.getProperty());
              Value enable =
                  mapping.lookupOrDefault(clockedCoverOp.getEnable());
              Value clock = mapping.lookupOrDefault(clockedCoverOp.getClock());
              Value guardValue;
              if (guardInfo && guardInfo->condition) {
                guardValue = mapping.lookupOrDefault(guardInfo->condition);
                if (guardInfo->negate) {
                  auto one = hw::ConstantOp::create(
                      builder, op->getLoc(), builder.getI1Type(), 1);
                  guardValue = comb::XorOp::create(builder, op->getLoc(),
                                                   guardValue, one);
                }
              }
              if (isDefinedInsideProcess(clockedCoverOp.getProperty(),
                                         process) &&
                  property == clockedCoverOp.getProperty())
                continue;
              if (clockedCoverOp.getEnable() &&
                  isDefinedInsideProcess(clockedCoverOp.getEnable(),
                                         process) &&
                  enable == clockedCoverOp.getEnable())
                continue;
              if (isDefinedInsideProcess(clockedCoverOp.getClock(), process) &&
                  clock == clockedCoverOp.getClock())
                continue;
              if (guardValue) {
                if (enable)
                  enable = comb::AndOp::create(builder, op->getLoc(), guardValue,
                                               enable);
                else
                  enable = guardValue;
              }
              verif::ClockedCoverOp::create(
                  builder, op->getLoc(), property, clockedCoverOp.getEdge(),
                  clock, enable, clockedCoverOp.getLabelAttr());
            }
          }

          SmallVector<DriveOp> initDrives;
          SmallVector<DriveOp> dynamicDrives;
          DenseMap<Value, Type> driveValueTypes;
          process.walk([&](DriveOp drvOp) {
            if (!isValueFromAbove(drvOp.getSignal(), process))
              return;
            if (hasWait)
              dynamicDrives.push_back(drvOp);
            else
              initDrives.push_back(drvOp);
            if (!driveValueTypes.count(drvOp.getSignal()))
              driveValueTypes[drvOp.getSignal()] = drvOp.getValue().getType();
          });

        auto hasExternalDriver = [&](Value signal) -> bool {
          for (auto *user : signal.getUsers()) {
            auto drv = dyn_cast<DriveOp>(user);
            if (!drv)
              continue;
            if (!drv->getParentOfType<ProcessOp>())
              return true;
          }
          return false;
        };

        if (!dynamicDrives.empty()) {
          auto isZeroTimeConst = [](Value time) -> bool {
            auto ct = time.getDefiningOp<ConstantTimeOp>();
            if (!ct)
              return false;
            auto attr = ct.getValue();
            return attr.getTime() == 0 && attr.getDelta() == 0;
          };
          auto resolveDynamicDriveValue = [&](Value signal) -> Value {
            Value resolved;
            for (auto drvOp : dynamicDrives) {
              if (drvOp.getSignal() != signal)
                continue;
              if (drvOp.getEnable())
                return {};
              if (!isValueFromAbove(drvOp.getValue(), process))
                return {};
              if (!isValueFromAbove(drvOp.getTime(), process))
                return {};
              if (!isZeroTimeConst(drvOp.getTime()))
                return {};
              Value val = drvOp.getValue();
              if (!resolved)
                resolved = val;
              else if (resolved != val)
                return {};
            }
            return resolved;
          };
          DenseSet<Value> drivenSignals;
          for (auto drvOp : dynamicDrives)
            drivenSignals.insert(drvOp.getSignal());
          for (auto signal : drivenSignals) {
            if (hasExternalDriver(signal))
              continue;
            if (signalInputs.contains(signal))
              continue;
            if (Value resolved = resolveDynamicDriveValue(signal)) {
              OpBuilder builder(hwModule.getContext());
              builder.setInsertionPoint(
                  hwModule.getBodyBlock()->getTerminator());
              DriveOp::create(builder, process.getLoc(), signal, resolved,
                              getZeroTime(), Value{});
              continue;
            }
            auto typeIt = driveValueTypes.find(signal);
            if (typeIt == driveValueTypes.end())
              continue;
            auto baseName = ns.newName(getSignalName(signal));
              auto newInput =
                  addInputPort(StringAttr::get(hwModule.getContext(), baseName),
                               typeIt->second);
              signalInputs[signal] = newInput.second;
              OpBuilder builder(hwModule.getContext());
              builder.setInsertionPoint(
                  hwModule.getBodyBlock()->getTerminator());
              DriveOp::create(builder, process.getLoc(), signal,
                              newInput.second, getZeroTime(), Value{});
            }
          } else {
            OpBuilder builder(process);
            builder.setInsertionPointAfter(process);
            for (auto drvOp : initDrives) {
              if (!isValueFromAbove(drvOp.getValue(), process))
                continue;
              if (!isValueFromAbove(drvOp.getTime(), process))
                continue;
              if (drvOp.getEnable() &&
                  !isValueFromAbove(drvOp.getEnable(), process))
                continue;
              builder.clone(*drvOp);
            }
          }

          for (Value result : process.getResults()) {
            auto name = ns.newName("llhd_process_result");
            auto newInput = addInputPort(
                StringAttr::get(hwModule.getContext(), name),
                result.getType());
            result.replaceAllUsesWith(newInput.second);
            ++abstractedProcessResultCount;
          }
          process.erase();
        }

        if (abstractedProcessResultCount > 0) {
          hwModule->setAttr(
              "circt.bmc_abstracted_llhd_process_results",
              IntegerAttr::get(IntegerType::get(hwModule.getContext(), 32),
                               abstractedProcessResultCount));
        } else {
          hwModule->removeAttr("circt.bmc_abstracted_llhd_process_results");
        }

        SmallVector<InstanceOp> instances;
        hwModule.walk(
            [&](InstanceOp instance) { instances.push_back(instance); });

        for (auto instance : instances) {
          auto moduleNameAttr = instance.getModuleNameAttr().getAttr();
          auto &childInputs = addedInputs[moduleNameAttr];
          if (childInputs.empty())
            continue;
          auto &childInputNames = addedInputNames[moduleNameAttr];

          OpBuilder builder(instance);
          SmallVector<Value> operands(instance.getInputs().begin(),
                                      instance.getInputs().end());
          SmallVector<Attribute> argNames(
              instance.getInputNames().getValue());
          for (auto [type, name] : llvm::zip(childInputs, childInputNames)) {
            auto newInput = addInputPort(name, type);
            operands.push_back(newInput.second);
            argNames.push_back(name);
          }

          SmallVector<Attribute> resultNames(
              instance.getOutputNames().getValue());
          SmallVector<Type> resultTypes(instance->getResultTypes());
          auto newInst = InstanceOp::create(
              builder, instance.getLoc(), resultTypes,
              instance.getInstanceNameAttr(), instance.getModuleNameAttr(),
              operands, builder.getArrayAttr(argNames),
              builder.getArrayAttr(resultNames), instance.getParametersAttr(),
              instance.getInnerSymAttr(), instance.getDoNotPrintAttr());
          instance.replaceAllUsesWith(newInst.getResults());
          instanceGraph.replaceInstance(instance, newInst);
          instance.erase();
        }
      }
    }
  }
};
} // namespace
