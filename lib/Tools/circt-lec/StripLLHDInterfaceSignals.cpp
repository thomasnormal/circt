//===- StripLLHDInterfaceSignals.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BoolCondition.h"
#include "circt/Support/FourStateUtils.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <string>

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_STRIPLLHDINTERFACESIGNALS
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {
struct FieldAccess {
  Value storedValue;
  SmallVector<llhd::ProbeOp, 2> reads;
  SmallVector<LLVM::StoreOp, 2> stores;
  bool needsAbstraction = false;
  bool hasMultipleStoreValues = false;
};

struct ModuleState {
  Namespace ns;
  unsigned insertIndex = 0;

  explicit ModuleState(hw::HWModuleOp module) {
    for (auto nameAttr : module.getModuleType().getInputNames())
      ns.add(cast<StringAttr>(nameAttr).getValue());
    for (auto nameAttr : module.getModuleType().getOutputNames())
      ns.add(cast<StringAttr>(nameAttr).getValue());
    insertIndex = module.getNumInputPorts();
    if (auto numRegsAttr = module->getAttrOfType<IntegerAttr>("num_regs"))
      insertIndex = module.getNumInputPorts() - numRegsAttr.getInt();
  }

  Value addInput(hw::HWModuleOp module, StringRef baseName, Type type) {
    auto name = ns.newName(baseName);
    return module
        .insertInput(insertIndex++, StringAttr::get(module.getContext(), name),
                     type)
        .second;
  }
};

static bool isAllOnesConstant(Value value) {
  auto constant = value.getDefiningOp<hw::ConstantOp>();
  return constant && constant.getValue().isAllOnes();
}

static bool isNegationOf(Value maybeNot, Value base) {
  if (maybeNot == base)
    return false;
  auto xorOp = maybeNot.getDefiningOp<comb::XorOp>();
  if (!xorOp)
    return false;
  Value lhs = xorOp.getOperand(0);
  Value rhs = xorOp.getOperand(1);
  if (lhs == base && isAllOnesConstant(rhs))
    return true;
  if (rhs == base && isAllOnesConstant(lhs))
    return true;
  return false;
}

static bool areComplementary(const BoolCondition &lhs, const BoolCondition &rhs) {
  if (lhs.isTrue())
    return rhs.isFalse();
  if (lhs.isFalse())
    return rhs.isTrue();
  Value lhsVal = lhs.getValue();
  Value rhsVal = rhs.getValue();
  if (!lhsVal || !rhsVal)
    return false;
  return isNegationOf(lhsVal, rhsVal) || isNegationOf(rhsVal, lhsVal);
}

static bool hasComplementaryEnables(ArrayRef<llhd::DriveOp> drives) {
  if (drives.size() != 2)
    return false;
  llhd::DriveOp lhsDrive = drives[0];
  llhd::DriveOp rhsDrive = drives[1];
  Value lhsEnable = lhsDrive.getEnable();
  Value rhsEnable = rhsDrive.getEnable();
  if (!lhsEnable || !rhsEnable)
    return false;
  return areComplementary(BoolCondition(lhsEnable), BoolCondition(rhsEnable));
}

static bool hasExclusiveEnables(ArrayRef<llhd::DriveOp> drives,
                                OpBuilder &builder) {
  if (drives.size() < 2)
    return false;
  SmallVector<BoolCondition, 4> conds;
  conds.reserve(drives.size());
  for (auto drive : drives) {
    Value enable = drive.getEnable();
    if (!enable)
      return false;
    conds.emplace_back(enable);
  }

  BoolCondition covered(false);
  for (const auto &cond : conds)
    covered = covered.orWith(cond, builder);
  if (!covered.isTrue())
    return false;

  for (size_t i = 0; i < conds.size(); ++i) {
    for (size_t j = i + 1; j < conds.size(); ++j) {
      BoolCondition overlap = conds[i].andWith(conds[j], builder);
      if (!overlap.isFalse())
        return false;
    }
  }
  return true;
}

static BoolCondition getBranchDecisionsFromDominatorToTarget(
    OpBuilder &builder, Block *dominator, Block *target,
    SmallDenseMap<std::pair<Block *, Block *>, BoolCondition> &decisions) {
  if (auto decision = decisions.lookup({dominator, target}))
    return decision;

  SmallPtrSet<Block *, 8> visitedBlocks;
  visitedBlocks.insert(dominator);
  if (auto &decision = decisions[{dominator, dominator}]; !decision)
    decision = BoolCondition(true);

  for (auto *block : llvm::inverse_post_order_ext(target, visitedBlocks)) {
    auto merged = BoolCondition(false);
    for (auto *pred : block->getPredecessors()) {
      auto predDecision = decisions.lookup({dominator, pred});
      if (!predDecision)
        continue;
      if (pred->getTerminator()->getNumSuccessors() != 1) {
        auto condBr = cast<cf::CondBranchOp>(pred->getTerminator());
        if (condBr.getTrueDest() == condBr.getFalseDest()) {
          merged = merged.orWith(predDecision, builder);
        } else {
          auto cond = BoolCondition(condBr.getCondition());
          if (condBr.getFalseDest() == block)
            cond = cond.inverted(builder);
          merged = merged.orWith(predDecision.andWith(cond, builder), builder);
        }
      } else {
        merged = merged.orWith(predDecision, builder);
      }
    }
    decisions[{dominator, block}] = merged;
  }

  return decisions.lookup({dominator, target});
}

struct CFRemover {
  explicit CFRemover(Region &region) : region(region) {}
  void run();

  Region &region;
  SmallVector<Block *> sortedBlocks;
  DominanceInfo domInfo;
};

void CFRemover::run() {
  SmallVector<llhd::YieldOp, 2> yieldOps;
  Block *entry = region.empty() ? nullptr : &region.front();
  if (!entry)
    return;

  // Use ReversePostOrderTraversal to visit blocks in topological order
  // (predecessors before successors). This is needed for correct block merging.
  llvm::ReversePostOrderTraversal<Block *> rpot(entry);
  for (auto *block : rpot)
    sortedBlocks.push_back(block);

  if (sortedBlocks.size() != region.getBlocks().size())
    return;

  DenseMap<Block *, unsigned> blockOrder;
  blockOrder.reserve(sortedBlocks.size());
  for (auto [index, block] : llvm::enumerate(sortedBlocks))
    blockOrder[block] = index;
  for (auto *block : sortedBlocks) {
    auto blockIndex = blockOrder.lookup(block);
    for (auto *pred : block->getPredecessors()) {
      auto it = blockOrder.find(pred);
      if (it == blockOrder.end())
        return;
      if (it->second >= blockIndex)
        return;
    }
  }

  for (auto *block : sortedBlocks) {
    for (auto &op : *block) {
      if (!isMemoryEffectFree(&op) &&
          !isa<llhd::ProbeOp, llhd::DriveOp, llhd::SignalOp, LLVM::AllocaOp,
               UnrealizedConversionCastOp, LLVM::UndefOp, LLVM::InsertValueOp,
               LLVM::ConstantOp>(op)) {
        return;
      }
    }

    if (!isa<llhd::YieldOp, cf::BranchOp, cf::CondBranchOp>(
            block->getTerminator())) {
      return;
    }

    if (auto yieldOp = dyn_cast<llhd::YieldOp>(block->getTerminator()))
      yieldOps.push_back(yieldOp);
  }

  if (yieldOps.empty())
    return;

  auto yieldOp = yieldOps[0];
  if (yieldOps.size() > 1) {
    OpBuilder builder(region.getContext());
    SmallVector<Location> locs(yieldOps[0].getNumOperands(), region.getLoc());
    auto *yieldBlock = builder.createBlock(&region, region.end(),
                                           yieldOps[0].getOperandTypes(), locs);
    sortedBlocks.push_back(yieldBlock);
    yieldOp = llhd::YieldOp::create(builder, region.getLoc(),
                                    yieldBlock->getArguments());
    for (auto otherYield : yieldOps) {
      builder.setInsertionPoint(otherYield);
      cf::BranchOp::create(builder, otherYield.getLoc(), yieldBlock,
                           otherYield.getOperands());
      otherYield.erase();
    }
  }

  domInfo = DominanceInfo(region.getParentOp());

  SmallDenseMap<std::pair<Block *, Block *>, BoolCondition> decisionCache;
  auto *entryBlock = sortedBlocks.front();
  for (auto *block : sortedBlocks) {
    if (!domInfo.isReachableFromEntry(block))
      continue;

    auto *domBlock = block;
    for (auto *pred : block->getPredecessors())
      if (domInfo.isReachableFromEntry(pred))
        domBlock = domInfo.findNearestCommonDominator(domBlock, pred);

    OpBuilder builder(entryBlock->getTerminator());
    SmallVector<Value> mergedArgs;
    SmallPtrSet<Block *, 4> seenPreds;
    for (auto *pred : block->getPredecessors()) {
      if (!seenPreds.insert(pred).second)
        continue;
      if (!domInfo.isReachableFromEntry(pred))
        continue;

      auto mergeArgs = [&](ValueRange args, BoolCondition cond, bool invCond) {
        if (mergedArgs.empty()) {
          mergedArgs = args;
          return;
        }
        auto decision = getBranchDecisionsFromDominatorToTarget(
            builder, domBlock, pred, decisionCache);
        if (cond) {
          if (invCond)
            cond = cond.inverted(builder);
          decision = decision.andWith(cond, builder);
        }
        for (auto [mergedArg, arg] : llvm::zip(mergedArgs, args)) {
          if (decision.isTrue())
            mergedArg = arg;
          else if (decision.isFalse())
            continue;
          else
            mergedArg = builder.createOrFold<comb::MuxOp>(
                arg.getLoc(), decision.materialize(builder, arg.getLoc()), arg,
                mergedArg);
        }
      };

      if (auto condBrOp = dyn_cast<cf::CondBranchOp>(pred->getTerminator())) {
        if (condBrOp.getTrueDest() == condBrOp.getFalseDest()) {
          SmallVector<Value> mergedOperands;
          mergedOperands.reserve(block->getNumArguments());
          for (auto [trueArg, falseArg] :
               llvm::zip(condBrOp.getTrueDestOperands(),
                         condBrOp.getFalseDestOperands())) {
            mergedOperands.push_back(builder.createOrFold<comb::MuxOp>(
                trueArg.getLoc(), condBrOp.getCondition(), trueArg, falseArg));
          }
          mergeArgs(mergedOperands, Value{}, false);
        } else if (condBrOp.getTrueDest() == block) {
          mergeArgs(condBrOp.getTrueDestOperands(), condBrOp.getCondition(),
                    false);
        } else {
          mergeArgs(condBrOp.getFalseDestOperands(), condBrOp.getCondition(),
                    true);
        }
      } else {
        auto brOp = cast<cf::BranchOp>(pred->getTerminator());
        mergeArgs(brOp.getDestOperands(), Value{}, false);
      }
    }
    for (auto [blockArg, mergedArg] :
         llvm::zip(block->getArguments(), mergedArgs))
      blockArg.replaceAllUsesWith(mergedArg);

    if (block != entryBlock) {
      // Compute the condition under which this block executes
      auto blockCondition = getBranchDecisionsFromDominatorToTarget(
          builder, entryBlock, block, decisionCache);

      // Move operations from this block to entry, adding enables to drives
      for (auto it = block->begin(); it != std::prev(block->end());) {
        Operation *op = &*it++;
        op->moveBefore(entryBlock->getTerminator());

        // Add enable condition to drive operations
        if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
          if (!driveOp.getEnable() && blockCondition &&
              !blockCondition.isTrue()) {
            driveOp.getEnableMutable().assign(
                blockCondition.materialize(builder, driveOp.getLoc()));
          }
        }
      }
    }
  }

  if (yieldOp != entryBlock->getTerminator()) {
    yieldOp->moveBefore(entryBlock->getTerminator());
    entryBlock->getTerminator()->erase();
  }

  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->clear();
  for (auto *block : sortedBlocks)
    if (block != entryBlock)
      block->erase();
}

static std::optional<unsigned> getStructFieldIndex(LLVM::GEPOp gep) {
  auto rawIndices = gep.getRawConstantIndices();
  if (rawIndices.size() != 2)
    return std::nullopt;
  if (rawIndices[0] == LLVM::GEPOp::kDynamicIndex ||
      rawIndices[1] == LLVM::GEPOp::kDynamicIndex)
    return std::nullopt;
  if (rawIndices[0] != 0 || rawIndices[1] < 0)
    return std::nullopt;
  if (!gep.getDynamicIndices().empty())
    return std::nullopt;
  return static_cast<unsigned>(rawIndices[1]);
}

static Value unwrapStoredValue(Value value) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
      return cast->getOperand(0);
  }
  return value;
}

static Value findInsertedValue(Value value, ArrayRef<int64_t> path) {
  auto insert = value.getDefiningOp<LLVM::InsertValueOp>();
  if (!insert)
    return {};

  ArrayRef<int64_t> insertPos = insert.getPosition();
  if (path == insertPos)
    return insert.getValue();

  if (path.size() >= insertPos.size() &&
      llvm::equal(insertPos, path.take_front(insertPos.size()))) {
    if (Value nested = findInsertedValue(
            insert.getValue(), path.drop_front(insertPos.size())))
      return nested;
  }
  return findInsertedValue(insert.getContainer(), path);
}

static Value buildHWStructFromLLVM(Value llvmStruct, hw::StructType hwType,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> prefix) {
  SmallVector<Value, 8> fields;
  auto elements = hwType.getElements();
  fields.reserve(elements.size());
  for (auto [index, field] : llvm::enumerate(elements)) {
    SmallVector<int64_t, 4> path(prefix.begin(), prefix.end());
    path.push_back(static_cast<int64_t>(index));
    if (auto nested = dyn_cast<hw::StructType>(field.type)) {
      Value nestedValue =
          buildHWStructFromLLVM(llvmStruct, nested, builder, loc, path);
      if (!nestedValue)
        return {};
      fields.push_back(nestedValue);
      continue;
    }
    Value leaf = findInsertedValue(llvmStruct, path);
    if (!leaf || leaf.getType() != field.type)
      return {};
    fields.push_back(leaf);
  }
  return hw::StructCreateOp::create(builder, loc, hwType, fields);
}

static Value adjustIntegerWidth(OpBuilder &builder, Value value,
                                unsigned targetWidth, Location loc) {
  auto intType = dyn_cast<IntegerType>(value.getType());
  if (!intType)
    return value;
  unsigned intWidth = intType.getWidth();
  if (intWidth == targetWidth)
    return value;
  if (intWidth < targetWidth) {
    Value zeroExt = hw::ConstantOp::create(
        builder, loc, builder.getIntegerType(targetWidth - intWidth), 0);
    return comb::ConcatOp::create(builder, loc, ValueRange{zeroExt, value});
  }
  return comb::ExtractOp::create(builder, loc, value, 0, targetWidth);
}

static Value createZeroValue(OpBuilder &builder, Location loc, Type type) {
  int64_t width = hw::getBitWidth(type);
  if (width <= 0)
    return Value();
  auto intType = IntegerType::get(builder.getContext(), width);
  Value zero = hw::ConstantOp::create(builder, loc, intType, 0);
  if (intType == type)
    return zero;
  return hw::BitcastOp::create(builder, loc, type, zero);
}

static LogicalResult lowerCombinationalOp(llhd::CombinationalOp combOp,
                                          ModuleState &state,
                                          bool strictMode) {
  auto module = combOp->getParentOfType<hw::HWModuleOp>();
  if (!module)
    return combOp.emitError("expected llhd.combinational in hw.module for LEC");

  Region &region = combOp.getBody();

  auto getLocalRefPtr = [](Value ref) -> Value {
    if (ref.getDefiningOp<llhd::SignalOp>())
      return {};
    if (auto castOp =
            ref.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1 &&
          isa<LLVM::LLVMPointerType>(castOp.getInputs()[0].getType()))
        return castOp.getInputs()[0];
    }
    return {};
  };

  auto getPtrValueType = [](Value ptr, Type fallback) -> Type {
    Type valueType;
    for (auto *user : ptr.getUsers()) {
      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        auto storeTy = store.getValue().getType();
        if (!valueType)
          valueType = storeTy;
        else if (valueType != storeTy)
          return {};
      } else if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
        auto loadTy = load.getResult().getType();
        if (!valueType)
          valueType = loadTy;
        else if (valueType != loadTy)
          return {};
      }
    }
    if (valueType)
      return valueType;
    if (fallback && LLVM::isCompatibleType(fallback))
      return fallback;
    return {};
  };

  SmallVector<llhd::ProbeOp> localProbes;
  region.walk([&](llhd::ProbeOp probe) {
    if (getLocalRefPtr(probe.getSignal()))
      localProbes.push_back(probe);
  });
  for (auto probe : localProbes) {
    Value ptr = getLocalRefPtr(probe.getSignal());
    if (!ptr)
      continue;
    auto ptrValueType = getPtrValueType(ptr, probe.getResult().getType());
    if (!ptrValueType)
      return probe.emitError(
          "unsupported llhd.probe on local ref without LLVM value type");
    OpBuilder builder(probe);
    Value loaded =
        LLVM::LoadOp::create(builder, probe.getLoc(), ptrValueType, ptr);
    if (loaded.getType() != probe.getResult().getType())
      loaded = UnrealizedConversionCastOp::create(
                   builder, probe.getLoc(), probe.getResult().getType(), loaded)
                   .getResult(0);
    probe.getResult().replaceAllUsesWith(loaded);
    probe.erase();
  }

  SmallVector<llhd::DriveOp> localDrives;
  region.walk([&](llhd::DriveOp drive) {
    if (getLocalRefPtr(drive.getSignal()))
      localDrives.push_back(drive);
  });
  for (auto drive : localDrives) {
    Value ptr = getLocalRefPtr(drive.getSignal());
    if (!ptr)
      continue;
    if (drive.getEnable())
      return drive.emitError(
          "unsupported conditional drive on local ref in LEC");
    if (!drive.getTime().getDefiningOp<llhd::ConstantTimeOp>())
      return drive.emitError(
          "unsupported non-constant drive time on local ref in LEC");
    auto ptrValueType = getPtrValueType(ptr, drive.getValue().getType());
    if (!ptrValueType)
      return drive.emitError(
          "unsupported llhd.drive on local ref without LLVM value type");
    OpBuilder builder(drive);
    Value storedValue = drive.getValue();
    if (storedValue.getType() != ptrValueType)
      storedValue = UnrealizedConversionCastOp::create(
                        builder, drive.getLoc(), ptrValueType, storedValue)
                        .getResult(0);
    LLVM::StoreOp::create(builder, drive.getLoc(), storedValue, ptr);
    drive.erase();
  }

  SmallVector<comb::MuxOp> refMuxes;
  region.walk([&](comb::MuxOp mux) {
    if (isa<llhd::RefType>(mux.getType()))
      refMuxes.push_back(mux);
  });
  for (auto mux : refMuxes) {
    if (!llvm::all_of(mux.getResult().getUsers(), [](Operation *user) {
          return isa<llhd::ProbeOp>(user);
        }))
      return mux.emitError("unsupported comb.mux on LLHD refs with non-probe "
                           "uses in LEC");
    Value cond = mux.getCond();
    Value trueRef = mux.getTrueValue();
    Value falseRef = mux.getFalseValue();
    for (auto *user : llvm::make_early_inc_range(mux.getResult().getUsers())) {
      auto probe = cast<llhd::ProbeOp>(user);
      OpBuilder builder(probe);
      Value trueValue = llhd::ProbeOp::create(builder, probe.getLoc(), trueRef);
      Value falseValue =
          llhd::ProbeOp::create(builder, probe.getLoc(), falseRef);
      Value valueMux = comb::MuxOp::create(builder, probe.getLoc(), cond,
                                           trueValue, falseValue);
      probe.getResult().replaceAllUsesWith(valueMux);
      probe.erase();
    }
    mux.erase();
  }

  // Clean up dead casts so they don't block mem2reg.
  SmallVector<UnrealizedConversionCastOp> deadCasts;
  region.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->use_empty())
      deadCasts.push_back(castOp);
  });
  for (auto castOp : deadCasts)
    castOp.erase();

  if (!region.empty()) {
    Block &entryBlock = region.front();
    for (auto alloca :
         llvm::make_early_inc_range(region.getOps<LLVM::AllocaOp>())) {
      if (alloca->getBlock() == &entryBlock)
        continue;
      bool canHoist = true;
      for (Value operand : alloca->getOperands()) {
        if (auto def = operand.getDefiningOp()) {
          if (def->getParentRegion() != &region)
            continue;
          if (def->getBlock() != &entryBlock) {
            if (def->hasTrait<OpTrait::ConstantLike>())
              def->moveBefore(&entryBlock, entryBlock.begin());
            else {
              canHoist = false;
              break;
            }
          }
        } else if (auto arg = dyn_cast<BlockArgument>(operand)) {
          if (arg.getOwner()->getParent() != &region)
            continue;
          if (arg.getOwner() != &entryBlock) {
            canHoist = false;
            break;
          }
        }
      }
      if (canHoist)
        alloca->moveBefore(&entryBlock, entryBlock.begin());
    }

    for (Block &block : region) {
      for (unsigned argIndex = 0; argIndex < block.getNumArguments();
           ++argIndex) {
        BlockArgument arg = block.getArgument(argIndex);
        if (!isa<LLVM::LLVMPointerType>(arg.getType()))
          continue;

        SmallVector<LLVM::LoadOp> loads;
        bool hasOtherUses = false;
        for (OpOperand &use : arg.getUses()) {
          auto load = dyn_cast<LLVM::LoadOp>(use.getOwner());
          if (!load || load.getAddr() != arg) {
            hasOtherUses = true;
            break;
          }
          loads.push_back(load);
        }
        if (hasOtherUses || loads.empty())
          continue;

        Type loadType = loads.front().getType();
        if (llvm::any_of(loads,
                         [&](LLVM::LoadOp load) {
                           return load.getType() != loadType;
                         }))
          continue;

        arg.setType(loadType);
        for (LLVM::LoadOp load : loads) {
          load.replaceAllUsesWith(arg);
          load.erase();
        }

        for (Block *pred : block.getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch)
            continue;
          OpBuilder builder(pred->getTerminator());
          for (unsigned succIndex = 0; succIndex < branch->getNumSuccessors();
               ++succIndex) {
            if (branch->getSuccessor(succIndex) != &block)
              continue;
            auto succOperands = branch.getSuccessorOperands(succIndex);
            if (argIndex >= succOperands.size())
              continue;
            if (succOperands.isOperandProduced(argIndex))
              continue;
            unsigned operandIndex = succOperands.getOperandIndex(argIndex);
            Value ptrValue = branch->getOperand(operandIndex);
            Value loaded =
                LLVM::LoadOp::create(builder, branch.getLoc(), loadType,
                                     ptrValue);
            branch->setOperand(operandIndex, loaded);
          }
        }
      }
    }

    OpBuilder builder(&entryBlock, entryBlock.begin());
    SmallVector<PromotableAllocationOpInterface> allocators;
    region.walk([&](PromotableAllocationOpInterface allocator) {
      allocators.push_back(allocator);
    });
    if (!allocators.empty()) {
      DataLayout dataLayout = DataLayout::closest(combOp);
      DominanceInfo dominance(combOp);
      (void)tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance);
    }
  }

  if (!llvm::hasSingleElement(region)) {
    CFRemover(region).run();
  }

  if (!llvm::hasSingleElement(region)) {
    if (strictMode)
      return combOp.emitError(
          "LLHD combinational control flow requires abstraction; rerun without "
          "--strict-llhd");
    SmallVector<Value> inputs;
    for (Type resultType : combOp.getResultTypes())
      inputs.push_back(state.addInput(module, "llhd_comb", resultType));
    combOp.replaceAllUsesWith(inputs);
    combOp.erase();
    return success();
  }
  Block &body = combOp.getBody().front();
  auto *terminator = body.getTerminator();
  auto yieldOp = dyn_cast<llhd::YieldOp>(terminator);
  if (!yieldOp)
    return combOp.emitError("expected llhd.yield terminator in combinational");

  IRMapping mapping;
  OpBuilder builder(combOp);
  for (auto &op : body.without_terminator())
    builder.clone(op, mapping);

  SmallVector<Value> replacements;
  for (Value operand : yieldOp.getOperands())
    replacements.push_back(mapping.lookupOrDefault(operand));

  combOp.replaceAllUsesWith(replacements);
  combOp.erase();
  return success();
}

static LogicalResult stripPlainSignal(llhd::SignalOp sigOp, DominanceInfo &dom,
                                      ModuleState &state, bool strictMode) {
  struct RefStep {
    enum Kind { StructField, Extract } kind;
    StringAttr field;
    Value index;
    Type elemType;
  };
  using RefPath = SmallVector<RefStep, 4>;
  auto pathsEqual = [&](const RefPath &lhs, const RefPath &rhs) -> bool {
    if (lhs.size() != rhs.size())
      return false;
    for (auto [left, right] : llvm::zip(lhs, rhs)) {
      if (left.kind != right.kind)
        return false;
      if (left.field != right.field)
        return false;
      if (left.index != right.index)
        return false;
      if (left.elemType != right.elemType)
        return false;
    }
    return true;
  };
  auto allDrivesSameValue = [&](ArrayRef<llhd::DriveOp> drives) -> bool {
    if (drives.empty())
      return false;
    Value first;
    for (auto drive : drives) {
      if (drive.getEnable())
        return false;
      if (!drive.getTime().getDefiningOp<llhd::ConstantTimeOp>())
        return false;
      Value val = unwrapStoredValue(drive.getValue());
      if (!first) {
        first = val;
        continue;
      }
      if (first != val)
        return false;
    }
    return true;
  };
  auto hasEnabledDrive = [&](ArrayRef<llhd::DriveOp> drives) -> bool {
    return llvm::any_of(drives, [](llhd::DriveOp drive) {
      return drive.getEnable() != nullptr;
    });
  };
  auto canResolveEnabledDrives = [&](ArrayRef<llhd::DriveOp> drives) -> bool {
    if (drives.size() < 2)
      return false;
    Type valueType;
    for (auto drive : drives) {
      Value val = unwrapStoredValue(drive.getValue());
      if (!val)
        return false;
      if (!valueType)
        valueType = val.getType();
      else if (val.getType() != valueType)
        return false;
    }
    return valueType && isFourStateStructType(valueType);
  };

  SmallVector<llhd::ProbeOp> probes;
  SmallVector<llhd::DriveOp> drives;
  SmallVector<BlockArgument> forwardedArgs;
  SmallVector<Value> worklist;
  llvm::SmallPtrSet<Value, 16> visited;
  DenseMap<Value, RefPath> refPaths;
  DenseMap<llhd::ProbeOp, RefPath> probePaths;
  DenseMap<llhd::DriveOp, RefPath> drivePaths;
  SmallVector<Operation *> derivedRefs;

  Value zeroTime;
  auto getZeroTime = [&]() -> Value {
    if (!zeroTime) {
      auto parentModule = sigOp->getParentOfType<hw::HWModuleOp>();
      OpBuilder timeBuilder(parentModule.getBodyBlock(),
                            parentModule.getBodyBlock()->begin());
      zeroTime = llhd::ConstantTimeOp::create(timeBuilder, sigOp.getLoc(), 0,
                                              "ns", 0, 1);
    }
    return zeroTime;
  };

  refPaths[sigOp.getResult()] = RefPath{};
  worklist.push_back(sigOp.getResult());
  while (!worklist.empty()) {
    Value ref = worklist.pop_back_val();
    if (!visited.insert(ref).second)
      continue;
    auto pathIt = refPaths.find(ref);
    if (pathIt == refPaths.end())
      return sigOp.emitError("missing ref path for LLHD signal");
    const RefPath &path = pathIt->second;
    for (auto *user : ref.getUsers()) {
      if (auto probe = dyn_cast<llhd::ProbeOp>(user)) {
        probes.push_back(probe);
        probePaths.try_emplace(probe, path);
        continue;
      }
      if (auto drive = dyn_cast<llhd::DriveOp>(user)) {
        if (drive.getSignal() != ref)
          return drive.emitError("unexpected drive target for signal");
        drives.push_back(drive);
        drivePaths.try_emplace(drive, path);
        continue;
      }
      if (auto extract = dyn_cast<llhd::SigStructExtractOp>(user)) {
        RefPath derived = path;
        derived.push_back(
            RefStep{RefStep::StructField, extract.getFieldAttr(), Value(), {}});
        refPaths[extract.getResult()] = derived;
        derivedRefs.push_back(extract);
        worklist.push_back(extract.getResult());
        continue;
      }
      if (auto extract = dyn_cast<llhd::SigExtractOp>(user)) {
        RefPath derived = path;
        auto resultRefType =
            dyn_cast<llhd::RefType>(extract.getResult().getType());
        if (!resultRefType)
          return extract.emitError("expected llhd.ref result for sig.extract");
        derived.push_back(RefStep{RefStep::Extract, {}, extract.getLowBit(),
                                  resultRefType.getNestedType()});
        refPaths[extract.getResult()] = derived;
        derivedRefs.push_back(extract);
        worklist.push_back(extract.getResult());
        continue;
      }
      if (auto mux = dyn_cast<comb::MuxOp>(user)) {
        if (mux.getTrueValue() != ref && mux.getFalseValue() != ref)
          return sigOp.emitError()
                 << "unexpected comb.mux use for LLHD signal";
        Value other =
            mux.getTrueValue() == ref ? mux.getFalseValue() : mux.getTrueValue();
        auto otherPathIt = refPaths.find(other);
        if (otherPathIt == refPaths.end() ||
            !pathsEqual(otherPathIt->second, path))
          return sigOp.emitError()
                 << "unsupported comb.mux on divergent LLHD refs";
        refPaths[mux.getResult()] = path;
        worklist.push_back(mux.getResult());
        continue;
      }
      if (auto br = dyn_cast<cf::BranchOp>(user)) {
        auto operands = br.getDestOperands();
        auto it = llvm::find(operands, ref);
        if (it == operands.end())
          return br.emitError("expected signal operand for branch");
        auto index = static_cast<unsigned>(std::distance(operands.begin(), it));
        auto arg = br.getDest()->getArgument(index);
        forwardedArgs.push_back(arg);
        refPaths[arg] = path;
        worklist.push_back(arg);
        continue;
      }
      if (auto br = dyn_cast<cf::CondBranchOp>(user)) {
        auto trueOperands = br.getTrueDestOperands();
        if (auto it = llvm::find(trueOperands, ref);
            it != trueOperands.end()) {
          auto index =
              static_cast<unsigned>(std::distance(trueOperands.begin(), it));
          auto arg = br.getTrueDest()->getArgument(index);
          forwardedArgs.push_back(arg);
          refPaths[arg] = path;
          worklist.push_back(arg);
          continue;
        }
        auto falseOperands = br.getFalseDestOperands();
        if (auto it = llvm::find(falseOperands, ref);
            it != falseOperands.end()) {
          auto index =
              static_cast<unsigned>(std::distance(falseOperands.begin(), it));
          auto arg = br.getFalseDest()->getArgument(index);
          forwardedArgs.push_back(arg);
          refPaths[arg] = path;
          worklist.push_back(arg);
          continue;
        }
        return br.emitError("expected signal operand for branch");
      }
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
        if (cast.getInputs().size() != 1 || cast.getResults().size() != 1 ||
            cast.getInputs().front() != ref)
          return cast.emitError(
              "unsupported unrealized conversion on LLHD signal in LEC");
        if (!isa<LLVM::LLVMPointerType>(cast.getResult(0).getType()))
          return cast.emitError(
              "unsupported LLHD signal cast without LLVM pointer type");
        SmallVector<Operation *, 4> castUsers(
            llvm::to_vector(cast.getResult(0).getUsers()));
        for (Operation *castUser : castUsers) {
          if (auto load = dyn_cast<LLVM::LoadOp>(castUser)) {
            OpBuilder builder(load);
            auto probe =
                llhd::ProbeOp::create(builder, load.getLoc(), ref);
            Value probeValue = probe.getResult();
            probes.push_back(probe);
            probePaths.try_emplace(probe, path);
            if (probeValue.getType() == load.getType()) {
              load.getResult().replaceAllUsesWith(probeValue);
              load.erase();
              continue;
            }
            SmallVector<Operation *, 4> loadUsers(
                llvm::to_vector(load.getResult().getUsers()));
            for (Operation *loadUser : loadUsers) {
              auto castOut = dyn_cast<UnrealizedConversionCastOp>(loadUser);
              if (!castOut || castOut.getInputs().size() != 1 ||
                  castOut.getResults().size() != 1)
                return load.emitError(
                    "unsupported load from LLHD signal in LEC");
              if (castOut.getResult(0).getType() != probeValue.getType())
                return castOut.emitError(
                    "unsupported load conversion for LLHD signal in LEC");
              castOut.getResult(0).replaceAllUsesWith(probeValue);
              castOut.erase();
            }
            load.erase();
            continue;
          }
          if (auto store = dyn_cast<LLVM::StoreOp>(castUser)) {
            OpBuilder builder(store);
            auto refType = dyn_cast<llhd::RefType>(ref.getType());
            if (!refType)
              return store.emitError(
                  "expected llhd.ref for stored LLHD signal");
            Type targetType = refType.getNestedType();
            UnrealizedConversionCastOp storedCast;
            Value storedValue = store.getValue();
            if (auto castOp =
                    storedValue.getDefiningOp<UnrealizedConversionCastOp>()) {
              if (castOp.getInputs().size() == 1 &&
                  castOp.getResults().size() == 1)
                storedCast = castOp;
            }
            if (storedValue.getType() != targetType)
              storedValue = unwrapStoredValue(storedValue);
            if (storedValue.getType() != targetType) {
              if (auto hwType = dyn_cast<hw::StructType>(targetType)) {
                if (isa<LLVM::LLVMStructType>(storedValue.getType())) {
                  Value rebuilt = buildHWStructFromLLVM(
                      storedValue, hwType, builder, store.getLoc(), {});
                  if (rebuilt)
                    storedValue = rebuilt;
                }
              }
            }
            if (storedValue.getType() != targetType)
              return store.emitError(
                  "unsupported store to LLHD signal in LEC");
            auto drive = llhd::DriveOp::create(
                builder, store.getLoc(), ref, storedValue, getZeroTime(),
                Value{});
            drives.push_back(drive);
            drivePaths.try_emplace(drive, path);
            store.erase();
            if (storedCast && storedCast.use_empty())
              storedCast.erase();
            continue;
          }
          return castUser->emitError(
              "unsupported LLVM user of LLHD signal cast in LEC");
        }
        if (cast.use_empty())
          cast.erase();
        continue;
      }
      return sigOp.emitError()
             << "unsupported LLHD signal use in LEC: " << user->getName();
    }
  }

  OpBuilder builder(sigOp);
  auto materializePath = [&](Value baseValue, const RefPath &path,
                             Location loc) -> Value {
    Value value = baseValue;
    for (const auto &step : path) {
      if (step.kind == RefStep::StructField) {
        value = hw::StructExtractOp::create(builder, loc, value, step.field);
        continue;
      }
      if (step.kind == RefStep::Extract) {
        auto elemType = dyn_cast<IntegerType>(step.elemType);
        auto baseType = dyn_cast<IntegerType>(value.getType());
        if (!elemType || !baseType)
          return Value();
        if (auto constant = step.index.getDefiningOp<hw::ConstantOp>()) {
          uint64_t low = constant.getValue().getZExtValue();
          value = comb::ExtractOp::create(builder, loc, value, low,
                                          elemType.getWidth());
          continue;
        }
        Value shift =
            adjustIntegerWidth(builder, step.index, baseType.getWidth(), loc);
        Value shifted = comb::ShrUOp::create(builder, loc, value, shift);
        value = comb::ExtractOp::create(builder, loc, shifted, 0,
                                        elemType.getWidth());
        continue;
      }
    }
    return value;
  };

  bool canInline = forwardedArgs.empty();
  Block *singleBlock = nullptr;
  for (auto probe : probes) {
    if (!probePaths.lookup(probe).empty())
      canInline = false;
    Block *block = probe->getBlock();
    if (!singleBlock)
      singleBlock = block;
    else if (singleBlock != block)
      canInline = false;
  }
  for (auto driveOp : drives) {
    if (!drivePaths.lookup(driveOp).empty())
      canInline = false;
    if (driveOp.getEnable())
      canInline = false;
    if (!driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>())
      canInline = false;
    Block *block = driveOp->getBlock();
    if (!singleBlock)
      singleBlock = block;
    else if (singleBlock != block)
      canInline = false;
  }

  if (drives.size() == 1) {
    llhd::DriveOp driveOp = drives.front();
    if (drivePaths.lookup(driveOp).empty() && !driveOp.getEnable() &&
        driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>() &&
        isa<BlockArgument>(driveOp.getValue())) {
      Value drivenValue = driveOp.getValue();
      for (auto probe : probes) {
        Value replacement = materializePath(drivenValue,
                                            probePaths.lookup(probe),
                                            probe.getLoc());
        if (!replacement)
          return probe.emitError("unsupported LLHD probe path in LEC");
        if (replacement.getType() != probe.getResult().getType())
          return probe.emitError("signal probe type mismatch in LEC");
        probe.getResult().replaceAllUsesWith(replacement);
        probe.erase();
      }
      driveOp.erase();
      for (Operation *refOp : llvm::reverse(derivedRefs)) {
        if (refOp->use_empty())
          refOp->erase();
      }
      if (sigOp.use_empty()) {
        Value init = sigOp.getInit();
        sigOp.erase();
        if (auto *def = init.getDefiningOp())
          if (def->use_empty())
            def->erase();
      }
      return success();
    }
  }

  if (canInline && singleBlock) {
    // In strict mode, reject multiple unconditional drives as requiring
    // abstraction since simultaneous drives have undefined priority.
    if (strictMode && drives.size() > 1 && hasEnabledDrive(drives))
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");

    if (drives.size() > 1 && !allDrivesSameValue(drives)) {
      bool probesAfterDrives = true;
      Operation *firstProbe = nullptr;
      bool sawProbe = false;
      for (auto &op : *singleBlock) {
        if (auto probe = dyn_cast<llhd::ProbeOp>(&op)) {
          if (probe.getSignal() == sigOp.getResult()) {
            sawProbe = true;
            if (!firstProbe)
              firstProbe = probe.getOperation();
          }
        }
        if (auto drive = dyn_cast<llhd::DriveOp>(&op)) {
          if (drive.getSignal() == sigOp.getResult() && sawProbe) {
            probesAfterDrives = false;
            break;
          }
        }
      }

      bool inputsDominateProbe = false;
      if (firstProbe && !probesAfterDrives) {
        auto dominatesProbe = [&](Value value) -> bool {
          if (!value)
            return false;
          if (auto arg = dyn_cast<BlockArgument>(value))
            return arg.getOwner() == firstProbe->getBlock();
          if (auto *def = value.getDefiningOp()) {
            if (def->getBlock() == firstProbe->getBlock())
              return def->isBeforeInBlock(firstProbe);
            return dom.dominates(def, firstProbe);
          }
          return false;
        };
        inputsDominateProbe =
            llvm::all_of(drives, [&](llhd::DriveOp drive) {
              if (!dominatesProbe(drive.getValue()))
                return false;
              if (Value enable = drive.getEnable())
                return dominatesProbe(enable);
              return true;
            });
      }

      if ((probesAfterDrives || inputsDominateProbe) && firstProbe) {
        SmallVector<Value> driveValues;
        SmallVector<Value> driveEnables;
        SmallVector<unsigned> driveStrength0;
        SmallVector<unsigned> driveStrength1;
        bool anyStrength = false;
        driveValues.reserve(drives.size());
        driveEnables.reserve(drives.size());
        driveStrength0.reserve(drives.size());
        driveStrength1.reserve(drives.size());
        OpBuilder resolveBuilder(firstProbe);
        Value enableTrue = hw::ConstantOp::create(
                               resolveBuilder, firstProbe->getLoc(),
                               resolveBuilder.getI1Type(), 1)
                               .getResult();
        for (auto drive : drives)
          driveValues.push_back(drive.getValue());
        unsigned defaultStrength =
            static_cast<unsigned>(llhd::DriveStrength::Strong);
        for (auto drive : drives) {
          driveEnables.push_back(drive.getEnable() ? drive.getEnable()
                                                   : enableTrue);
          auto s0Attr =
              drive->getAttrOfType<llhd::DriveStrengthAttr>("strength0");
          auto s1Attr =
              drive->getAttrOfType<llhd::DriveStrengthAttr>("strength1");
          if (s0Attr || s1Attr)
            anyStrength = true;
          driveStrength0.push_back(
              s0Attr ? static_cast<unsigned>(s0Attr.getValue())
                     : defaultStrength);
          driveStrength1.push_back(
              s1Attr ? static_cast<unsigned>(s1Attr.getValue())
                     : defaultStrength);
        }
        Value resolved = anyStrength
                             ? resolveFourStateValuesWithStrength(
                                   resolveBuilder, firstProbe->getLoc(),
                                   driveValues, driveEnables, driveStrength0,
                                   driveStrength1,
                                   static_cast<unsigned>(
                                       llhd::DriveStrength::HighZ))
                             : resolveFourStateValuesWithEnable(
                                   resolveBuilder, firstProbe->getLoc(),
                                   driveValues, driveEnables);
        if (resolved) {
          for (auto probe : probes) {
            probe.getResult().replaceAllUsesWith(resolved);
            probe.erase();
          }
          for (auto drive : drives)
            drive.erase();
          for (Operation *refOp : llvm::reverse(derivedRefs)) {
            if (refOp->use_empty())
              refOp->erase();
          }
          if (sigOp.use_empty()) {
            Value init = sigOp.getInit();
            sigOp.erase();
            if (auto *def = init.getDefiningOp())
              if (def->use_empty())
                def->erase();
          }
          return success();
        }
      }

      if (strictMode)
        return sigOp.emitError(
            "LLHD signal requires abstraction; rerun without --strict-llhd");
    }

    Value current = sigOp.getInit();
    for (auto it = singleBlock->begin(); it != singleBlock->end();) {
      Operation *op = &*it++;
      if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
        if (driveOp.getSignal() == sigOp.getResult()) {
          current = driveOp.getValue();
          driveOp.erase();
        }
        continue;
      }
      if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
        if (probe.getSignal() == sigOp.getResult()) {
          probe.getResult().replaceAllUsesWith(current);
          probe.erase();
        }
        continue;
      }
    }
    for (Operation *refOp : llvm::reverse(derivedRefs)) {
      if (refOp->use_empty())
        refOp->erase();
    }
    if (sigOp.use_empty()) {
      Value init = sigOp.getInit();
      sigOp.erase();
      if (auto *def = init.getDefiningOp())
        if (def->use_empty())
          def->erase();
    }
    return success();
  }

  // Handle multiple drives with enable conditions by creating a mux chain.
  // This handles the case where CFRemover has added enable conditions to
  // drives from different branches.
  if (forwardedArgs.empty() && singleBlock && !drives.empty()) {
    bool allDrivesHaveTime = llvm::all_of(drives, [](llhd::DriveOp d) {
      return d.getTime().getDefiningOp<llhd::ConstantTimeOp>() != nullptr;
    });
    bool allDrivesSimplePath = llvm::all_of(drives, [&](llhd::DriveOp d) {
      return drivePaths.lookup(d).empty();
    });
    bool allProbesSimplePath = llvm::all_of(probes, [&](llhd::ProbeOp p) {
      return probePaths.lookup(p).empty();
    });
    bool resolveEnabledDrives =
        strictMode && canResolveEnabledDrives(drives) &&
        hasEnabledDrive(drives);
    // In strict mode, reject multiple conditional drives unless they are
    // complementary enables on exactly two drives.
    bool hasMultipleUnconditionalDrives =
        drives.size() > 1 &&
        llvm::any_of(drives, [](llhd::DriveOp d) { return !d.getEnable(); });
    if (strictMode && drives.size() > 1 && hasEnabledDrive(drives) &&
        !resolveEnabledDrives && !hasComplementaryEnables(drives) &&
        !hasExclusiveEnables(drives, builder))
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");
    if (strictMode && hasMultipleUnconditionalDrives &&
        !allDrivesSameValue(drives))
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");

    if (allDrivesHaveTime && allDrivesSimplePath && allProbesSimplePath) {
      if (resolveEnabledDrives) {
        SmallVector<Value> activeValues;
        SmallVector<Value> activeEnables;
        SmallVector<unsigned> activeStrength0;
        SmallVector<unsigned> activeStrength1;
        bool activeAnyStrength = false;
        Value enableTrue;
        auto getEnableTrue = [&]() -> Value {
          if (enableTrue)
            return enableTrue;
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(singleBlock);
          enableTrue = hw::ConstantOp::create(
                           builder, sigOp.getLoc(), builder.getI1Type(), 1)
                           .getResult();
          return enableTrue;
        };
        unsigned defaultStrength =
            static_cast<unsigned>(llhd::DriveStrength::Strong);

        for (auto it = singleBlock->begin(); it != singleBlock->end();) {
          Operation *op = &*it++;
          if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
            if (driveOp.getSignal() == sigOp.getResult()) {
              Value driveValue = unwrapStoredValue(driveOp.getValue());
              if (!driveValue)
                return driveOp.emitError(
                    "failed to resolve LLHD drive value");
              Value enable =
                  driveOp.getEnable() ? driveOp.getEnable() : getEnableTrue();
              auto s0Attr =
                  driveOp->getAttrOfType<llhd::DriveStrengthAttr>("strength0");
              auto s1Attr =
                  driveOp->getAttrOfType<llhd::DriveStrengthAttr>("strength1");
              if (s0Attr || s1Attr)
                activeAnyStrength = true;
              activeValues.push_back(driveValue);
              activeEnables.push_back(enable);
              activeStrength0.push_back(
                  s0Attr ? static_cast<unsigned>(s0Attr.getValue())
                         : defaultStrength);
              activeStrength1.push_back(
                  s1Attr ? static_cast<unsigned>(s1Attr.getValue())
                         : defaultStrength);
              driveOp.erase();
            }
            continue;
          }
          if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
            if (probe.getSignal() == sigOp.getResult()) {
              Value replacement;
              if (activeValues.empty()) {
                replacement = sigOp.getInit();
              } else {
                builder.setInsertionPoint(probe);
                replacement =
                    activeAnyStrength
                        ? resolveFourStateValuesWithStrength(
                              builder, probe.getLoc(), activeValues,
                              activeEnables, activeStrength0, activeStrength1,
                              static_cast<unsigned>(
                                  llhd::DriveStrength::HighZ))
                        : resolveFourStateValuesWithEnable(
                              builder, probe.getLoc(), activeValues,
                              activeEnables);
              }
              if (!replacement ||
                  replacement.getType() != probe.getResult().getType())
                return probe.emitError(
                    "failed to resolve LLHD multi-drive signal value");
              probe.getResult().replaceAllUsesWith(replacement);
              probe.erase();
            }
            continue;
          }
        }
        for (Operation *refOp : llvm::reverse(derivedRefs)) {
          if (refOp->use_empty())
            refOp->erase();
        }
        if (sigOp.use_empty()) {
          Value init = sigOp.getInit();
          sigOp.erase();
          if (auto *def = init.getDefiningOp())
            if (def->use_empty())
              def->erase();
        }
        return success();
      }

      // Build a mux chain: for each drive, if enabled select its value,
      // otherwise keep the previous value.
      Value current = sigOp.getInit();
      for (auto it = singleBlock->begin(); it != singleBlock->end();) {
        Operation *op = &*it++;
        if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
          if (driveOp.getSignal() == sigOp.getResult()) {
            if (Value enable = driveOp.getEnable()) {
              // current = enable ? driveValue : current
              builder.setInsertionPoint(driveOp);
              current = builder.createOrFold<comb::MuxOp>(
                  driveOp.getLoc(), enable, driveOp.getValue(), current);
            } else {
              current = driveOp.getValue();
            }
            driveOp.erase();
          }
          continue;
        }
        if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
          if (probe.getSignal() == sigOp.getResult()) {
            probe.getResult().replaceAllUsesWith(current);
            probe.erase();
          }
          continue;
        }
      }
      for (Operation *refOp : llvm::reverse(derivedRefs)) {
        if (refOp->use_empty())
          refOp->erase();
      }
      if (sigOp.use_empty()) {
        Value init = sigOp.getInit();
        sigOp.erase();
        if (auto *def = init.getDefiningOp())
          if (def->use_empty())
            def->erase();
      }
      return success();
    }
  }

  llhd::DriveOp drive;
  bool needsInput = false;
  if (!drives.empty()) {
    if (drives.size() != 1)
      needsInput = true;
    drive = drives.front();
    if (!drivePaths.lookup(drive).empty())
      needsInput = true;
    if (drive.getEnable())
      needsInput = true;
    if (!drive.getTime().getDefiningOp<llhd::ConstantTimeOp>())
      needsInput = true;
  }

  Value drivenValue = drive ? drive.getValue() : sigOp.getInit();
  if (!needsInput) {
    for (auto probe : probes) {
      if (!dom.dominates(drivenValue, probe.getOperation())) {
        needsInput = true;
        break;
      }
    }
  }

  if (needsInput) {
    if (strictMode)
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");
    auto module = sigOp->getParentOfType<hw::HWModuleOp>();
    if (!module)
      return sigOp.emitError("expected LLHD signal in hw.module for LEC");
    auto baseName = sigOp.getNameAttr()
                        ? sigOp.getNameAttr().getValue()
                        : StringRef("llhd_sig");
    Value newInput = state.addInput(module, baseName, sigOp.getInit().getType());
    for (auto probe : probes) {
      Value replacement =
          materializePath(newInput, probePaths.lookup(probe), probe.getLoc());
      if (!replacement)
        return probe.emitError("unsupported LLHD probe path in LEC");
      if (replacement.getType() != probe.getResult().getType())
        return probe.emitError("signal probe type mismatch in LEC");
      probe.getResult().replaceAllUsesWith(replacement);
      probe.erase();
    }
    for (auto driveOp : drives)
      driveOp.erase();
    for (Operation *refOp : llvm::reverse(derivedRefs)) {
      if (refOp->use_empty())
        refOp->erase();
    }
    if (sigOp.use_empty()) {
      Value init = sigOp.getInit();
      sigOp.erase();
      if (auto *def = init.getDefiningOp())
        if (def->use_empty())
          def->erase();
    }
    return success();
  }

  for (auto probe : probes) {
    Value replacement =
        materializePath(drivenValue, probePaths.lookup(probe), probe.getLoc());
    if (!replacement)
      return probe.emitError("unsupported LLHD probe path in LEC");
    if (replacement.getType() != probe.getResult().getType())
      return probe.emitError("signal probe type mismatch in LEC");
    probe.getResult().replaceAllUsesWith(replacement);
    probe.erase();
  }

  if (drive)
    drive.erase();

  for (Operation *refOp : llvm::reverse(derivedRefs)) {
    if (refOp->use_empty())
      refOp->erase();
  }

  if (!forwardedArgs.empty()) {
    DenseMap<Block *, SmallVector<unsigned>> argsByBlock;
    for (auto arg : forwardedArgs) {
      if (!arg.use_empty())
        continue;
      argsByBlock[arg.getOwner()].push_back(arg.getArgNumber());
    }
    for (auto &entry : argsByBlock) {
      auto *block = entry.first;
      auto &indices = entry.second;
      llvm::sort(indices, std::greater<>());
      for (unsigned index : indices) {
        for (auto *pred : block->getPredecessors()) {
          auto *terminator = pred->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
            if (br.getDest() != block)
              continue;
            auto operands = llvm::to_vector(br.getDestOperands());
            operands.erase(operands.begin() + index);
            br.getDestOperandsMutable().assign(operands);
            continue;
          }
          if (auto br = dyn_cast<cf::CondBranchOp>(terminator)) {
            if (br.getTrueDest() == block) {
              auto operands = llvm::to_vector(br.getTrueDestOperands());
              operands.erase(operands.begin() + index);
              br.getTrueDestOperandsMutable().assign(operands);
              continue;
            }
            if (br.getFalseDest() == block) {
              auto operands = llvm::to_vector(br.getFalseDestOperands());
              operands.erase(operands.begin() + index);
              br.getFalseDestOperandsMutable().assign(operands);
              continue;
            }
            continue;
          }
          return terminator->emitError("unsupported predecessor for LLHD ref");
        }
        block->eraseArgument(index);
      }
    }
  }

  if (sigOp.use_empty()) {
    Value init = sigOp.getInit();
    sigOp.erase();
    if (auto *def = init.getDefiningOp())
      if (def->use_empty())
        def->erase();
  }

  return success();
}

struct StripLLHDInterfaceSignalsPass
    : public circt::impl::StripLLHDInterfaceSignalsBase<
          StripLLHDInterfaceSignalsPass> {
  using StripLLHDInterfaceSignalsBase::StripLLHDInterfaceSignalsBase;
  void runOnOperation() override;
};
} // namespace

static LogicalResult
stripInterfaceSignal(llhd::SignalOp sigOp, DominanceInfo &dom,
                     ModuleState &state, bool strictMode) {
  bool globalZeroInit = false;
  if (auto addr = sigOp.getInit().getDefiningOp<LLVM::AddressOfOp>()) {
    if (auto module = sigOp->getParentOfType<ModuleOp>()) {
      if (auto global = module.lookupSymbol<LLVM::GlobalOp>(
              addr.getGlobalNameAttr().getValue())) {
        if (auto valueAttr = global.getValueOrNull())
          globalZeroInit = isa<LLVM::ZeroAttr>(valueAttr);
      }
    }
  }
  auto refType = dyn_cast<llhd::RefType>(sigOp.getType());
  if (!refType)
    return success();
  if (!isa<LLVM::LLVMPointerType>(refType.getNestedType()))
    return stripPlainSignal(sigOp, dom, state, strictMode);

  SmallVector<llhd::ProbeOp> ptrProbes;
  for (auto *user : sigOp->getUsers()) {
    auto probe = dyn_cast<llhd::ProbeOp>(user);
    if (!probe)
      return sigOp.emitError("unsupported LLHD signal use in LEC");
    ptrProbes.push_back(probe);
  }

  DenseMap<unsigned, FieldAccess> fields;
  llvm::SmallPtrSet<Operation *, 16> eraseSet;
  SmallVector<Operation *> eraseOrder;
  auto markErase = [&](Operation *op) {
    if (eraseSet.insert(op).second)
      eraseOrder.push_back(op);
  };

  for (auto probe : ptrProbes) {
    for (Operation *user : probe.getResult().getUsers()) {
      auto gep = dyn_cast<LLVM::GEPOp>(user);
      if (!gep)
        return probe.emitError("unsupported LLHD probe use in LEC");

      auto fieldIndex = getStructFieldIndex(gep);
      if (!fieldIndex)
        return gep.emitError("unsupported GEP pattern for interface signal");

      auto &field = fields[*fieldIndex];
      for (Operation *gepUser : gep.getResult().getUsers()) {
        if (auto store = dyn_cast<LLVM::StoreOp>(gepUser)) {
          if (store.getAddr() != gep.getResult())
            return store.emitError("unexpected store address for interface");
          Value stored = unwrapStoredValue(store.getValue());
          field.stores.push_back(store);
          if (!field.storedValue)
            field.storedValue = stored;
          else if (field.storedValue != stored)
            field.hasMultipleStoreValues = true;
          markErase(store);
          continue;
        }
        if (auto cast = dyn_cast<UnrealizedConversionCastOp>(gepUser)) {
          if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
            return cast.emitError("unsupported cast for interface signal");
          if (!isa<llhd::RefType>(cast->getResult(0).getType()))
            return cast.emitError("expected cast to llhd.ref");
          for (auto *castUser : cast->getUsers()) {
            auto read = dyn_cast<llhd::ProbeOp>(castUser);
            if (!read)
              return cast.emitError("unsupported cast use for interface");
            field.reads.push_back(read);
            markErase(read);
          }
          markErase(cast);
          continue;
        }
        return gep.emitError("unsupported interface signal use in LEC");
      }
      markErase(gep);
    }
    markErase(probe);
  }

  for (auto &entry : fields) {
    auto &field = entry.second;
    if (field.reads.empty())
      continue;
    bool needsAbstraction = field.needsAbstraction;
    if (!field.storedValue)
      needsAbstraction = true;
    if (field.storedValue &&
        field.storedValue.getType() != field.reads.front().getType())
      needsAbstraction = true;

    auto resolveStoresByDominance = [&]() -> bool {
      if (field.stores.empty() || field.reads.empty())
        return false;
      DenseMap<Operation *, Value> storeValues;
      for (auto store : field.stores)
        storeValues[store.getOperation()] = unwrapStoredValue(store.getValue());

      DenseMap<Operation *, Value> readValues;
      auto dominatesOp = [&](Operation *domOp, Operation *useOp) -> bool {
        if (domOp->getBlock() == useOp->getBlock())
          return domOp->isBeforeInBlock(useOp);
        return dom.dominates(domOp, useOp);
      };

      for (auto read : field.reads) {
        Operation *bestStore = nullptr;
        for (auto store : field.stores) {
          Operation *storeOp = store.getOperation();
          if (!dominatesOp(storeOp, read.getOperation()))
            continue;
          if (!bestStore) {
            bestStore = storeOp;
            continue;
          }
          if (dominatesOp(bestStore, storeOp)) {
            bestStore = storeOp;
            continue;
          }
          if (dominatesOp(storeOp, bestStore))
            continue;
          return false;
        }
        Value replacement;
        if (!bestStore) {
          if (!globalZeroInit)
            return false;
          OpBuilder builder(read);
          replacement =
              createZeroValue(builder, read.getLoc(), read.getResult().getType());
          if (!replacement)
            return false;
        } else {
          replacement = storeValues.lookup(bestStore);
        }
        if (!replacement || replacement.getType() != read.getResult().getType())
          return false;
        readValues[read.getOperation()] = replacement;
      }
      for (auto read : field.reads) {
        Value replacement = readValues.lookup(read.getOperation());
        if (!replacement)
          return false;
        read.getResult().replaceAllUsesWith(replacement);
      }
      return true;
    };

    auto resolveStoresByComplementaryConditions = [&]() -> bool {
      if (field.stores.size() != 2 || field.reads.empty())
        return false;

      auto module = sigOp->getParentOfType<hw::HWModuleOp>();
      if (!module)
        return false;
      Block *entryBlock = &module.getBody().front();
      Block *readBlock = field.reads.front()->getBlock();
      for (auto read : field.reads)
        if (read->getBlock() != readBlock)
          return false;

      OpBuilder builder(field.reads.front());
      auto storeA = field.stores[0];
      auto storeB = field.stores[1];

      auto getScfIfGuard = [](LLVM::StoreOp store)
          -> std::optional<std::pair<scf::IfOp, bool>> {
        if (auto ifOp = store->getParentOfType<scf::IfOp>()) {
          if (store->getParentRegion() == &ifOp.getThenRegion())
            return std::make_pair(ifOp, true);
          if (store->getParentRegion() == &ifOp.getElseRegion())
            return std::make_pair(ifOp, false);
        }
        return std::nullopt;
      };

      std::optional<BoolCondition> condA;
      std::optional<BoolCondition> condB;
      auto ifGuardA = getScfIfGuard(storeA);
      auto ifGuardB = getScfIfGuard(storeB);
      if (ifGuardA && ifGuardB && ifGuardA->first == ifGuardB->first &&
          ifGuardA->second != ifGuardB->second) {
        auto ifOp = ifGuardA->first;
        if (!dom.dominates(ifOp, field.reads.front().getOperation()))
          return false;
        BoolCondition baseCond(ifOp.getCondition());
        condA = ifGuardA->second ? baseCond : baseCond.inverted(builder);
        condB = ifGuardB->second ? baseCond : baseCond.inverted(builder);
      } else {
        PostDominanceInfo postDom(module);
        for (auto store : field.stores)
          if (!postDom.postDominates(readBlock, store->getBlock()))
            return false;
        SmallDenseMap<std::pair<Block *, Block *>, BoolCondition> decisionCache;
        condA = getBranchDecisionsFromDominatorToTarget(builder, entryBlock,
                                                        storeA->getBlock(),
                                                        decisionCache);
        condB = getBranchDecisionsFromDominatorToTarget(builder, entryBlock,
                                                        storeB->getBlock(),
                                                        decisionCache);
      }

      if (!condA || !condB)
        return false;
      if (!areComplementary(*condA, *condB))
        return false;

      Value valA = unwrapStoredValue(storeA.getValue());
      Value valB = unwrapStoredValue(storeB.getValue());
      auto *readOp = field.reads.front().getOperation();
      if (!dom.dominates(valA, readOp) || !dom.dominates(valB, readOp))
        return false;

      BoolCondition trueCond = *condA;
      Value trueVal = valA;
      Value falseVal = valB;
      if (condA->isFalse()) {
        trueCond = *condB;
        trueVal = valB;
        falseVal = valA;
      }
      Value condValue = trueCond.materialize(builder, readOp->getLoc());
      if (!dom.dominates(condValue, readOp))
        return false;

      Value mux = builder.createOrFold<comb::MuxOp>(
          readOp->getLoc(), condValue, trueVal, falseVal);
      for (auto read : field.reads)
        read.getResult().replaceAllUsesWith(mux);
      return true;
    };

    auto resolveStoresByExclusiveConditions = [&]() -> bool {
      if (field.stores.size() < 2 || field.reads.empty())
        return false;

      auto module = sigOp->getParentOfType<hw::HWModuleOp>();
      if (!module)
        return false;

      Block *readBlock = field.reads.front()->getBlock();
      for (auto read : field.reads)
        if (read->getBlock() != readBlock)
          return false;

      OpBuilder builder(field.reads.front());
      Operation *readOp = field.reads.front().getOperation();

      struct GuardTerm {
        Value cond;
        bool inverted = false;
      };

      DenseMap<Operation *, unsigned> ifBranchCoverage;

      auto getScfIfGuards = [&](LLVM::StoreOp store,
                                SmallVectorImpl<GuardTerm> &guards) -> bool {
        bool sawIf = false;
        Operation *cursor = store.getOperation();
        while (Operation *parent = cursor->getParentOp()) {
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            Region *parentRegion = cursor->getParentRegion();
            bool inThen = ifOp.getThenRegion().isAncestor(parentRegion);
            bool inElse = ifOp.getElseRegion().isAncestor(parentRegion);
            if (!inThen && !inElse)
              return false;
            unsigned &mask = ifBranchCoverage[ifOp.getOperation()];
            mask |= inThen ? 1u : 2u;
            guards.push_back({ifOp.getCondition(), inElse});
            sawIf = true;
          }
          cursor = parent;
        }
        return sawIf;
      };

      SmallVector<BoolCondition, 4> conditions;
      SmallVector<Value, 4> values;
      conditions.reserve(field.stores.size());
      values.reserve(field.stores.size());
      SmallVector<SmallVector<GuardTerm, 4>, 4> guardSets;
      guardSets.reserve(field.stores.size());
      bool hasScfIfConditions = true;
      for (auto store : field.stores) {
        SmallVector<GuardTerm, 4> guards;
        if (!getScfIfGuards(store, guards)) {
          hasScfIfConditions = false;
          break;
        }
        guardSets.push_back(std::move(guards));
        values.push_back(unwrapStoredValue(store.getValue()));
      }

      if (!hasScfIfConditions) {
        struct Literal {
          Value cond;
          bool inverted = false;
        };

        conditions.clear();
        values.clear();
        SmallVector<SmallVector<Literal, 4>, 4> literalSets;
        literalSets.reserve(field.stores.size());

        PostDominanceInfo postDom(module);
        for (auto store : field.stores)
          if (!postDom.postDominates(readBlock, store->getBlock()))
            return false;

        SmallVector<cf::CondBranchOp> condBranches;
        module.walk([&](cf::CondBranchOp op) { condBranches.push_back(op); });

        for (auto store : field.stores) {
          SmallVector<Literal, 4> literals;
          Block *storeBlock = store->getBlock();
          for (auto condBr : condBranches) {
            if (condBr.getTrueDest() == condBr.getFalseDest())
              continue;
            if (!dom.dominates(condBr->getBlock(), storeBlock))
              continue;
            bool postTrue =
                postDom.postDominates(storeBlock, condBr.getTrueDest());
            bool postFalse =
                postDom.postDominates(storeBlock, condBr.getFalseDest());
            if (postTrue == postFalse)
              continue;
            literals.push_back({condBr.getCondition(), postFalse});
          }
          literalSets.push_back(literals);
          values.push_back(unwrapStoredValue(store.getValue()));
        }

        DenseMap<Value, unsigned> coverage;
        for (const auto &literals : literalSets)
          for (const auto &literal : literals)
            coverage[literal.cond] |= literal.inverted ? 2u : 1u;
        for (const auto &entry : coverage)
          if ((entry.second & 3u) != 3u)
            return false;

        auto hasConflict = [&](const SmallVectorImpl<Literal> &lhs,
                               const SmallVectorImpl<Literal> &rhs) {
          for (const auto &lhsLit : lhs)
            for (const auto &rhsLit : rhs)
              if (lhsLit.cond == rhsLit.cond &&
                  lhsLit.inverted != rhsLit.inverted)
                return true;
          return false;
        };

        for (size_t i = 0; i < literalSets.size(); ++i) {
          for (size_t j = i + 1; j < literalSets.size(); ++j) {
            if (!hasConflict(literalSets[i], literalSets[j]))
              return false;
          }
        }

        for (const auto &literals : literalSets) {
          BoolCondition cond(true);
          for (const auto &literal : literals) {
            BoolCondition guard(literal.cond);
            if (literal.inverted)
              guard = guard.inverted(builder);
            cond = cond.andWith(guard, builder);
          }
          conditions.push_back(cond);
        }
      } else {
        for (const auto &entry : ifBranchCoverage)
          if ((entry.second & 3u) != 3u)
            return false;

        auto areExclusive = [&](const SmallVectorImpl<GuardTerm> &lhs,
                                const SmallVectorImpl<GuardTerm> &rhs) {
          for (const auto &lhsTerm : lhs) {
            for (const auto &rhsTerm : rhs) {
              if (lhsTerm.cond == rhsTerm.cond &&
                  lhsTerm.inverted != rhsTerm.inverted)
                return true;
            }
          }
          return false;
        };

        for (size_t i = 0; i < guardSets.size(); ++i) {
          BoolCondition cond(true);
          for (const auto &term : guardSets[i]) {
            BoolCondition guard(term.cond);
            if (term.inverted)
              guard = guard.inverted(builder);
            cond = cond.andWith(guard, builder);
          }
          conditions.push_back(cond);
        }

        for (size_t i = 0; i < guardSets.size(); ++i) {
          for (size_t j = i + 1; j < guardSets.size(); ++j) {
            if (!areExclusive(guardSets[i], guardSets[j]))
              return false;
          }
        }
      }

      if (!hasScfIfConditions) {
        BoolCondition covered(false);
        for (const auto &cond : conditions)
          covered = covered.orWith(cond, builder);
        if (!covered.isTrue())
          return false;
      }

      for (size_t i = 0; i < conditions.size(); ++i) {
        Value condValue = conditions[i].materialize(builder, readOp->getLoc());
        if (!condValue || !dom.dominates(condValue, readOp))
          return false;
        if (!values[i] || !dom.dominates(values[i], readOp))
          return false;
      }

      builder.setInsertionPoint(readOp);
      Value mux = values.back();
      for (size_t i = conditions.size(); i-- > 1;) {
        Value condValue =
            conditions[i - 1].materialize(builder, readOp->getLoc());
        mux = builder.createOrFold<comb::MuxOp>(readOp->getLoc(), condValue,
                                                values[i - 1], mux);
      }
      for (auto read : field.reads)
        read.getResult().replaceAllUsesWith(mux);
      return true;
    };

    if (!field.stores.empty() && !field.reads.empty()) {
      if (resolveStoresByDominance())
        continue;
      if (resolveStoresByComplementaryConditions())
        continue;
      if (resolveStoresByExclusiveConditions())
        continue;
      needsAbstraction = true;
    }
    if (field.hasMultipleStoreValues)
      needsAbstraction = true;

    if (needsAbstraction) {
      if (strictMode)
        return sigOp.emitError(
            "LLHD interface signal requires abstraction; rerun without "
            "--strict-llhd");
      auto module = sigOp->getParentOfType<hw::HWModuleOp>();
      if (!module)
        return sigOp.emitError("expected LLHD signal in hw.module for LEC");
      auto baseName = sigOp.getNameAttr()
                          ? sigOp.getNameAttr().getValue()
                          : StringRef("llhd_if");
      std::string name = baseName.str();
      name += "_field";
      name += std::to_string(entry.first);
      Value newInput =
          state.addInput(module, name, field.reads.front().getType());
      for (auto read : field.reads) {
        read.getResult().replaceAllUsesWith(newInput);
      }
      continue;
    }

    for (auto read : field.reads) {
      if (!dom.dominates(field.storedValue, read.getOperation())) {
        if (strictMode)
          return sigOp.emitError(
              "LLHD interface signal read before dominating store; rerun "
              "without --strict-llhd");
        auto module = sigOp->getParentOfType<hw::HWModuleOp>();
        if (!module)
          return sigOp.emitError("expected LLHD signal in hw.module for LEC");
        auto baseName = sigOp.getNameAttr()
                            ? sigOp.getNameAttr().getValue()
                            : StringRef("llhd_if");
        std::string name = baseName.str();
        name += "_field";
        name += std::to_string(entry.first);
        Value newInput =
            state.addInput(module, name, field.reads.front().getType());
        for (auto readToReplace : field.reads)
          readToReplace.getResult().replaceAllUsesWith(newInput);
        break;
      }
      read.getResult().replaceAllUsesWith(field.storedValue);
    }
  }

  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation *&op : eraseOrder) {
      if (!op)
        continue;
      if (!op->use_empty())
        continue;
      op->erase();
      op = nullptr;
      erased = true;
    }
  }

  if (sigOp->use_empty()) {
    Value init = sigOp.getInit();
    sigOp.erase();
    if (auto *def = init.getDefiningOp())
      if (def->use_empty())
        def->erase();
  }

  return success();
}

void StripLLHDInterfaceSignalsPass::runOnOperation() {
  auto module = getOperation();
  DominanceInfo dom(module);

  DenseMap<Operation *, ModuleState> moduleStates;
  module.walk([&](hw::HWModuleOp hwModule) {
    moduleStates.try_emplace(hwModule.getOperation(), hwModule);
  });

  SmallVector<llhd::CombinationalOp> combinationalOps;
  module.walk([&](llhd::CombinationalOp combOp) {
    combinationalOps.push_back(combOp);
  });
  for (auto combOp : combinationalOps) {
    auto parentModule = combOp->getParentOfType<hw::HWModuleOp>();
    if (!parentModule)
      return signalPassFailure();
    auto stateIt = moduleStates.find(parentModule.getOperation());
    if (stateIt == moduleStates.end())
      return signalPassFailure();
    if (failed(lowerCombinationalOp(combOp, stateIt->second, this->strict)))
      return signalPassFailure();
  }

  SmallVector<llhd::SignalOp> signals;
  module.walk([&](llhd::SignalOp sigOp) { signals.push_back(sigOp); });

  for (auto sigOp : signals) {
    auto parentModule = sigOp->getParentOfType<hw::HWModuleOp>();
    if (!parentModule)
      return signalPassFailure();
    auto stateIt = moduleStates.find(parentModule.getOperation());
    if (stateIt == moduleStates.end())
      return signalPassFailure();
    if (failed(stripInterfaceSignal(sigOp, dom, stateIt->second, this->strict)))
      return signalPassFailure();
  }

  SmallVector<llhd::ConstantTimeOp> times;
  module.walk([&](llhd::ConstantTimeOp op) { times.push_back(op); });
  for (auto timeOp : times)
    if (timeOp->use_empty())
      timeOp.erase();

  bool hasLLHD = false;
  module.walk([&](Operation *op) {
    if (isa<llhd::LLHDDialect>(op->getDialect()))
      hasLLHD = true;
  });
  if (hasLLHD) {
    module.emitError("LLHD operations are not supported by circt-lec");
    return signalPassFailure();
  }
}
