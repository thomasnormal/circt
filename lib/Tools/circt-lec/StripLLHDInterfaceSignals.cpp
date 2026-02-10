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
#include "circt/Support/TwoStateUtils.h"
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
#include "llvm/ADT/APInt.h"
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

static constexpr StringLiteral kLECLocalSignalAttr = "lec.local";

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

  if (sortedBlocks.size() != region.getBlocks().size()) {
    llvm::SmallPtrSet<Block *, 8> reachable(sortedBlocks.begin(),
                                            sortedBlocks.end());
    for (auto &block : llvm::make_early_inc_range(region)) {
      if (!reachable.contains(&block))
        block.erase();
    }
  }

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

static Value buildDefaultValue(Type type, OpBuilder &builder, Location loc,
                               bool zeroUnknown) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return hw::ConstantOp::create(
        builder, loc,
        builder.getIntegerAttr(intType, APInt(intType.getWidth(), 0)));
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    SmallVector<Value, 4> fields;
    fields.reserve(structType.getElements().size());
    for (const auto &field : structType.getElements()) {
      if (auto nestedStruct = dyn_cast<hw::StructType>(field.type)) {
        Value nested =
            buildDefaultValue(nestedStruct, builder, loc, zeroUnknown);
        if (!nested)
          return {};
        fields.push_back(nested);
        continue;
      }
      auto fieldInt = dyn_cast<IntegerType>(field.type);
      if (!fieldInt)
        return {};
      bool isUnknownField =
          field.name && field.name.getValue() == "unknown";
      APInt value = (isUnknownField && !zeroUnknown)
                        ? APInt::getAllOnes(fieldInt.getWidth())
                        : APInt(fieldInt.getWidth(), 0);
      fields.push_back(hw::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(fieldInt, value)));
    }
    return hw::StructCreateOp::create(builder, loc, structType, fields);
  }
  return {};
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

static Value stripPointerCasts(Value ptr,
                               SmallVectorImpl<Operation *> *casts = nullptr) {
  Value current = ptr;
  while (true) {
    if (auto bitcast = current.getDefiningOp<LLVM::BitcastOp>()) {
      if (casts)
        casts->push_back(bitcast);
      current = bitcast.getArg();
      continue;
    }
    if (auto addr = current.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      if (casts)
        casts->push_back(addr);
      current = addr.getArg();
      continue;
    }
    return current;
  }
}

static bool collectAllocaBases(Value current,
                               llvm::SmallPtrSetImpl<Value> &visited,
                               llvm::SmallPtrSetImpl<Operation *> &bases) {
  current = stripPointerCasts(current);
  if (!visited.insert(current).second)
    return true;
  if (auto alloca = current.getDefiningOp<LLVM::AllocaOp>()) {
    bases.insert(alloca.getOperation());
    return true;
  }
  if (auto select = current.getDefiningOp<LLVM::SelectOp>()) {
    return collectAllocaBases(select.getTrueValue(), visited, bases) &&
           collectAllocaBases(select.getFalseValue(), visited, bases);
  }
  auto arg = dyn_cast<BlockArgument>(current);
  if (!arg)
    return false;
  Block *block = arg.getOwner();
  for (Block *pred : block->getPredecessors()) {
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return false;
    bool found = false;
    for (unsigned succIndex = 0, e = branch->getNumSuccessors();
         succIndex < e; ++succIndex) {
      if (branch->getSuccessor(succIndex) != block)
        continue;
      auto succOperands = branch.getSuccessorOperands(succIndex);
      if (arg.getArgNumber() >= succOperands.size())
        return false;
      if (!collectAllocaBases(succOperands[arg.getArgNumber()], visited,
                              bases))
        return false;
      found = true;
    }
    if (!found)
      return false;
  }
  return true;
}

static LLVM::AllocaOp resolveAllocaBase(
    Value ptr, llvm::SmallPtrSetImpl<Value> &visited) {
  Value current = stripPointerCasts(ptr);
  if (auto alloca = current.getDefiningOp<LLVM::AllocaOp>())
    return alloca;
  if (!visited.insert(current).second)
    return nullptr;
  if (auto select = current.getDefiningOp<LLVM::SelectOp>()) {
    auto trueBase = resolveAllocaBase(select.getTrueValue(), visited);
    if (!trueBase)
      return nullptr;
    auto falseBase = resolveAllocaBase(select.getFalseValue(), visited);
    if (!falseBase || falseBase != trueBase)
      return nullptr;
    return trueBase;
  }
  auto arg = dyn_cast<BlockArgument>(current);
  if (!arg)
    return nullptr;
  Block *block = arg.getOwner();
  SmallVector<Value, 4> incomingPtrs;
  for (Block *pred : block->getPredecessors()) {
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return nullptr;
    bool found = false;
    for (unsigned succIndex = 0, e = branch->getNumSuccessors();
         succIndex < e; ++succIndex) {
      if (branch->getSuccessor(succIndex) != block)
        continue;
      auto succOperands = branch.getSuccessorOperands(succIndex);
      if (arg.getArgNumber() >= succOperands.size())
        return nullptr;
      incomingPtrs.push_back(succOperands[arg.getArgNumber()]);
      found = true;
    }
    if (!found)
      return nullptr;
  }
  if (incomingPtrs.empty())
    return nullptr;
  LLVM::AllocaOp base;
  for (Value incoming : incomingPtrs) {
    auto incomingBase = resolveAllocaBase(incoming, visited);
    if (!incomingBase)
      return nullptr;
    if (!base)
      base = incomingBase;
    else if (incomingBase != base)
      return nullptr;
  }
  return base;
}

static bool rewriteAllocaBackedLLHDRef(UnrealizedConversionCastOp castOp,
                                       bool zeroUnknown) {
  if (castOp.getInputs().size() != 1 || castOp.getResults().size() != 1)
    return false;
  auto refType = dyn_cast<llhd::RefType>(castOp.getResult(0).getType());
  if (!refType)
    return false;
  Value ptr = castOp.getInputs().front();
  if (!isa<LLVM::LLVMPointerType>(ptr.getType()))
    return false;

  llvm::SmallPtrSet<Value, 8> visited;
  llvm::SmallPtrSet<Operation *, 4> baseSet;
  if (!collectAllocaBases(ptr, visited, baseSet) || baseSet.empty())
    return false;
  SmallVector<LLVM::AllocaOp, 4> baseAllocas;
  baseAllocas.reserve(baseSet.size());
  for (Operation *op : baseSet)
    baseAllocas.push_back(cast<LLVM::AllocaOp>(op));
  Type allocaElemType = baseAllocas.front().getElemType();
  for (auto base : baseAllocas)
    if (base.getElemType() != allocaElemType)
      return false;

  auto baseSetContains = [&](Value value) -> bool {
    llvm::SmallPtrSet<Value, 8> localVisited;
    llvm::SmallPtrSet<Operation *, 4> valueBases;
    if (!collectAllocaBases(value, localVisited, valueBases) ||
        valueBases.empty())
      return false;
    for (Operation *op : valueBases)
      if (!baseSet.contains(op))
        return false;
    return true;
  };

  llvm::SmallPtrSet<Value, 16> derivedPtrs;
  SmallVector<Value, 16> worklist;
  auto addDerived = [&](Value value) {
    if (derivedPtrs.insert(value).second)
      worklist.push_back(value);
  };
  for (auto base : baseAllocas)
    addDerived(base.getResult());
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
        addDerived(bitcast.getResult());
        continue;
      }
      if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
        addDerived(addr.getResult());
        continue;
      }
      if (auto select = dyn_cast<LLVM::SelectOp>(user)) {
        if (baseSetContains(select.getResult()))
          addDerived(select.getResult());
        continue;
      }
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        for (unsigned succIndex = 0, e = branch->getNumSuccessors();
             succIndex < e; ++succIndex) {
          auto succOperands = branch.getSuccessorOperands(succIndex);
          Block *succ = branch->getSuccessor(succIndex);
          for (unsigned idx = 0, sz = succOperands.size(); idx < sz; ++idx) {
            if (succOperands[idx] != current)
              continue;
            BlockArgument arg = succ->getArgument(idx);
            if (baseSetContains(arg))
              addDerived(arg);
          }
        }
        continue;
      }
    }
  }

  SmallVector<LLVM::StoreOp, 8> stores;
  SmallVector<LLVM::LoadOp, 8> loads;
  SmallVector<UnrealizedConversionCastOp, 4> refCasts;
  SmallVector<Operation *, 8> ptrCasts;
  for (Value value : derivedPtrs) {
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        if (store.getAddr() != value)
          return false;
        stores.push_back(store);
        continue;
      }
      if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
        if (load.getAddr() != value)
          return false;
        loads.push_back(load);
        continue;
      }
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
        if (cast.getInputs().size() == 1 && cast.getResults().size() == 1 &&
            cast.getInputs().front() == value &&
            cast.getResult(0).getType() == refType) {
          refCasts.push_back(cast);
          continue;
        }
        if (cast->use_empty())
          continue;
        return false;
      }
      if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
        if (bitcast.getArg() != value)
          return false;
        if (!derivedPtrs.contains(bitcast.getResult()))
          return false;
        ptrCasts.push_back(bitcast);
        continue;
      }
      if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
        if (addr.getArg() != value)
          return false;
        if (!derivedPtrs.contains(addr.getResult()))
          return false;
        ptrCasts.push_back(addr);
        continue;
      }
      if (auto select = dyn_cast<LLVM::SelectOp>(user)) {
        if (!derivedPtrs.contains(select.getResult()))
          continue;
        continue;
      }
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        for (unsigned succIndex = 0, e = branch->getNumSuccessors();
             succIndex < e; ++succIndex) {
          auto succOperands = branch.getSuccessorOperands(succIndex);
          Block *succ = branch->getSuccessor(succIndex);
          for (unsigned idx = 0, sz = succOperands.size(); idx < sz; ++idx) {
            if (succOperands[idx] != value)
              continue;
            if (!derivedPtrs.contains(succ->getArgument(idx)))
              continue;
          }
        }
        continue;
      }
      return false;
    }
  }

  if (stores.empty() && loads.empty() && refCasts.empty())
    return false;

  Type nestedType = refType.getNestedType();
  DenseMap<LLVM::StoreOp, Value> storeValues;
  for (LLVM::StoreOp store : stores) {
    Value storedValue = unwrapStoredValue(store.getValue());
    if (storedValue.getType() != nestedType) {
      if (auto hwType = dyn_cast<hw::StructType>(nestedType)) {
        if (isa<LLVM::LLVMStructType>(storedValue.getType())) {
          OpBuilder builder(store);
          storedValue = buildHWStructFromLLVM(storedValue, hwType, builder,
                                              store.getLoc(), {});
        }
      }
    }
    if (!storedValue || storedValue.getType() != nestedType)
      return false;
    storeValues[store] = storedValue;
  }

  for (LLVM::LoadOp load : loads) {
    if (load.getType() == nestedType)
      continue;
    for (Operation *user : llvm::to_vector(load.getResult().getUsers())) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast || cast.getInputs().size() != 1 ||
          cast.getResults().size() != 1)
        return false;
      if (cast.getResult(0).getType() != nestedType)
        return false;
    }
  }

  OpBuilder sigBuilder(baseAllocas.front());
  Value init =
      buildDefaultValue(nestedType, sigBuilder, castOp.getLoc(), zeroUnknown);
  if (!init)
    return false;
  auto sigOp = llhd::SignalOp::create(sigBuilder, castOp.getLoc(), refType,
                                      StringAttr{}, init);
  sigOp->setAttr(kLECLocalSignalAttr, sigBuilder.getUnitAttr());
  Value sig = sigOp.getResult();

  for (auto refCast : refCasts) {
    refCast.getResult(0).replaceAllUsesWith(sig);
    refCast.erase();
  }

  for (LLVM::LoadOp load : loads) {
    OpBuilder loadBuilder(load);
    Value probe = llhd::ProbeOp::create(loadBuilder, load.getLoc(), sig);
    if (probe.getType() == load.getType()) {
      load.getResult().replaceAllUsesWith(probe);
      load.erase();
      continue;
    }
    for (Operation *user : llvm::to_vector(load.getResult().getUsers())) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast)
        return false;
      cast.getResult(0).replaceAllUsesWith(probe);
      cast.erase();
    }
    load.erase();
  }

  for (LLVM::StoreOp store : stores) {
    OpBuilder storeBuilder(store);
    Value zeroTime = llhd::ConstantTimeOp::create(storeBuilder, store.getLoc(),
                                                  0, "ns", 0, 1);
    llhd::DriveOp::create(storeBuilder, store.getLoc(), sig,
                          storeValues.lookup(store), zeroTime, Value{});
    store.erase();
  }

  for (Operation *cast : llvm::reverse(ptrCasts))
    if (cast->use_empty())
      cast->erase();
  for (auto base : baseAllocas)
    if (base.use_empty())
      base.erase();
  return true;
}

static bool rewriteAllocaBackedStruct(LLVM::AllocaOp alloca,
                                      bool zeroUnknown) {
  llvm::SmallPtrSet<Value, 8> visited;
  if (resolveAllocaBase(alloca.getResult(), visited) != alloca)
    return false;

  llvm::SmallPtrSet<Value, 16> derivedPtrs;
  SmallVector<Value, 16> worklist;
  auto addDerived = [&](Value value) {
    if (derivedPtrs.insert(value).second)
      worklist.push_back(value);
  };
  addDerived(alloca.getResult());
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
        addDerived(bitcast.getResult());
        continue;
      }
      if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
        addDerived(addr.getResult());
        continue;
      }
      if (auto select = dyn_cast<LLVM::SelectOp>(user)) {
        llvm::SmallPtrSet<Value, 8> localVisited;
        if (resolveAllocaBase(select.getResult(), localVisited) == alloca)
          addDerived(select.getResult());
        continue;
      }
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        for (unsigned succIndex = 0, e = branch->getNumSuccessors();
             succIndex < e; ++succIndex) {
          auto succOperands = branch.getSuccessorOperands(succIndex);
          Block *succ = branch->getSuccessor(succIndex);
          for (unsigned idx = 0, sz = succOperands.size(); idx < sz; ++idx) {
            if (succOperands[idx] != current)
              continue;
            BlockArgument arg = succ->getArgument(idx);
            llvm::SmallPtrSet<Value, 8> localVisited;
            if (resolveAllocaBase(arg, localVisited) == alloca)
              addDerived(arg);
          }
        }
        continue;
      }
    }
  }

  SmallVector<LLVM::StoreOp, 8> stores;
  SmallVector<LLVM::LoadOp, 8> loads;
  SmallVector<Operation *, 8> ptrCasts;
  for (Value value : derivedPtrs) {
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
        if (store.getAddr() != value)
          return false;
        stores.push_back(store);
        continue;
      }
      if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
        if (load.getAddr() != value)
          return false;
        loads.push_back(load);
        continue;
      }
      if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
        if (bitcast.getArg() != value)
          return false;
        if (!derivedPtrs.contains(bitcast.getResult()))
          return false;
        ptrCasts.push_back(bitcast);
        continue;
      }
      if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
        if (addr.getArg() != value)
          return false;
        if (!derivedPtrs.contains(addr.getResult()))
          return false;
        ptrCasts.push_back(addr);
        continue;
      }
      if (auto select = dyn_cast<LLVM::SelectOp>(user)) {
        if (!derivedPtrs.contains(select.getResult()))
          continue;
        continue;
      }
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        for (unsigned succIndex = 0, e = branch->getNumSuccessors();
             succIndex < e; ++succIndex) {
          auto succOperands = branch.getSuccessorOperands(succIndex);
          Block *succ = branch->getSuccessor(succIndex);
          for (unsigned idx = 0, sz = succOperands.size(); idx < sz; ++idx) {
            if (succOperands[idx] != value)
              continue;
            if (!derivedPtrs.contains(succ->getArgument(idx)))
              continue;
          }
        }
        continue;
      }
      return false;
    }
  }

  if (stores.empty() || loads.empty())
    return false;

  hw::StructType hwType;
  for (LLVM::LoadOp load : loads) {
    if (load.getResult().use_empty())
      return false;
    for (Operation *user : llvm::to_vector(load.getResult().getUsers())) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast || cast.getInputs().size() != 1 ||
          cast.getResults().size() != 1)
        return false;
      auto castType = dyn_cast<hw::StructType>(cast.getResult(0).getType());
      if (!castType)
        return false;
      if (!hwType)
        hwType = castType;
      else if (hwType != castType)
        return false;
    }
  }
  if (!hwType)
    return false;

  DenseMap<LLVM::StoreOp, Value> storeValues;
  for (LLVM::StoreOp store : stores) {
    Value storedValue = unwrapStoredValue(store.getValue());
    if (storedValue.getType() != hwType) {
      if (isa<LLVM::LLVMStructType>(storedValue.getType())) {
        OpBuilder builder(store);
        storedValue =
            buildHWStructFromLLVM(storedValue, hwType, builder,
                                  store.getLoc(), {});
      }
    }
    if (!storedValue || storedValue.getType() != hwType)
      return false;
    storeValues[store] = storedValue;
  }

  llhd::RefType refType = llhd::RefType::get(hwType);
  OpBuilder sigBuilder(alloca);
  Value init =
      buildDefaultValue(hwType, sigBuilder, alloca.getLoc(), zeroUnknown);
  if (!init)
    return false;
  auto sigOp =
      llhd::SignalOp::create(sigBuilder, alloca.getLoc(), refType,
                             StringAttr{}, init);
  sigOp->setAttr(kLECLocalSignalAttr, sigBuilder.getUnitAttr());
  Value sig = sigOp.getResult();

  for (LLVM::LoadOp load : loads) {
    OpBuilder loadBuilder(load);
    Value probe = llhd::ProbeOp::create(loadBuilder, load.getLoc(), sig);
    for (Operation *user : llvm::to_vector(load.getResult().getUsers())) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast)
        return false;
      cast.getResult(0).replaceAllUsesWith(probe);
      cast.erase();
    }
    load.erase();
  }

  for (LLVM::StoreOp store : stores) {
    OpBuilder storeBuilder(store);
    Value zeroTime = llhd::ConstantTimeOp::create(storeBuilder, store.getLoc(),
                                                  0, "ns", 0, 1);
    llhd::DriveOp::create(storeBuilder, store.getLoc(), sig,
                          storeValues.lookup(store), zeroTime, Value{});
    store.erase();
  }

  for (Operation *cast : llvm::reverse(ptrCasts))
    if (cast->use_empty())
      cast->erase();
  if (alloca.use_empty())
    alloca.erase();
  return true;
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

  SmallVector<UnrealizedConversionCastOp> refCasts;
  region.walk([&](UnrealizedConversionCastOp op) {
    if (op.getInputs().size() == 1 && op.getResults().size() == 1 &&
        isa<LLVM::LLVMPointerType>(op.getInputs().front().getType()) &&
        isa<llhd::RefType>(op.getResult(0).getType()))
      refCasts.push_back(op);
  });
  for (auto castOp : refCasts)
    (void)rewriteAllocaBackedLLHDRef(castOp, /*zeroUnknown=*/!strictMode);

  SmallVector<LLVM::AllocaOp> allocas;
  region.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
  for (LLVM::AllocaOp alloca : allocas)
    if (alloca->getParentOp())
      (void)rewriteAllocaBackedStruct(alloca, /*zeroUnknown=*/!strictMode);

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

  // Remove unused block arguments to avoid invalid pointer-typed merges.
  if (!region.empty()) {
    for (Block &block : region) {
      for (int argIndex = static_cast<int>(block.getNumArguments()) - 1;
           argIndex >= 0; --argIndex) {
        BlockArgument arg = block.getArgument(argIndex);
        if (!arg.use_empty())
          continue;
        for (Block *pred : block.getPredecessors()) {
          Operation *terminator = pred->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
            auto operands = llvm::to_vector(br.getDestOperands());
            operands.erase(operands.begin() + argIndex);
            br.getDestOperandsMutable().assign(operands);
            continue;
          }
          if (auto br = dyn_cast<cf::CondBranchOp>(terminator)) {
            if (br.getTrueDest() == &block) {
              auto operands = llvm::to_vector(br.getTrueDestOperands());
              operands.erase(operands.begin() + argIndex);
              br.getTrueDestOperandsMutable().assign(operands);
              continue;
            }
            if (br.getFalseDest() == &block) {
              auto operands = llvm::to_vector(br.getFalseDestOperands());
              operands.erase(operands.begin() + argIndex);
              br.getFalseDestOperandsMutable().assign(operands);
              continue;
            }
            continue;
          }
          return combOp.emitError(
              "unsupported predecessor for unused block argument in LEC");
        }
        block.eraseArgument(argIndex);
      }
    }
  }

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

    // If a pointer-typed block argument selects between multiple allocas, MLIR
    // mem2reg cannot promote the underlying slots because the pointer escapes
    // through the CFG. For common LEC patterns, the different allocas are
    // "pointer-only" (only used to feed that block argument), so the identity
    // of the chosen alloca is unobservable. In those cases, rewrite the block
    // argument to use a single canonical alloca and drop the CFG operand.
    for (Block &block : region) {
      for (int argIndex = static_cast<int>(block.getNumArguments()) - 1;
           argIndex >= 0; --argIndex) {
        BlockArgument arg = block.getArgument(argIndex);
        if (!isa<LLVM::LLVMPointerType>(arg.getType()))
          continue;

        // Collect all incoming operands for this block argument.
        SmallVector<Value, 4> incoming;
        llvm::SmallPtrSet<Value, 8> incomingSet;
        for (Block *pred : block.getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch)
            continue;
          for (unsigned succIndex = 0, e = branch->getNumSuccessors();
               succIndex < e; ++succIndex) {
            if (branch->getSuccessor(succIndex) != &block)
              continue;
            auto succOperands = branch.getSuccessorOperands(succIndex);
            if (static_cast<unsigned>(argIndex) >= succOperands.size())
              continue;
            unsigned operandIndex = succOperands.getOperandIndex(argIndex);
            Value operand = branch->getOperand(operandIndex);
            if (incomingSet.insert(operand).second)
              incoming.push_back(operand);
          }
        }
        if (incoming.size() <= 1)
          continue;

        // We only collapse when all incoming pointers originate from allocas
        // that are otherwise "pointer-only", i.e. they are not used for memory
        // accesses outside of feeding this specific block argument.
        auto feedsOnlyThisArg = [&](Value root) -> bool {
          SmallVector<Value, 8> worklist;
          llvm::SmallPtrSet<Value, 16> visited;
          worklist.push_back(root);
          visited.insert(root);
          while (!worklist.empty()) {
            Value current = worklist.pop_back_val();
            for (OpOperand &use : current.getUses()) {
              Operation *user = use.getOwner();
              if (isa<LLVM::LoadOp, LLVM::StoreOp>(user))
                return false;
              if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
                if (bitcast.getArg() != current)
                  return false;
                Value next = bitcast.getResult();
                if (visited.insert(next).second)
                  worklist.push_back(next);
                continue;
              }
              if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
                if (addr.getArg() != current)
                  return false;
                Value next = addr.getResult();
                if (visited.insert(next).second)
                  worklist.push_back(next);
                continue;
              }
              if (auto select = dyn_cast<LLVM::SelectOp>(user)) {
                // Follow the select result; a merged pointer is still fine as
                // long as it is only used to feed this argument.
                Value next = select.getResult();
                if (visited.insert(next).second)
                  worklist.push_back(next);
                continue;
              }
              if (auto branch = dyn_cast<BranchOpInterface>(user)) {
                bool ok = false;
                for (unsigned succIndex = 0, e = branch->getNumSuccessors();
                     succIndex < e; ++succIndex) {
                  auto succOperands = branch.getSuccessorOperands(succIndex);
                  for (unsigned index = 0, sz = succOperands.size(); index < sz;
                       ++index) {
                    Value operand = succOperands[index];
                    if (!operand || operand != current)
                      continue;
                    if (branch->getSuccessor(succIndex) != &block ||
                        static_cast<int>(index) != argIndex)
                      return false;
                    ok = true;
                  }
                }
                if (!ok)
                  return false;
                continue;
              }
              return false;
            }
          }
          return true;
        };

        SmallVector<Value, 4> bases;
        bases.reserve(incoming.size());
        bool allPointerOnly = true;
        for (Value in : incoming) {
          if (!isa<LLVM::LLVMPointerType>(in.getType())) {
            allPointerOnly = false;
            break;
          }
          Value stripped = stripPointerCasts(in);
          auto alloca = stripped.getDefiningOp<LLVM::AllocaOp>();
          if (!alloca) {
            allPointerOnly = false;
            break;
          }
          if (!feedsOnlyThisArg(alloca.getResult())) {
            allPointerOnly = false;
            break;
          }
          bases.push_back(alloca.getResult());
        }
        if (!allPointerOnly || bases.empty())
          continue;

        Value canonical = bases.front();
        if (canonical.getType() != arg.getType())
          continue;

        // Ensure all predecessor edges provide an eraseable operand for this
        // argument. If not, skip collapsing to avoid producing invalid CFG
        // operand lists.
        bool canEraseAllEdges = true;
        SmallPtrSet<Block *, 8> seenPreds;
        for (Block *pred : block.getPredecessors()) {
          if (!seenPreds.insert(pred).second)
            continue;
          Operation *terminator = pred->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
            if (br.getDest() == &block &&
                static_cast<unsigned>(argIndex) >= br.getDestOperands().size())
              canEraseAllEdges = false;
            continue;
          }
          if (auto br = dyn_cast<cf::CondBranchOp>(terminator)) {
            if (br.getTrueDest() == &block &&
                static_cast<unsigned>(argIndex) >=
                    br.getTrueDestOperands().size())
              canEraseAllEdges = false;
            if (br.getFalseDest() == &block &&
                static_cast<unsigned>(argIndex) >=
                    br.getFalseDestOperands().size())
              canEraseAllEdges = false;
            continue;
          }
          canEraseAllEdges = false;
        }
        if (!canEraseAllEdges)
          continue;

        arg.replaceAllUsesWith(canonical);
        seenPreds.clear();
        for (Block *pred : block.getPredecessors()) {
          if (!seenPreds.insert(pred).second)
            continue;
          Operation *terminator = pred->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(terminator)) {
            auto operands = llvm::to_vector(br.getDestOperands());
            operands.erase(operands.begin() + argIndex);
            br.getDestOperandsMutable().assign(operands);
            continue;
          }
          if (auto br = dyn_cast<cf::CondBranchOp>(terminator)) {
            if (br.getTrueDest() == &block) {
              auto operands = llvm::to_vector(br.getTrueDestOperands());
              operands.erase(operands.begin() + argIndex);
              br.getTrueDestOperandsMutable().assign(operands);
            }
            if (br.getFalseDest() == &block) {
              auto operands = llvm::to_vector(br.getFalseDestOperands());
              operands.erase(operands.begin() + argIndex);
              br.getFalseDestOperandsMutable().assign(operands);
            }
            continue;
          }
          return combOp.emitError(
              "unsupported predecessor for pointer block argument collapse in "
              "LEC");
        }
        block.eraseArgument(argIndex);
      }
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
            unsigned operandIndex = succOperands.getOperandIndex(argIndex);
            Value operandValue = branch->getOperand(operandIndex);
            if (operandValue.getType() == loadType)
              continue;
            if (!isa<LLVM::LLVMPointerType>(operandValue.getType()))
              continue;
            Value loaded = LLVM::LoadOp::create(builder, branch.getLoc(),
                                                loadType, operandValue);
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
      // Use dominance/data layout from the surrounding module op. The
      // combinational body may contain control flow and nested regions; using
      // the parent op here avoids known cases where dominance for the isolated
      // op fails to promote otherwise-promotable slots.
      DataLayout dataLayout = DataLayout::closest(module);
      DominanceInfo dominance(module);
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

  // After CFRemover merges blocks, ops may reference values defined later in
  // the block (e.g. a hw.constant created for drive enables placed after the
  // drive that uses it). Sort the block topologically so that cloning visits
  // definitions before their uses.
  {
    llvm::SmallPtrSet<Operation *, 16> scheduled;
    SmallVector<Operation *> sorted;
    std::function<void(Operation *)> schedule = [&](Operation *op) {
      if (!scheduled.insert(op).second)
        return;
      for (Value operand : op->getOperands()) {
        if (auto *def = operand.getDefiningOp()) {
          if (def->getBlock() == &body)
            schedule(def);
        }
      }
      sorted.push_back(op);
    };
    for (auto &op : body)
      schedule(&op);
    for (Operation *op : sorted)
      op->moveBefore(terminator);
  }

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
  bool isLocalSignal = sigOp->hasAttr(kLECLocalSignalAttr);
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
    if (!valueType)
      return false;
    if (isFourStateStructType(valueType))
      return true;
    return getTwoStateValueWidth(valueType).has_value();
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
  auto getStructFieldType =
      [&](hw::StructType structType, StringAttr field) -> Type {
    for (const auto &element : structType.getElements()) {
      if (element.name == field)
        return element.type;
    }
    return Type();
  };
  auto canInlinePath = [&](Type baseType, const RefPath &path) -> bool {
    Type currentType = baseType;
    for (const auto &step : path) {
      if (step.kind == RefStep::StructField) {
        auto structType = dyn_cast<hw::StructType>(currentType);
        if (!structType)
          return false;
        currentType = getStructFieldType(structType, step.field);
        if (!currentType)
          return false;
        continue;
      }
      if (step.kind == RefStep::Extract) {
        auto baseInt = dyn_cast<IntegerType>(currentType);
        auto elemInt = dyn_cast<IntegerType>(step.elemType);
        if (!baseInt || !elemInt)
          return false;
        auto constant = step.index.getDefiningOp<hw::ConstantOp>();
        if (!constant)
          return false;
        uint64_t low = constant.getValue().getZExtValue();
        if (low + elemInt.getWidth() > baseInt.getWidth())
          return false;
        currentType = elemInt;
        continue;
      }
    }
    return true;
  };
  auto materializePath = [&](OpBuilder &pathBuilder, Value baseValue,
                             const RefPath &path, Location loc) -> Value {
    Value value = baseValue;
    for (const auto &step : path) {
      if (step.kind == RefStep::StructField) {
        value =
            hw::StructExtractOp::create(pathBuilder, loc, value, step.field);
        continue;
      }
      if (step.kind == RefStep::Extract) {
        auto elemType = dyn_cast<IntegerType>(step.elemType);
        auto baseType = dyn_cast<IntegerType>(value.getType());
        if (!elemType || !baseType)
          return Value();
        if (auto constant = step.index.getDefiningOp<hw::ConstantOp>()) {
          uint64_t low = constant.getValue().getZExtValue();
          value = comb::ExtractOp::create(pathBuilder, loc, value, low,
                                          elemType.getWidth());
          continue;
        }
        Value shift = adjustIntegerWidth(pathBuilder, step.index,
                                         baseType.getWidth(), loc);
        Value shifted =
            comb::ShrUOp::create(pathBuilder, loc, value, shift);
        value = comb::ExtractOp::create(pathBuilder, loc, shifted, 0,
                                        elemType.getWidth());
        continue;
      }
    }
    return value;
  };

  // If the signal has a name matching a module input port and it is only ever
  // read (no drives), treat it as an alias for that port. This makes the
  // stripping robust against earlier LLHD transforms that drop the explicit
  // `llhd.drv %sig, %port after 0` and leave the signal stuck at its init.
  if (drives.empty() && !isLocalSignal && forwardedArgs.empty()) {
    if (auto nameAttr = sigOp.getNameAttr()) {
      auto parentModule = sigOp->getParentOfType<hw::HWModuleOp>();
      if (parentModule) {
        Value portValue;
        auto inputNames = parentModule.getModuleType().getInputNames();
        for (auto [index, inputNameAttr] : llvm::enumerate(inputNames)) {
          auto inputName = cast<StringAttr>(inputNameAttr).getValue();
          if (inputName != nameAttr.getValue())
            continue;
          portValue = parentModule.getArgumentForInput(index);
          break;
        }
        if (portValue &&
            portValue.getType() == sigOp.getType().getNestedType()) {
          for (auto probe : probes) {
            OpBuilder probeBuilder(probe);
            Value replacement =
                materializePath(probeBuilder, portValue,
                                probePaths.lookup(probe), probe.getLoc());
            if (!replacement)
              return probe.emitError("unsupported LLHD probe path in LEC");
            if (replacement.getType() != probe.getResult().getType())
              return probe.emitError("signal probe type mismatch in LEC");
            probe.getResult().replaceAllUsesWith(replacement);
            probe.erase();
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
    }
  }
  auto updatePath = [&](auto &&self, OpBuilder &pathBuilder, Value baseValue,
                        ArrayRef<RefStep> path, Value updateValue,
                        Location loc) -> Value {
    if (path.empty())
      return updateValue;
    const RefStep &step = path.front();
    if (step.kind == RefStep::StructField) {
      auto structType = dyn_cast<hw::StructType>(baseValue.getType());
      if (!structType)
        return Value();
      bool isFourStateStruct = isFourStateStructType(structType);
      StringAttr unknownFieldName;
      if (isFourStateStruct) {
        for (const auto &element : structType.getElements()) {
          if (element.name && element.name.getValue() == "unknown") {
            unknownFieldName = element.name;
            break;
          }
        }
      }
      SmallVector<Value, 4> fields;
      fields.reserve(structType.getElements().size());
      for (const auto &element : structType.getElements()) {
        Value fieldValue =
            hw::StructExtractOp::create(pathBuilder, loc, baseValue,
                                        element.name);
        if (element.name == step.field) {
          fieldValue = self(self, pathBuilder, fieldValue, path.drop_front(),
                            updateValue, loc);
          if (!fieldValue)
            return Value();
          if (isFourStateStruct && element.name.getValue() == "value" &&
              unknownFieldName) {
            auto updateInt = dyn_cast<IntegerType>(updateValue.getType());
            if (updateInt) {
              Value unknownValue = hw::StructExtractOp::create(
                  pathBuilder, loc, baseValue, unknownFieldName);
              Value zeroUpdate =
                  createZeroValue(pathBuilder, loc, updateValue.getType());
              Value updatedUnknown =
                  self(self, pathBuilder, unknownValue, path.drop_front(),
                       zeroUpdate, loc);
              if (!updatedUnknown)
                return Value();
              Value unknownFieldValue = updatedUnknown;
              SmallVector<Value, 4> updatedFields;
              updatedFields.reserve(structType.getElements().size());
              for (const auto &nested : structType.getElements()) {
                Value nestedValue = hw::StructExtractOp::create(
                    pathBuilder, loc, baseValue, nested.name);
                if (nested.name == step.field)
                  nestedValue = fieldValue;
                else if (nested.name == unknownFieldName)
                  nestedValue = unknownFieldValue;
                updatedFields.push_back(nestedValue);
              }
              return hw::StructCreateOp::create(pathBuilder, loc, structType,
                                                updatedFields);
            }
          }
        }
        fields.push_back(fieldValue);
      }
      return hw::StructCreateOp::create(pathBuilder, loc, structType, fields);
    }
    if (step.kind == RefStep::Extract) {
      auto baseInt = dyn_cast<IntegerType>(baseValue.getType());
      auto elemInt = dyn_cast<IntegerType>(step.elemType);
      if (!baseInt || !elemInt)
        return Value();
      auto constant = step.index.getDefiningOp<hw::ConstantOp>();
      if (!constant)
        return Value();
      uint64_t low = constant.getValue().getZExtValue();
      unsigned elemWidth = elemInt.getWidth();
      unsigned baseWidth = baseInt.getWidth();
      if (low + elemWidth > baseWidth)
        return Value();
      Value updated =
          adjustIntegerWidth(pathBuilder, updateValue, elemWidth, loc);
      SmallVector<Value, 3> pieces;
      if (low + elemWidth < baseWidth) {
        Value upper = comb::ExtractOp::create(
            pathBuilder, loc, baseValue, low + elemWidth,
            baseWidth - low - elemWidth);
        pieces.push_back(upper);
      }
      pieces.push_back(updated);
      if (low > 0) {
        Value lower =
            comb::ExtractOp::create(pathBuilder, loc, baseValue, 0, low);
        pieces.push_back(lower);
      }
      if (pieces.size() == 1)
        return pieces.front();
      return comb::ConcatOp::create(pathBuilder, loc, pieces);
    }
    return Value();
  };

  bool canInline = forwardedArgs.empty();
  Block *singleBlock = nullptr;
  for (auto probe : probes) {
    if (!probePaths.lookup(probe).empty()) {
      if (!canInlinePath(sigOp.getType().getNestedType(),
                         probePaths.lookup(probe)))
        canInline = false;
    }
    Block *block = probe->getBlock();
    if (!singleBlock)
      singleBlock = block;
    else if (singleBlock != block)
      canInline = false;
  }
  for (auto driveOp : drives) {
    if (!drivePaths.lookup(driveOp).empty()) {
      if (!canInlinePath(sigOp.getType().getNestedType(),
                         drivePaths.lookup(driveOp)))
        canInline = false;
    }
    if (!isLocalSignal && driveOp.getEnable())
      canInline = false;
    if (!driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>())
      canInline = false;
    Block *block = driveOp->getBlock();
    if (!singleBlock)
      singleBlock = block;
    else if (singleBlock != block)
      canInline = false;
  }


  auto dominatesUse = [&](Value value, Operation *use) -> bool {
    if (!value || !use)
      return false;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *argBlock = arg.getOwner();
      Block *useBlock = use->getBlock();
      if (argBlock == useBlock)
        return true;
      return dom.dominates(argBlock, useBlock);
    }
    if (auto *def = value.getDefiningOp()) {
      if (def->getBlock() == use->getBlock())
        return def->isBeforeInBlock(use);
      return dom.dominates(def, use);
    }
    return false;
  };

  auto isZeroTimeLike = [&](llhd::DriveOp driveOp) -> bool {
    auto timeOp = driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
    if (!timeOp)
      return false;
    auto t = timeOp.getValue();
    // Accept 0-time drives, including epsilon scheduling used to model
    // combinational propagation in LLHD.
    return t.getTime() == 0 && t.getDelta() == 0;
  };

  if (drives.size() == 1) {
    llhd::DriveOp driveOp = drives.front();
    if (drivePaths.lookup(driveOp).empty() && !driveOp.getEnable() &&
        driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>() &&
        isa<BlockArgument>(driveOp.getValue())) {
      Value drivenValue = driveOp.getValue();
      for (auto probe : probes) {
        OpBuilder probeBuilder(probe);
        Value replacement = materializePath(probeBuilder, drivenValue,
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

  // Fast-path: a single unconditional 0-time (or epsilon-scheduled) drive.
  //
  // LLHD lowering for combinational logic may schedule drives with an epsilon
  // delay (`llhd.constant_time <0ns, 0d, 1e>`). For LEC, interpret such signals
  // as combinational wires (their driven value) when that driven value
  // dominates all probes.
  if (forwardedArgs.empty() && drives.size() == 1) {
    llhd::DriveOp driveOp = drives.front();
    if (drivePaths.lookup(driveOp).empty() && !driveOp.getEnable() &&
        isZeroTimeLike(driveOp)) {
      Value drivenValue = unwrapStoredValue(driveOp.getValue());
      if (!drivenValue)
        return driveOp.emitError("unsupported LLHD drive value in LEC");

      bool dominatesAllProbes = llvm::all_of(probes, [&](llhd::ProbeOp probe) {
        return dominatesUse(drivenValue, probe.getOperation());
      });
      if (dominatesAllProbes) {
        for (auto probe : probes) {
          OpBuilder probeBuilder(probe);
          Value replacement = materializePath(probeBuilder, drivenValue,
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
  }

  if (canInline && singleBlock) {
    // In strict mode, reject multiple unconditional drives as requiring
    // abstraction since simultaneous drives have undefined priority.
    if (strictMode && !isLocalSignal && drives.size() > 1 &&
        hasEnabledDrive(drives))
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");

    if (!isLocalSignal && drives.size() > 1 && !allDrivesSameValue(drives)) {
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
        Type valueType = driveValues.front().getType();
        Value resolved;
        if (isFourStateStructType(valueType)) {
          resolved = anyStrength
                         ? resolveFourStateValuesWithStrength(
                               resolveBuilder, firstProbe->getLoc(),
                               driveValues, driveEnables, driveStrength0,
                               driveStrength1,
                               static_cast<unsigned>(
                                   llhd::DriveStrength::HighZ))
                         : resolveFourStateValuesWithEnable(
                               resolveBuilder, firstProbe->getLoc(),
                               driveValues, driveEnables);
        } else {
          auto module = sigOp->getParentOfType<hw::HWModuleOp>();
          if (!module)
            return sigOp.emitError("expected LLHD signal in hw.module");
          Value unknownValue;
          auto getUnknownValue = [&]() -> Value {
            if (unknownValue)
              return unknownValue;
            std::string baseName = "llhd_sig_unknown";
            if (auto nameAttr = sigOp.getNameAttr())
              baseName = (nameAttr.getValue() + "_unknown").str();
            unknownValue = state.addInput(module, baseName, valueType);
            return unknownValue;
          };
          resolved =
              anyStrength
                  ? resolveTwoStateValuesWithStrength(
                        resolveBuilder, firstProbe->getLoc(), driveValues,
                        driveEnables, driveStrength0, driveStrength1,
                        getUnknownValue(),
                        static_cast<unsigned>(llhd::DriveStrength::HighZ))
                  : resolveTwoStateValuesWithEnable(
                        resolveBuilder, firstProbe->getLoc(), driveValues,
                        driveEnables, getUnknownValue());
        }
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

      if (strictMode && !isLocalSignal)
        return sigOp.emitError(
            "LLHD signal requires abstraction; rerun without --strict-llhd");
    }

    Value current = sigOp.getInit();
    for (auto it = singleBlock->begin(); it != singleBlock->end();) {
      Operation *op = &*it++;
      if (auto driveOp = dyn_cast<llhd::DriveOp>(op)) {
        auto pathIt = drivePaths.find(driveOp);
        if (pathIt == drivePaths.end())
          continue;
        OpBuilder driveBuilder(driveOp);
        Value driveValue = unwrapStoredValue(driveOp.getValue());
        if (!driveValue)
          return driveOp.emitError("unsupported LLHD drive value in LEC");
        Value next = driveValue;
        const RefPath &path = pathIt->second;
        if (!path.empty()) {
          next = updatePath(updatePath, driveBuilder, current, path, driveValue,
                            driveOp.getLoc());
          if (!next)
            return driveOp.emitError(
                "unsupported LLHD drive path update in LEC");
        }
        if (Value enable = driveOp.getEnable()) {
          next = comb::MuxOp::create(driveBuilder, driveOp.getLoc(), enable,
                                     next, current);
        }
        current = next;
        driveOp.erase();
        continue;
      }
      if (auto probe = dyn_cast<llhd::ProbeOp>(op)) {
        auto pathIt = probePaths.find(probe);
        if (pathIt == probePaths.end())
          continue;
        OpBuilder probeBuilder(probe);
        Value replacement =
            materializePath(probeBuilder, current, pathIt->second,
                            probe.getLoc());
        if (!replacement)
          return probe.emitError("unsupported LLHD probe path in LEC");
        probe.getResult().replaceAllUsesWith(replacement);
        probe.erase();
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
        strictMode && !isLocalSignal && canResolveEnabledDrives(drives) &&
        hasEnabledDrive(drives);
    // In strict mode, reject multiple conditional drives unless they are
    // complementary enables on exactly two drives.
    bool hasMultipleUnconditionalDrives =
        drives.size() > 1 &&
        llvm::any_of(drives, [](llhd::DriveOp d) { return !d.getEnable(); });
    if (strictMode && !isLocalSignal && drives.size() > 1 &&
        hasEnabledDrive(drives) &&
        !resolveEnabledDrives && !hasComplementaryEnables(drives) &&
        !hasExclusiveEnables(drives, builder))
      return sigOp.emitError(
          "LLHD signal requires abstraction; rerun without --strict-llhd");
    if (strictMode && !isLocalSignal && hasMultipleUnconditionalDrives &&
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
                Type valueType = activeValues.front().getType();
                if (isFourStateStructType(valueType)) {
                  replacement =
                      activeAnyStrength
                          ? resolveFourStateValuesWithStrength(
                                builder, probe.getLoc(), activeValues,
                                activeEnables, activeStrength0,
                                activeStrength1,
                                static_cast<unsigned>(
                                    llhd::DriveStrength::HighZ))
                          : resolveFourStateValuesWithEnable(
                                builder, probe.getLoc(), activeValues,
                                activeEnables);
                } else {
                  auto module = sigOp->getParentOfType<hw::HWModuleOp>();
                  if (!module)
                    return probe.emitError(
                        "expected LLHD signal in hw.module");
                  std::string baseName = "llhd_sig_unknown";
                  if (auto nameAttr = sigOp.getNameAttr())
                    baseName = (nameAttr.getValue() + "_unknown").str();
                  Value unknownValue =
                      state.addInput(module, baseName, valueType);
                  replacement =
                      activeAnyStrength
                          ? resolveTwoStateValuesWithStrength(
                                builder, probe.getLoc(), activeValues,
                                activeEnables, activeStrength0,
                                activeStrength1, unknownValue,
                                static_cast<unsigned>(
                                    llhd::DriveStrength::HighZ))
                          : resolveTwoStateValuesWithEnable(
                                builder, probe.getLoc(), activeValues,
                                activeEnables, unknownValue);
                }
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
    if (drive.getEnable() && !isLocalSignal)
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
          materializePath(builder, newInput, probePaths.lookup(probe),
                          probe.getLoc());
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
        materializePath(builder, drivenValue, probePaths.lookup(probe),
                        probe.getLoc());
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
        if (auto load = dyn_cast<LLVM::LoadOp>(gepUser)) {
          // Convert llvm.load into the equivalent cast+probe pattern that
          // the rest of the code already handles.
          auto loadType = load.getResult().getType();
          OpBuilder builder(load);
          auto refType = llhd::RefType::get(builder.getContext(), loadType);
          auto cast = builder.create<UnrealizedConversionCastOp>(
              load.getLoc(), TypeRange{refType}, ValueRange{gep.getResult()});
          auto probeOp = builder.create<llhd::ProbeOp>(load.getLoc(), loadType,
                                                        cast.getResult(0));
          load.getResult().replaceAllUsesWith(probeOp.getResult());
          field.reads.push_back(probeOp);
          markErase(probeOp);
          markErase(cast);
          markErase(load);
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

    auto resolveStoresByEnableResolution = [&]() -> bool {
      if (field.stores.empty() || field.reads.empty())
        return false;
      auto valueType = field.reads.front().getType();
      bool isFourState = isFourStateStructType(valueType);
      if (!isFourState && !getTwoStateValueWidth(valueType))
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
      PostDominanceInfo postDom(module);
      SmallDenseMap<std::pair<Block *, Block *>, BoolCondition> decisionCache;

      auto buildStoreCondition =
          [&](LLVM::StoreOp store) -> std::optional<BoolCondition> {
        BoolCondition condition(true);
        Operation *cursor = store.getOperation();
        while (Operation *parent = cursor->getParentOp()) {
          if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
            Region *parentRegion = cursor->getParentRegion();
            bool inThen = ifOp.getThenRegion().isAncestor(parentRegion);
            bool inElse = ifOp.getElseRegion().isAncestor(parentRegion);
            if (!inThen && !inElse)
              return std::nullopt;
            BoolCondition guard(ifOp.getCondition());
            if (inElse)
              guard = guard.inverted(builder);
            condition = condition.andWith(guard, builder);
          }
          cursor = parent;
        }

        Block *entryBlock = &store->getParentRegion()->front();
        BoolCondition branchCondition = getBranchDecisionsFromDominatorToTarget(
            builder, entryBlock, store->getBlock(), decisionCache);
        condition = condition.andWith(branchCondition, builder);
        return condition;
      };

      SmallVector<Value, 4> values;
      SmallVector<Value, 4> enables;
      values.reserve(field.stores.size());
      enables.reserve(field.stores.size());

      Value enableTrue;
      auto getEnableTrue = [&]() -> Value {
        if (!enableTrue)
          enableTrue = hw::ConstantOp::create(builder, readOp->getLoc(),
                                              builder.getI1Type(), 1)
                           .getResult();
        return enableTrue;
      };

      for (auto store : field.stores) {
        bool regionDominatesRead = false;
        for (Operation *parent = store->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (isa<hw::HWModuleOp>(parent))
            break;
          if (dom.dominates(parent, readOp)) {
            regionDominatesRead = true;
            break;
          }
        }
        if (!regionDominatesRead &&
            !postDom.postDominates(readBlock, store->getBlock()))
          return false;
        auto condOpt = buildStoreCondition(store);
        if (!condOpt)
          return false;
        BoolCondition cond = *condOpt;
        if (cond.isFalse())
          continue;
        Value stored = unwrapStoredValue(store.getValue());
        if (!stored || stored.getType() != valueType)
          return false;
        if (!dom.dominates(stored, readOp))
          return false;

        Value enable = cond.isTrue() ? getEnableTrue()
                                     : cond.materialize(builder,
                                                        readOp->getLoc());
        if (!enable || !enable.getType().isInteger(1))
          return false;
        if (!dom.dominates(enable, readOp))
          return false;
        values.push_back(stored);
        enables.push_back(enable);
      }

      if (values.empty())
        return false;

      builder.setInsertionPoint(readOp);
      Value resolved;
      if (isFourState) {
        resolved = resolveFourStateValuesWithEnable(builder, readOp->getLoc(),
                                                    values, enables);
      } else {
        auto baseName = sigOp.getNameAttr()
                            ? sigOp.getNameAttr().getValue()
                            : StringRef("llhd_if");
        std::string name = baseName.str();
        name += "_field";
        name += std::to_string(entry.first);
        name += "_unknown";
        Value unknownValue = state.addInput(module, name, valueType);
        resolved = resolveTwoStateValuesWithEnable(builder, readOp->getLoc(),
                                                   values, enables,
                                                   unknownValue);
      }
      if (!resolved || resolved.getType() != valueType)
        return false;
      for (auto read : field.reads)
        read.getResult().replaceAllUsesWith(resolved);
      return true;
    };

    if (!field.stores.empty() && !field.reads.empty()) {
      if (resolveStoresByDominance())
        continue;
      if (resolveStoresByComplementaryConditions())
        continue;
      if (resolveStoresByExclusiveConditions())
        continue;
      if (resolveStoresByEnableResolution())
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
    if (auto *def = init.getDefiningOp()) {
      if (def->use_empty()) {
        // If the init is an addressof, also erase the global if unused.
        if (auto addr = dyn_cast<LLVM::AddressOfOp>(def)) {
          if (auto mod = addr->getParentOfType<ModuleOp>()) {
            auto global = mod.lookupSymbol<LLVM::GlobalOp>(
                addr.getGlobalNameAttr().getValue());
            def->erase();
            if (global && global.getSymbolUses(mod)->empty())
              global.erase();
          } else {
            def->erase();
          }
        } else {
          def->erase();
        }
      }
    }
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
  if (hasLLHD && requireNoLLHD) {
    module.emitError("LLHD operations are not supported by circt-lec");
    return signalPassFailure();
  }
}
