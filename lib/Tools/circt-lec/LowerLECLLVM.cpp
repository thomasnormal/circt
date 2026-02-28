//===- LowerLECLLVM.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower simple LLVM struct construction patterns to HW structs for LEC.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Support/FourStateUtils.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Transforms/Mem2Reg.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <optional>

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_LOWERLECLLVM
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {
struct LowerLECLLVMPass
    : public circt::impl::LowerLECLLVMBase<LowerLECLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<comb::CombDialect, hw::HWDialect, llhd::LLHDDialect>();
  }
  void runOnOperation() override;
};

static bool isSupportedScalarType(Type type) {
  return isa<IntegerType, FloatType>(type);
}

static std::optional<uint64_t> getHWAggregateBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto structType = dyn_cast<hw::StructType>(type)) {
    uint64_t total = 0;
    for (const auto &field : structType.getElements()) {
      auto fieldWidth = getHWAggregateBitWidth(field.type);
      if (!fieldWidth)
        return std::nullopt;
      if (total > std::numeric_limits<uint64_t>::max() - *fieldWidth)
        return std::nullopt;
      total += *fieldWidth;
    }
    return total;
  }
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    auto elemWidth = getHWAggregateBitWidth(arrayType.getElementType());
    if (!elemWidth)
      return std::nullopt;
    uint64_t count = arrayType.getNumElements();
    if (count && *elemWidth > std::numeric_limits<uint64_t>::max() / count)
      return std::nullopt;
    return *elemWidth * count;
  }
  return std::nullopt;
}

static std::optional<TypedAttr> coerceToTypedScalarAttr(Attribute attr,
                                                        Type targetType) {
  if (!attr)
    return std::nullopt;
  if (!isSupportedScalarType(targetType))
    return std::nullopt;

  if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
    if (typedAttr.getType() == targetType)
      return typedAttr;
  }

  if (auto intTy = dyn_cast<IntegerType>(targetType)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return IntegerAttr::get(intTy, intAttr.getValue().zextOrTrunc(intTy.getWidth()));
    if (auto boolAttr = dyn_cast<BoolAttr>(attr))
      return IntegerAttr::get(intTy, boolAttr.getValue() ? 1 : 0);
    if (isa<LLVM::ZeroAttr>(attr))
      return IntegerAttr::get(intTy, 0);
    return std::nullopt;
  }

  if (auto floatTy = dyn_cast<FloatType>(targetType)) {
    if (auto floatAttr = dyn_cast<FloatAttr>(attr))
      return FloatAttr::get(floatTy, floatAttr.getValue());
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return FloatAttr::get(floatTy, intAttr.getValue().isZero() ? 0.0 : 1.0);
    if (isa<LLVM::ZeroAttr>(attr))
      return FloatAttr::get(floatTy, 0.0);
  }
  return std::nullopt;
}

struct GlobalLoadAccess {
  LLVM::AddressOfOp addrOf;
  SmallVector<LLVM::GEPOp, 4> geps;
};

static std::optional<GlobalLoadAccess> resolveGlobalLoadAccess(Value ptr) {
  GlobalLoadAccess access;
  while (true) {
    if (auto addrOf = ptr.getDefiningOp<LLVM::AddressOfOp>()) {
      access.addrOf = addrOf;
      return access;
    }
    if (auto gep = ptr.getDefiningOp<LLVM::GEPOp>()) {
      access.geps.push_back(gep);
      ptr = gep.getBase();
      continue;
    }
    if (auto bitcast = ptr.getDefiningOp<LLVM::BitcastOp>()) {
      ptr = bitcast.getArg();
      continue;
    }
    if (auto addrCast = ptr.getDefiningOp<LLVM::AddrSpaceCastOp>()) {
      ptr = addrCast.getArg();
      continue;
    }
    if (auto cast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
        return std::nullopt;
      ptr = cast.getOperand(0);
      continue;
    }
    return std::nullopt;
  }
}

static std::optional<StringAttr> getRootGlobalName(Value ptr) {
  auto access = resolveGlobalLoadAccess(ptr);
  if (!access)
    return std::nullopt;
  return access->addrOf.getGlobalNameAttr().getAttr();
}

static bool hasAnyStoreToGlobal(ModuleOp module, StringAttr globalName,
                                DenseMap<StringAttr, bool> &cache) {
  auto it = cache.find(globalName);
  if (it != cache.end())
    return it->second;

  bool hasStore = false;
  module.walk([&](LLVM::StoreOp store) -> WalkResult {
    auto rootName = getRootGlobalName(store.getAddr());
    if (!rootName || *rootName != globalName)
      return WalkResult::advance();
    hasStore = true;
    return WalkResult::interrupt();
  });
  cache.try_emplace(globalName, hasStore);
  return hasStore;
}

static std::optional<TypedAttr> extractGlobalScalarLoadConstant(
    LLVM::GlobalOp global, Type loadType, ArrayRef<LLVM::GEPOp> geps) {
  // Keep this targeted to direct scalar globals used by proc-assert guards.
  if (!geps.empty())
    return std::nullopt;
  if (global.getGlobalType() != loadType)
    return std::nullopt;
  if (!isSupportedScalarType(loadType))
    return std::nullopt;

  if (auto typedAttr = coerceToTypedScalarAttr(global.getValueOrNull(), loadType))
    return typedAttr;

  auto *initializerBlock = global.getInitializerBlock();
  if (!initializerBlock)
    return std::nullopt;
  auto initRet = dyn_cast<LLVM::ReturnOp>(initializerBlock->getTerminator());
  if (!initRet || initRet.getNumOperands() != 1)
    return std::nullopt;

  Value initValue = initRet.getOperand(0);
  if (auto cst = initValue.getDefiningOp<LLVM::ConstantOp>())
    return coerceToTypedScalarAttr(cst.getValue(), loadType);
  if (initValue.getDefiningOp<LLVM::ZeroOp>()) {
    if (auto intTy = dyn_cast<IntegerType>(loadType))
      return IntegerAttr::get(intTy, 0);
    if (auto floatTy = dyn_cast<FloatType>(loadType))
      return FloatAttr::get(floatTy, 0.0);
  }
  return std::nullopt;
}

static void eraseUnusedAddressChain(Value value) {
  auto *def = value.getDefiningOp();
  if (!def || !def->use_empty())
    return;

  Value next;
  if (auto gep = dyn_cast<LLVM::GEPOp>(def)) {
    next = gep.getBase();
  } else if (auto bitcast = dyn_cast<LLVM::BitcastOp>(def)) {
    next = bitcast.getArg();
  } else if (auto addrCast = dyn_cast<LLVM::AddrSpaceCastOp>(def)) {
    next = addrCast.getArg();
  } else if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return;
    next = cast.getOperand(0);
  } else if (isa<LLVM::AddressOfOp>(def)) {
    def->erase();
    return;
  } else {
    return;
  }

  def->erase();
  eraseUnusedAddressChain(next);
}

static bool foldSingleBlockAlloca(LLVM::AllocaOp alloca) {
  Value ptr = alloca.getResult();
  Block *singleBlock = nullptr;
  bool sawAccess = false;
  for (Operation *user : ptr.getUsers()) {
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      sawAccess = true;
      if (!singleBlock)
        singleBlock = load->getBlock();
      else if (singleBlock != load->getBlock())
        return false;
      continue;
    }
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      sawAccess = true;
      if (!singleBlock)
        singleBlock = store->getBlock();
      else if (singleBlock != store->getBlock())
        return false;
      continue;
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (!cast->use_empty())
        return false;
      continue;
    }
    return false;
  }
  if (!sawAccess || !singleBlock)
    return false;

  bool hasCurrentValue = false;
  for (Operation &op : *singleBlock) {
    if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
      if (store.getAddr() == ptr) {
        hasCurrentValue = true;
        continue;
      }
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
      if (load.getAddr() == ptr && !hasCurrentValue)
        return false;
    }
  }

  Value currentValue;
  SmallVector<Operation *, 16> eraseList;
  for (Operation &op : *singleBlock) {
    if (auto store = dyn_cast<LLVM::StoreOp>(&op)) {
      if (store.getAddr() == ptr) {
        currentValue = store.getValue();
        eraseList.push_back(store);
      }
      continue;
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
      if (load.getAddr() == ptr) {
        load.getResult().replaceAllUsesWith(currentValue);
        eraseList.push_back(load);
      }
    }
  }
  for (Operation *op : eraseList)
    op->erase();
  if (ptr.use_empty())
    alloca.erase();
  return true;
}

static bool eraseDeadAllocaStores(LLVM::AllocaOp alloca) {
  Value ptr = alloca.getResult();
  SmallVector<Operation *, 4> eraseList;
  SmallVector<LLVM::StoreOp, 4> stores;
  for (Operation *user : ptr.getUsers()) {
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      stores.push_back(store);
      continue;
    }
    if (auto bitcast = dyn_cast<LLVM::BitcastOp>(user)) {
      if (bitcast->use_empty()) {
        eraseList.push_back(bitcast);
        continue;
      }
    }
    if (auto addr = dyn_cast<LLVM::AddrSpaceCastOp>(user)) {
      if (addr->use_empty()) {
        eraseList.push_back(addr);
        continue;
      }
    }
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (cast->use_empty()) {
        eraseList.push_back(cast);
        continue;
      }
    }
    return false;
  }
  for (LLVM::StoreOp store : stores)
    eraseList.push_back(store);
  for (Operation *op : eraseList)
    op->erase();
  alloca.erase();
  return true;
}

static bool forwardSingleStoreAlloca(LLVM::AllocaOp alloca,
                                     DominanceInfo &domInfo) {
  Value ptr = alloca.getResult();
  LLVM::StoreOp singleStore;
  SmallVector<LLVM::LoadOp, 8> loads;
  SmallVector<Operation *, 8> deadCasts;
  for (Operation *user : ptr.getUsers()) {
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      if (load.getAddr() != ptr)
        return false;
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (store.getAddr() != ptr)
        return false;
      if (singleStore)
        return false;
      singleStore = store;
      continue;
    }
    if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, UnrealizedConversionCastOp>(
            user)) {
      if (!user->use_empty())
        return false;
      deadCasts.push_back(user);
      continue;
    }
    return false;
  }
  if (!singleStore || loads.empty())
    return false;

  Value storedValue = singleStore.getValue();
  if (storedValue.getType() != loads.front().getType())
    return false;
  for (LLVM::LoadOp load : loads) {
    if (load.getType() != storedValue.getType())
      return false;
    if (!domInfo.dominates(singleStore, load))
      return false;
  }

  for (LLVM::LoadOp load : loads) {
    load.replaceAllUsesWith(storedValue);
    load.erase();
  }
  singleStore.erase();
  for (Operation *cast : deadCasts)
    cast->erase();
  if (ptr.use_empty())
    alloca.erase();
  return true;
}

static bool replaceUninitializedAllocaLoadsWithUndef(LLVM::AllocaOp alloca) {
  Value ptr = alloca.getResult();
  SmallVector<LLVM::LoadOp, 8> loads;
  SmallVector<Operation *, 8> deadCasts;
  bool sawStore = false;
  for (Operation *user : ptr.getUsers()) {
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      if (load.getAddr() != ptr)
        return false;
      loads.push_back(load);
      continue;
    }
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      if (store.getAddr() != ptr)
        return false;
      sawStore = true;
      continue;
    }
    if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, UnrealizedConversionCastOp>(
            user)) {
      if (!user->use_empty())
        return false;
      deadCasts.push_back(user);
      continue;
    }
    return false;
  }
  if (sawStore || loads.empty())
    return false;

  for (LLVM::LoadOp load : loads) {
    OpBuilder builder(load);
    auto undef = LLVM::UndefOp::create(builder, load.getLoc(), load.getType());
    load.getResult().replaceAllUsesWith(undef.getResult());
    load.erase();
  }
  for (Operation *cast : deadCasts)
    cast->erase();
  if (ptr.use_empty())
    alloca.erase();
  return true;
}

static Value computeHWValueBeforeOp(Operation *op, Value ptr, Type hwType,
                                    DominanceInfo &domInfo,
                                    OpBuilder &builder, Location loc,
                                    DenseMap<Value, Value> *allocaSignals);
static LLVM::AllocaOp resolveAllocaBase(
    Value ptr, llvm::SmallPtrSetImpl<Value> &visited);
static Value buildHWAggregateFromLLVM(Value llvmValue, Type hwType,
                                      OpBuilder &builder, Location loc,
                                      ArrayRef<int64_t> path,
                                      bool isUnknownLeaf = false,
                                      llvm::SmallDenseSet<size_t, 16> *visiting =
                                          nullptr);

using EntryKey = std::pair<void *, void *>;

struct EntryKeyInfo {
  static inline EntryKey getEmptyKey() {
    return {llvm::DenseMapInfo<void *>::getEmptyKey(),
            llvm::DenseMapInfo<void *>::getEmptyKey()};
  }
  static inline EntryKey getTombstoneKey() {
    return {llvm::DenseMapInfo<void *>::getTombstoneKey(),
            llvm::DenseMapInfo<void *>::getTombstoneKey()};
  }
  static unsigned getHashValue(const EntryKey &key) {
    return llvm::hash_combine(key.first, key.second);
  }
  static bool isEqual(const EntryKey &lhs, const EntryKey &rhs) {
    return lhs == rhs;
  }
};

static EntryKey getEntryKey(Block *block, Value ptr) {
  return {block, ptr.getAsOpaquePointer()};
}

static hw::StructType getNestedStructType(hw::StructType hwType,
                                          ArrayRef<int64_t> prefix) {
  hw::StructType current = hwType;
  for (int64_t index : prefix) {
    auto elements = current.getElements();
    if (index < 0 || static_cast<size_t>(index) >= elements.size())
      return {};
    auto nested = dyn_cast<hw::StructType>(elements[index].type);
    if (!nested)
      return {};
    current = nested;
  }
  return current;
}

enum class AggregateDefaultKind { none, undef, zero };

static AggregateDefaultKind getAggregateDefaultKind(Value value) {
  while (auto insert = value.getDefiningOp<LLVM::InsertValueOp>())
    value = insert.getContainer();
  if (value.getDefiningOp<LLVM::UndefOp>())
    return AggregateDefaultKind::undef;
  if (value.getDefiningOp<LLVM::ZeroOp>())
    return AggregateDefaultKind::zero;
  return AggregateDefaultKind::none;
}

static bool isInsertValueChainAggregate(Value value,
                                        unsigned maxInsertOps = 4096) {
  unsigned insertCount = 0;
  while (auto insert = value.getDefiningOp<LLVM::InsertValueOp>()) {
    if (++insertCount > maxInsertOps)
      return false;
    value = insert.getContainer();
  }
  return insertCount != 0;
}

static hw::StructType getFourStateStructForLLVM(LLVM::LLVMStructType llvmType) {
  if (llvmType.isOpaque())
    return {};
  auto body = llvmType.getBody();
  if (body.size() != 2)
    return {};
  auto valueType = dyn_cast<IntegerType>(body[0]);
  auto unknownType = dyn_cast<IntegerType>(body[1]);
  if (!valueType || !unknownType)
    return {};
  if (valueType.getWidth() != unknownType.getWidth())
    return {};
  auto *ctx = llvmType.getContext();
  return hw::StructType::get(
      ctx, {{StringAttr::get(ctx, "value"), valueType},
            {StringAttr::get(ctx, "unknown"), unknownType}});
}

static Value extractFromHWStruct(Value hwStruct, ArrayRef<int64_t> path,
                                 OpBuilder &builder, Location loc) {
  Value current = hwStruct;
  Type currentType = hwStruct.getType();
  for (int64_t index : path) {
    auto structTy = dyn_cast<hw::StructType>(currentType);
    if (!structTy)
      return {};
    auto elements = structTy.getElements();
    if (index < 0 || static_cast<size_t>(index) >= elements.size())
      return {};
    auto field = elements[index];
    current = hw::StructExtractOp::create(builder, loc, current, field.name);
    currentType = field.type;
  }
  return current;
}

static Value findInsertedValue(Value value, hw::StructType hwType,
                               ArrayRef<int64_t> path,
                               DominanceInfo *domInfo, OpBuilder &builder,
                               Location loc,
                               DenseMap<Value, Value> *allocaSignals) {
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 &&
        cast->getOperand(0).getType() == hwType)
      return extractFromHWStruct(cast->getOperand(0), path, builder, loc);
  }
  if (auto mux = value.getDefiningOp<comb::MuxOp>()) {
    Value trueVal =
        findInsertedValue(mux.getTrueValue(), hwType, path, domInfo, builder,
                          loc, allocaSignals);
    if (!trueVal)
      return {};
    Value falseVal =
        findInsertedValue(mux.getFalseValue(), hwType, path, domInfo, builder,
                          loc, allocaSignals);
    if (!falseVal || falseVal.getType() != trueVal.getType())
      return {};
    return comb::MuxOp::create(builder, loc, mux.getCond(), trueVal, falseVal,
                               mux.getTwoState());
  }

  if (auto select = value.getDefiningOp<LLVM::SelectOp>()) {
    Value trueVal = findInsertedValue(select.getTrueValue(), hwType, path,
                                      domInfo, builder, loc, allocaSignals);
    if (!trueVal)
      return {};
    Value falseVal = findInsertedValue(select.getFalseValue(), hwType, path,
                                       domInfo, builder, loc, allocaSignals);
    if (!falseVal || falseVal.getType() != trueVal.getType())
      return {};
    return comb::MuxOp::create(builder, loc, select.getCondition(), trueVal,
                               falseVal);
  }

  if (auto load = value.getDefiningOp<LLVM::LoadOp>()) {
    if (!domInfo)
      return {};
    Value hwStruct = computeHWValueBeforeOp(load, load.getAddr(), hwType,
                                            *domInfo, builder, loc,
                                            allocaSignals);
    if (!hwStruct)
      return {};
    return extractFromHWStruct(hwStruct, path, builder, loc);
  }

  auto insert = value.getDefiningOp<LLVM::InsertValueOp>();
  if (!insert)
    return {};

  ArrayRef<int64_t> insertPos = insert.getPosition();
  if (path == insertPos)
    return insert.getValue();

  if (path.size() >= insertPos.size() &&
      llvm::equal(insertPos, path.take_front(insertPos.size()))) {
    auto nestedType = getNestedStructType(hwType, insertPos);
    if (!nestedType)
      return {};
    if (Value nested = findInsertedValue(
            insert.getValue(), nestedType, path.drop_front(insertPos.size()),
            domInfo, builder, loc, allocaSignals))
      return nested;
  }
  return findInsertedValue(insert.getContainer(), hwType, path, domInfo,
                           builder, loc, allocaSignals);
}

static Value mergePredValues(Block *block,
                             ArrayRef<std::pair<Block *, Value>> preds,
                             OpBuilder &builder, Location loc);

static Value buildHWStructFromLLVM(Value llvmStruct, hw::StructType hwType,
                                   OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> prefix,
                                   DominanceInfo *domInfo,
                                   DenseMap<Value, Value> *allocaSignals) {
  AggregateDefaultKind defaultKind = getAggregateDefaultKind(llvmStruct);
  if (auto cast = llvmStruct.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 &&
        cast->getOperand(0).getType() == hwType)
      return cast->getOperand(0);
  }
  if (auto select = llvmStruct.getDefiningOp<LLVM::SelectOp>()) {
    Value trueVal =
        buildHWStructFromLLVM(select.getTrueValue(), hwType, builder, loc,
                              prefix, domInfo, allocaSignals);
    if (!trueVal)
      return {};
    Value falseVal =
        buildHWStructFromLLVM(select.getFalseValue(), hwType, builder, loc,
                              prefix, domInfo, allocaSignals);
    if (!falseVal || falseVal.getType() != trueVal.getType())
      return {};
    return comb::MuxOp::create(builder, loc, select.getCondition(), trueVal,
                               falseVal);
  }
  if (auto arg = dyn_cast<BlockArgument>(llvmStruct)) {
    if (!isa<LLVM::LLVMStructType>(arg.getType()))
      return {};
    Block *block = arg.getOwner();
    SmallVector<std::pair<Block *, Value>, 4> predValues;
    for (Block *pred : block->getPredecessors()) {
      auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
      if (!branch)
        return {};
      unsigned succIndex = 0;
      bool found = false;
      for (unsigned idx = 0, e = branch->getNumSuccessors(); idx < e; ++idx) {
        if (branch->getSuccessor(idx) == block) {
          succIndex = idx;
          found = true;
          break;
        }
      }
      if (!found)
        return {};
      auto succOperands = branch.getSuccessorOperands(succIndex);
      if (arg.getArgNumber() >= succOperands.size())
        return {};
      Value incoming = succOperands[arg.getArgNumber()];
      OpBuilder predBuilder(pred->getTerminator());
      Value hwValue = buildHWStructFromLLVM(incoming, hwType, predBuilder, loc,
                                            {}, domInfo, allocaSignals);
      if (!hwValue)
        return {};
      predValues.push_back({pred, hwValue});
    }
    Value merged = mergePredValues(block, predValues, builder, loc);
    if (merged)
      return merged;
  }

  SmallVector<Value, 8> fields;
  auto elements = hwType.getElements();
  fields.reserve(elements.size());
  for (auto [index, field] : llvm::enumerate(elements)) {
    SmallVector<int64_t, 4> path(prefix.begin(), prefix.end());
    path.push_back(static_cast<int64_t>(index));
    if (auto nested = dyn_cast<hw::StructType>(field.type)) {
      Value nestedValue =
          buildHWStructFromLLVM(llvmStruct, nested, builder, loc, path,
                                domInfo, allocaSignals);
      if (!nestedValue)
        return {};
      fields.push_back(nestedValue);
      continue;
    }
    Value leaf = findInsertedValue(llvmStruct, hwType, path, domInfo, builder,
                                   loc, allocaSignals);
    if (!leaf || leaf.getType() != field.type) {
      if (defaultKind == AggregateDefaultKind::none)
        return {};
      auto fieldInt = dyn_cast<IntegerType>(field.type);
      if (!fieldInt)
        return {};
      bool isUnknownField =
          field.name && field.name.getValue() == "unknown";
      bool useZeroUnknown = false;
      if (defaultKind == AggregateDefaultKind::undef && isUnknownField &&
          isFourStateStructType(hwType)) {
        SmallVector<int64_t, 4> valuePath(prefix.begin(), prefix.end());
        valuePath.push_back(0);
        Value valueLeaf = findInsertedValue(
            llvmStruct, hwType, valuePath, domInfo, builder, loc, allocaSignals);
        if (valueLeaf && isa<IntegerType>(valueLeaf.getType()))
          useZeroUnknown = true;
      }
      APInt value(fieldInt.getWidth(), 0);
      if (defaultKind == AggregateDefaultKind::undef && isUnknownField &&
          !useZeroUnknown)
        value = APInt::getAllOnes(fieldInt.getWidth());
      leaf = hw::ConstantOp::create(
          builder, loc, builder.getIntegerAttr(fieldInt, value));
    }
    fields.push_back(leaf);
  }
  return hw::StructCreateOp::create(builder, loc, hwType, fields);
}

static Value lowerStoredAggregateToHW(Value storedValue, Type hwType,
                                      OpBuilder &builder, Location loc,
                                      DominanceInfo *domInfo,
                                      DenseMap<Value, Value> *allocaSignals) {
  if (storedValue.getType() == hwType)
    return storedValue;
  if (auto cast = storedValue.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast.getOperand(0).getType() == hwType)
      return cast.getOperand(0);
  }
  if (auto hwStructType = dyn_cast<hw::StructType>(hwType)) {
    if (isa<LLVM::LLVMStructType>(storedValue.getType()))
      return buildHWStructFromLLVM(storedValue, hwStructType, builder, loc, {},
                                   domInfo, allocaSignals);
  }
  if (isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(storedValue.getType()))
    return buildHWAggregateFromLLVM(storedValue, hwType, builder, loc, {},
                                    /*isUnknownLeaf=*/false);
  return {};
}

static Value mergePredValues(Block *block,
                             ArrayRef<std::pair<Block *, Value>> preds,
                             OpBuilder &builder, Location loc) {
  if (preds.empty())
    return {};
  if (preds.size() == 1)
    return preds.front().second;

  Type valueType = preds.front().second.getType();
  for (auto &pred : preds)
    if (pred.second.getType() != valueType)
      return {};

  SmallVector<std::pair<BranchOpInterface, SmallVector<unsigned, 2>>, 4>
      predBranches;
  for (auto &pred : preds) {
    auto branch = dyn_cast<BranchOpInterface>(pred.first->getTerminator());
    if (!branch)
      return {};
    SmallVector<unsigned, 2> succIndices;
    for (unsigned idx = 0, e = branch->getNumSuccessors(); idx < e; ++idx) {
      if (branch->getSuccessor(idx) == block)
        succIndices.push_back(idx);
    }
    if (succIndices.empty())
      return {};
    predBranches.push_back({branch, succIndices});
  }

  unsigned argIndex = block->getNumArguments();
  for (auto &entry : predBranches) {
    auto branch = entry.first;
    auto &succIndices = entry.second;
    for (unsigned succIndex : succIndices) {
      auto succOperands = branch.getSuccessorOperands(succIndex);
      if (succOperands.size() != argIndex)
        return {};
      if (succOperands.isOperandProduced(argIndex))
        return {};
    }
  }

  block->addArgument(valueType, loc);
  for (auto entry : llvm::enumerate(predBranches)) {
    auto &pred = preds[entry.index()];
    auto branch = entry.value().first;
    auto &succIndices = entry.value().second;
    for (unsigned succIndex : succIndices) {
      auto succOperands = branch.getSuccessorOperands(succIndex);
      succOperands.append(pred.second);
    }
  }
  return block->getArgument(argIndex);
}

static Value computeHWValueAtBlockEntry(
    Block *block, Value ptr, Type hwType,
    DenseMap<EntryKey, Value, EntryKeyInfo> &entryCache,
    llvm::SmallDenseSet<EntryKey, 8, EntryKeyInfo> &visiting,
    DominanceInfo &domInfo,
    const llvm::SmallPtrSetImpl<Block *> &storeBlocks, OpBuilder &builder,
    Location loc);

static Value computeHWValueAtBlockExit(
    Block *block, Value ptr, Type hwType,
    DenseMap<EntryKey, Value, EntryKeyInfo> &entryCache,
    llvm::SmallDenseSet<EntryKey, 8, EntryKeyInfo> &visiting,
    DominanceInfo &domInfo,
    const llvm::SmallPtrSetImpl<Block *> &storeBlocks, OpBuilder &builder,
    Location loc) {
  Value current =
      computeHWValueAtBlockEntry(block, ptr, hwType, entryCache, visiting,
                                 domInfo, storeBlocks, builder, loc);
  OpBuilder blockBuilder(block->getTerminator());
  for (Operation &op : *block) {
    auto store = dyn_cast<LLVM::StoreOp>(&op);
    if (!store || store.getAddr() != ptr)
      continue;
    Value stored = lowerStoredAggregateToHW(store.getValue(), hwType,
                                            blockBuilder, loc, &domInfo,
                                            /*allocaSignals=*/nullptr);
    if (!stored)
      return {};
    current = stored;
  }
  return current;
}

static Value mapPtrForPred(Block *block, Value ptr, Block *pred) {
  auto arg = dyn_cast<BlockArgument>(ptr);
  if (!arg || arg.getOwner() != block)
    return ptr;
  auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
  if (!branch)
    return {};
  unsigned succIndex = 0;
  bool found = false;
  for (unsigned idx = 0, e = branch->getNumSuccessors(); idx < e; ++idx) {
    if (branch->getSuccessor(idx) == block) {
      succIndex = idx;
      found = true;
      break;
    }
  }
  if (!found)
    return {};
  auto succOperands = branch.getSuccessorOperands(succIndex);
  if (arg.getArgNumber() >= succOperands.size())
    return {};
  return succOperands[arg.getArgNumber()];
}

static Value getDominatingStoredValue(Operation *op, Value ptr,
                                      Type hwType,
                                      DominanceInfo &domInfo,
                                      OpBuilder &builder, Location loc) {
  LLVM::StoreOp bestStore;
  for (Operation *user : ptr.getUsers()) {
    auto store = dyn_cast<LLVM::StoreOp>(user);
    if (!store || store.getAddr() != ptr)
      continue;
    if (!domInfo.dominates(store, op))
      continue;
    if (!bestStore || domInfo.dominates(bestStore, store))
      bestStore = store;
  }
  if (!bestStore)
    return {};
  return lowerStoredAggregateToHW(bestStore.getValue(), hwType, builder, loc,
                                  &domInfo, /*allocaSignals=*/nullptr);
}

static Value probeLLHDRefForPtr(Value ptr, Type hwType, OpBuilder &builder,
                                Location loc) {
  auto findRefTypeForValue = [&](Value value) -> llhd::RefType {
    for (Operation *user : value.getUsers()) {
      auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!cast || cast.getInputs().size() != 1 ||
          cast.getResults().size() != 1 || cast.getInputs().front() != value)
        continue;
      auto refType = dyn_cast<llhd::RefType>(cast.getResult(0).getType());
      if (!refType || refType.getNestedType() != hwType)
        continue;
      return refType;
    }
    return {};
  };

  llhd::RefType refType = findRefTypeForValue(ptr);
  if (!refType) {
    llvm::SmallPtrSet<Value, 8> visited;
    if (auto alloca = resolveAllocaBase(ptr, visited))
      refType = findRefTypeForValue(alloca.getResult());
  }
  if (!refType)
    return {};

  auto cast = UnrealizedConversionCastOp::create(
      builder, loc, TypeRange{refType}, ValueRange{ptr});
  Value probe = llhd::ProbeOp::create(builder, loc, cast.getResult(0));
  if (probe.getType() != hwType)
    return {};
  return probe;
}

static Value computeHWValueAtBlockEntry(
    Block *block, Value ptr, Type hwType,
    DenseMap<EntryKey, Value, EntryKeyInfo> &entryCache,
    llvm::SmallDenseSet<EntryKey, 8, EntryKeyInfo> &visiting,
    DominanceInfo &domInfo,
    const llvm::SmallPtrSetImpl<Block *> &storeBlocks, OpBuilder &builder,
    Location loc) {
  EntryKey key = getEntryKey(block, ptr);
  if (auto cached = entryCache.lookup(key))
    return cached;
  if (visiting.contains(key))
    return {};
  visiting.insert(key);
  SmallVector<std::pair<Block *, Value>, 4> predValues;
  OpBuilder entryBuilder(block, block->begin());
  for (Block *pred : block->getPredecessors()) {
    Value predPtr = mapPtrForPred(block, ptr, pred);
    if (!predPtr) {
      visiting.erase(key);
      return {};
    }
    EntryKey predKey = getEntryKey(pred, predPtr);
    if (visiting.contains(predKey)) {
      if (!storeBlocks.contains(pred))
        continue;
      visiting.erase(key);
      return {};
    }
    Value predExit =
        computeHWValueAtBlockExit(pred, predPtr, hwType, entryCache, visiting,
                                  domInfo, storeBlocks, builder, loc);
    if (!predExit) {
      if (!storeBlocks.contains(pred))
        continue;
      visiting.erase(key);
      return {};
    }
    predValues.push_back({pred, predExit});
  }
  Value merged = mergePredValues(block, predValues, entryBuilder, loc);
  visiting.erase(key);
  if (!merged)
    return {};
  entryCache[key] = merged;
  return merged;
}

static Value computeHWValueBeforeOp(Operation *op, Value ptr,
                                    Type hwType,
                                    DominanceInfo &domInfo,
                                    OpBuilder &builder, Location loc,
                                    DenseMap<Value, Value> *allocaSignals) {
  if (Value direct = getDominatingStoredValue(op, ptr, hwType, domInfo, builder,
                                              loc))
    return direct;

  auto probeSignalForPtr = [&](Value ptrValue, OpBuilder &probeBuilder,
                               Location probeLoc) -> Value {
    llvm::SmallPtrSet<Value, 8> visited;
    auto baseAlloca = resolveAllocaBase(ptrValue, visited);
    if (!baseAlloca)
      return {};
    Value sig;
    if (allocaSignals)
      sig = allocaSignals->lookup(baseAlloca.getResult());
    if (!sig) {
      for (Operation *user : baseAlloca.getResult().getUsers()) {
        auto cast = dyn_cast<UnrealizedConversionCastOp>(user);
        if (!cast || cast.getInputs().size() != 1 ||
            cast.getResults().size() != 1)
          continue;
        auto refType = dyn_cast<llhd::RefType>(cast.getResult(0).getType());
        if (!refType || refType.getNestedType() != hwType)
          continue;
        sig = cast.getResult(0);
        break;
      }
    }
    if (!sig)
      return {};
    Value probe = llhd::ProbeOp::create(probeBuilder, probeLoc, sig);
    if (probe.getType() != hwType)
      return {};
    return probe;
  };
  if (auto arg = dyn_cast<BlockArgument>(ptr)) {
    Block *block = arg.getOwner();
    SmallVector<std::pair<Block *, Value>, 4> predPtrs;
    SmallVector<std::pair<Block *, Value>, 4> predValues;
    bool allDominating = true;
    for (Block *pred : block->getPredecessors()) {
      Value predPtr = mapPtrForPred(block, ptr, pred);
      if (!predPtr)
        return {};
      predPtrs.push_back({pred, predPtr});
      Value stored =
          getDominatingStoredValue(pred->getTerminator(), predPtr, hwType,
                                   domInfo, builder, loc);
      if (!stored) {
        allDominating = false;
        break;
      }
      predValues.push_back({pred, stored});
    }
    if (allDominating && !predValues.empty()) {
      if (Value merged =
            mergePredValues(block, predValues, builder, loc))
        return merged;
    }

    if (allocaSignals) {
      predValues.clear();
      for (auto [pred, predPtr] : predPtrs) {
        OpBuilder predBuilder(pred->getTerminator());
        Value probe = probeSignalForPtr(predPtr, predBuilder, loc);
        if (!probe)
          break;
        predValues.push_back({pred, probe});
      }
      if (!predValues.empty() && predValues.size() == predPtrs.size()) {
        if (Value merged =
              mergePredValues(block, predValues, builder, loc))
          return merged;
      }
    }

    if (!predPtrs.empty()) {
      llvm::SmallPtrSet<Block *, 8> storeBlocks;
      for (auto [pred, predPtr] : predPtrs) {
        (void)pred;
        for (Operation *user : predPtr.getUsers()) {
          auto store = dyn_cast<LLVM::StoreOp>(user);
          if (store && store.getAddr() == predPtr)
            storeBlocks.insert(store->getBlock());
        }
      }

      DenseMap<EntryKey, Value, EntryKeyInfo> entryCache;
      llvm::SmallDenseSet<EntryKey, 8, EntryKeyInfo> visiting;
      predValues.clear();
      for (auto [pred, predPtr] : predPtrs) {
        Value predExit =
            computeHWValueAtBlockExit(pred, predPtr, hwType, entryCache,
                                      visiting, domInfo, storeBlocks, builder,
                                      loc);
        if (!predExit)
          return {};
        predValues.push_back({pred, predExit});
      }
      if (!predValues.empty()) {
        if (Value merged =
              mergePredValues(block, predValues, builder, loc))
          return merged;
      }
    }
  }

  llvm::SmallPtrSet<Block *, 8> storeBlocks;
  for (Operation *user : ptr.getUsers()) {
    auto store = dyn_cast<LLVM::StoreOp>(user);
    if (store && store.getAddr() == ptr)
      storeBlocks.insert(store->getBlock());
  }
  DenseMap<EntryKey, Value, EntryKeyInfo> entryCache;
  llvm::SmallDenseSet<EntryKey, 8, EntryKeyInfo> visiting;
  Block *block = op->getBlock();
  Value current = computeHWValueAtBlockEntry(block, ptr, hwType, entryCache,
                                             visiting, domInfo, storeBlocks,
                                             builder, loc);
  for (Operation &it : *block) {
    if (&it == op)
      break;
    auto store = dyn_cast<LLVM::StoreOp>(&it);
    if (!store || store.getAddr() != ptr)
      continue;
    Value stored = lowerStoredAggregateToHW(store.getValue(), hwType, builder,
                                            loc, &domInfo,
                                            /*allocaSignals=*/nullptr);
    if (!stored)
      return {};
    current = stored;
  }
  return current;
}

static Value unwrapTrivialLoad(Value value) {
  auto load = value.getDefiningOp<LLVM::LoadOp>();
  if (!load)
    return value;
  Value ptr = load.getAddr();
  SmallVector<LLVM::StoreOp, 2> stores;
  for (Operation *user : ptr.getUsers()) {
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      stores.push_back(store);
      continue;
    }
    if (isa<LLVM::LoadOp>(user))
      continue;
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(user)) {
      if (cast->use_empty())
        continue;
    }
    return value;
  }
  if (stores.empty())
    return value;
  Block *block = load->getBlock();
  for (LLVM::StoreOp store : stores)
    if (store->getBlock() != block)
      return value;
  LLVM::StoreOp lastStore;
  for (Operation &op : *block) {
    if (&op == load)
      break;
    if (auto store = dyn_cast<LLVM::StoreOp>(&op))
      if (store.getAddr() == ptr)
        lastStore = store;
  }
  if (!lastStore)
    return value;
  return lastStore.getValue();
}

static Value unwrapHWStructCast(Value value, hw::StructType hwType) {
  auto cast = value.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1)
    return {};
  Value input = cast.getInputs().front();
  if (input.getType() != hwType)
    return {};
  return input;
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

static LLVM::AllocaOp resolveAllocaBase(Value ptr,
                                        llvm::SmallPtrSetImpl<Value> &visited) {
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

static bool dropUnusedBlockArguments(Region &region) {
  bool changed = false;
  bool localChange = true;
  // Skip entry blocks of hw.module operations - their block arguments are
  // tied to the module signature and cannot be dropped without also updating
  // the module type attribute.
  bool isHWModuleRegion = isa<hw::HWModuleOp>(region.getParentOp());
  while (localChange) {
    localChange = false;
    for (Block &block : region) {
      // Skip entry blocks for hw.module - they are tied to the module signature
      if (isHWModuleRegion && block.isEntryBlock())
        continue;
      for (int index = static_cast<int>(block.getNumArguments()) - 1; index >= 0;
           --index) {
        BlockArgument arg = block.getArgument(index);
        if (!arg.use_empty())
          continue;
        bool canDrop = true;
        SmallVector<std::pair<BranchOpInterface, unsigned>, 4> succs;
        for (Block *pred : block.getPredecessors()) {
          auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
          if (!branch) {
            canDrop = false;
            break;
          }
          bool found = false;
          for (unsigned succIndex = 0, e = branch->getNumSuccessors();
               succIndex < e; ++succIndex) {
            if (branch->getSuccessor(succIndex) != &block)
              continue;
            auto succOperands = branch.getSuccessorOperands(succIndex);
            if (index >= static_cast<int>(succOperands.size())) {
              canDrop = false;
              break;
            }
            if (succOperands.isOperandProduced(index)) {
              canDrop = false;
              break;
            }
            succs.push_back({branch, succIndex});
            found = true;
          }
          if (!found || !canDrop)
            break;
        }
        if (!canDrop)
          continue;
        for (auto [branch, succIndex] : succs) {
          auto succOperands = branch.getSuccessorOperands(succIndex);
          succOperands.erase(index);
        }
        block.eraseArgument(index);
        changed = true;
        localChange = true;
      }
    }
  }
  return changed;
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
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    SmallVector<Value, 4> elements;
    int64_t numElements = static_cast<int64_t>(arrayType.getNumElements());
    elements.reserve(numElements);
    for (int64_t index = 0; index < numElements; ++index) {
      Value element =
          buildDefaultValue(arrayType.getElementType(), builder, loc, zeroUnknown);
      if (!element)
        return {};
      elements.push_back(element);
    }
    return hw::ArrayCreateOp::create(builder, loc, arrayType, elements);
  }
  return {};
}

static Value findInsertedAggregateValue(Value value, ArrayRef<int64_t> path) {
  if (path.empty())
    return value;

  if (auto extract = value.getDefiningOp<LLVM::ExtractValueOp>()) {
    SmallVector<int64_t, 8> composedPath;
    composedPath.reserve(extract.getPosition().size() + path.size());
    llvm::append_range(composedPath, extract.getPosition());
    llvm::append_range(composedPath, path);
    return findInsertedAggregateValue(extract.getContainer(), composedPath);
  }

  auto insert = value.getDefiningOp<LLVM::InsertValueOp>();
  if (!insert)
    return {};
  ArrayRef<int64_t> pos = insert.getPosition();
  if (path.size() >= pos.size() &&
      llvm::equal(pos, path.take_front(pos.size()))) {
    if (path.size() == pos.size())
      return insert.getValue();
    if (Value nested = findInsertedAggregateValue(
            insert.getValue(), path.drop_front(pos.size())))
      return nested;
  }
  return findInsertedAggregateValue(insert.getContainer(), path);
}

static Value buildHWAggregateFromLLVM(Value llvmValue, Type hwType,
                                      OpBuilder &builder, Location loc,
                                      ArrayRef<int64_t> path,
                                      bool isUnknownLeaf,
                                      llvm::SmallDenseSet<size_t, 16> *visiting) {
  llvm::SmallDenseSet<size_t, 16> localVisiting;
  if (!visiting)
    visiting = &localVisiting;
  size_t key = static_cast<size_t>(
      llvm::hash_combine(llvmValue.getAsOpaquePointer(),
                         hwType.getAsOpaquePointer(),
                         llvm::hash_combine_range(path.begin(), path.end())));
  if (!visiting->insert(key).second)
    return {};
  auto eraseGuard = llvm::scope_exit([&]() { visiting->erase(key); });

  if (path.empty()) {
    if (auto cast = llvmValue.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 &&
          cast.getOperand(0).getType() == hwType)
        return cast.getOperand(0);
    }

    if (auto select = llvmValue.getDefiningOp<LLVM::SelectOp>()) {
      Value trueValue = buildHWAggregateFromLLVM(select.getTrueValue(), hwType,
                                                 builder, loc, path,
                                                 isUnknownLeaf, visiting);
      if (!trueValue)
        return {};
      Value falseValue = buildHWAggregateFromLLVM(select.getFalseValue(),
                                                  hwType, builder, loc, path,
                                                  isUnknownLeaf, visiting);
      if (!falseValue || falseValue.getType() != trueValue.getType())
        return {};
      return comb::MuxOp::create(builder, loc, select.getCondition(), trueValue,
                                 falseValue);
    }

    if (auto arg = dyn_cast<BlockArgument>(llvmValue)) {
      if (!isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(arg.getType()))
        return {};
      Block *block = arg.getOwner();
      SmallVector<std::pair<Block *, Value>, 4> predValues;
      for (Block *pred : block->getPredecessors()) {
        auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
        if (!branch)
          return {};
        unsigned succIndex = 0;
        bool found = false;
        for (unsigned idx = 0, e = branch->getNumSuccessors(); idx < e;
             ++idx) {
          if (branch->getSuccessor(idx) == block) {
            succIndex = idx;
            found = true;
            break;
          }
        }
        if (!found)
          return {};
        auto succOperands = branch.getSuccessorOperands(succIndex);
        if (arg.getArgNumber() >= succOperands.size())
          return {};
        Value incoming = succOperands[arg.getArgNumber()];
        OpBuilder predBuilder(pred->getTerminator());
        Value incomingValue = buildHWAggregateFromLLVM(
            incoming, hwType, predBuilder, loc, path, isUnknownLeaf, visiting);
        if (!incomingValue)
          return {};
        predValues.push_back({pred, incomingValue});
      }
      if (Value merged = mergePredValues(block, predValues, builder, loc))
        return merged;
    }
  }

  if (auto intType = dyn_cast<IntegerType>(hwType)) {
    Value leaf = findInsertedAggregateValue(llvmValue, path);
    if (leaf && leaf.getType() == intType)
      return leaf;

    auto defaultKind = getAggregateDefaultKind(llvmValue);
    if (defaultKind == AggregateDefaultKind::none)
      return {};
    APInt value(intType.getWidth(), 0);
    if (defaultKind == AggregateDefaultKind::undef && isUnknownLeaf)
      value = APInt::getAllOnes(intType.getWidth());
    return hw::ConstantOp::create(
        builder, loc, builder.getIntegerAttr(intType, value));
  }

  if (auto structType = dyn_cast<hw::StructType>(hwType)) {
    SmallVector<Value, 4> fields;
    fields.reserve(structType.getElements().size());
    for (auto [index, field] : llvm::enumerate(structType.getElements())) {
      SmallVector<int64_t, 8> fieldPath(path.begin(), path.end());
      fieldPath.push_back(static_cast<int64_t>(index));
      bool childUnknownLeaf = field.name && field.name.getValue() == "unknown";
      Value fieldValue =
          buildHWAggregateFromLLVM(llvmValue, field.type, builder, loc,
                                   fieldPath, childUnknownLeaf, visiting);
      if (!fieldValue)
        return {};
      fields.push_back(fieldValue);
    }
    return hw::StructCreateOp::create(builder, loc, structType, fields);
  }

  if (auto arrayType = dyn_cast<hw::ArrayType>(hwType)) {
    SmallVector<Value, 8> elements;
    int64_t numElements = static_cast<int64_t>(arrayType.getNumElements());
    elements.reserve(numElements);
    for (int64_t index = 0; index < numElements; ++index) {
      SmallVector<int64_t, 8> elementPath(path.begin(), path.end());
      elementPath.push_back(index);
      Value element = buildHWAggregateFromLLVM(
          llvmValue, arrayType.getElementType(), builder, loc, elementPath,
          isUnknownLeaf, visiting);
      if (!element)
        return {};
      elements.push_back(element);
    }
    return hw::ArrayCreateOp::create(builder, loc, arrayType, elements);
  }
  return {};
}

static bool rewriteAllocaBackedLLHDRef(UnrealizedConversionCastOp castOp,
                                       DenseMap<Value, Value> &allocaSignals) {
  if (castOp.getInputs().size() != 1 || castOp.getResults().size() != 1)
    return false;
  auto refType = dyn_cast<llhd::RefType>(castOp.getResult(0).getType());
  if (!refType)
    return false;
  Type nestedType = refType.getNestedType();
  bool isStructPayload = isa<hw::StructType>(nestedType);
  bool isArrayPayload = isa<hw::ArrayType>(nestedType);
  if (!isStructPayload && !isArrayPayload)
    return false;
  Value ptr = castOp.getInputs().front();
  if (!isa<LLVM::LLVMPointerType>(ptr.getType()))
    return false;
  llvm::SmallPtrSet<Value, 8> visited;
  auto alloca = resolveAllocaBase(ptr, visited);
  if (!alloca)
    return false;

  llvm::SmallPtrSet<Value, 16> derivedPtrs;
  SmallVector<Value, 16> worklist;
  bool sawPointerControlFlow = false;
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
        sawPointerControlFlow = true;
        llvm::SmallPtrSet<Value, 8> localVisited;
        if (resolveAllocaBase(select.getResult(), localVisited) == alloca)
          addDerived(select.getResult());
        continue;
      }
      if (auto branch = dyn_cast<BranchOpInterface>(user)) {
        sawPointerControlFlow = true;
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

  if (isArrayPayload) {
    if (sawPointerControlFlow)
      return false;
    if (llvm::any_of(derivedPtrs, [](Value value) {
          return isa<BlockArgument>(value);
        }))
      return false;
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
  for (auto cast : refCasts) {
    if (!derivedPtrs.contains(cast.getInputs().front()))
      return false;
  }

  for (auto store : stores) {
    if (!derivedPtrs.contains(store.getAddr()))
      return false;
  }
  for (auto load : loads) {
    if (!derivedPtrs.contains(load.getAddr()))
      return false;
  }
  if (isArrayPayload) {
    Block *baseBlock = alloca->getBlock();
    auto inBaseBlock = [baseBlock](Operation *op) {
      return op->getBlock() == baseBlock;
    };
    if (llvm::any_of(stores, [&](LLVM::StoreOp store) {
          return !inBaseBlock(store);
        }))
      return false;
    if (llvm::any_of(loads, [&](LLVM::LoadOp load) {
          return !inBaseBlock(load);
        }))
      return false;
    if (llvm::any_of(refCasts, [&](UnrealizedConversionCastOp cast) {
          return !inBaseBlock(cast);
        }))
      return false;
  }

  OpBuilder sigBuilder(alloca);
  // Local ref allocas act like temporary wires; initialize unknowns to 0.
  Value init =
      buildDefaultValue(nestedType, sigBuilder, castOp.getLoc(),
                        /*zeroUnknown=*/true);
  if (!init)
    return false;
  auto sigOp = llhd::SignalOp::create(sigBuilder, castOp.getLoc(), refType,
                                      StringAttr{}, init);
  sigOp->setAttr("lec.local", UnitAttr::get(sigBuilder.getContext()));
  auto sig = sigOp.getResult();
  allocaSignals[alloca.getResult()] = sig;

  for (auto refCast : refCasts) {
    refCast.getResult(0).replaceAllUsesWith(sig);
    refCast.erase();
  }

  DominanceInfo domInfo(castOp->getParentOp());
  for (LLVM::LoadOp load : loads) {
    OpBuilder loadBuilder(load);
    Value probe = llhd::ProbeOp::create(loadBuilder, load.getLoc(), sig);
    if (probe.getType() == load.getType()) {
      load.getResult().replaceAllUsesWith(probe);
      load.erase();
      continue;
    }
    SmallVector<Operation *, 4> loadUsers(
        llvm::to_vector(load.getResult().getUsers()));
    for (Operation *user : loadUsers) {
      auto castOut = dyn_cast<UnrealizedConversionCastOp>(user);
      if (!castOut || castOut.getInputs().size() != 1 ||
          castOut.getResults().size() != 1)
        return false;
      if (castOut.getResult(0).getType() != probe.getType())
        return false;
      castOut.getResult(0).replaceAllUsesWith(probe);
      castOut.erase();
    }
    load.erase();
  }

  for (LLVM::StoreOp store : stores) {
    OpBuilder storeBuilder(store);
    Value storedValue = store.getValue();
    if (storedValue.getType() != nestedType) {
      if (auto hwType = dyn_cast<hw::StructType>(nestedType)) {
        if (isa<LLVM::LLVMStructType>(storedValue.getType())) {
          storedValue = buildHWStructFromLLVM(storedValue, hwType, storeBuilder,
                                              store.getLoc(), {}, &domInfo,
                                              nullptr);
        }
      }
      if (storedValue && storedValue.getType() != nestedType &&
          isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(
              storedValue.getType())) {
        storedValue = buildHWAggregateFromLLVM(storedValue, nestedType,
                                               storeBuilder, store.getLoc(), {});
      }
    }
    if (!storedValue || storedValue.getType() != nestedType)
      return false;
    Value zeroTime = llhd::ConstantTimeOp::create(storeBuilder, store.getLoc(),
                                                  0, "ns", 0, 1);
    llhd::DriveOp::create(storeBuilder, store.getLoc(), sig, storedValue,
                          zeroTime, Value{});
    store.erase();
  }

  for (Operation *cast : llvm::reverse(ptrCasts))
    if (cast->use_empty())
      cast->erase();
  if (alloca.use_empty())
    alloca.erase();
  return true;
}

static bool rewriteLLVMAggregateCast(UnrealizedConversionCastOp castOp,
                                     DenseMap<Value, Value> &allocaSignals) {
  if (castOp.getInputs().size() != 1 || castOp.getResults().size() != 1)
    return false;
  Type hwType = castOp.getResult(0).getType();
  bool wantsStruct = isa<hw::StructType>(hwType);
  bool wantsArray = isa<hw::ArrayType>(hwType);
  if (!wantsStruct && !wantsArray)
    return false;
  Value input = castOp.getInputs().front();
  auto inputLoad = input.getDefiningOp<LLVM::LoadOp>();
  if (!isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(input.getType()))
    return false;

  Value llvmValue = unwrapTrivialLoad(input);
  if (auto cast = llvmValue.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast.getOperand(0).getType() == hwType) {
      castOp.getResult(0).replaceAllUsesWith(cast.getOperand(0));
      castOp.erase();
      return true;
    }
  }
  if (auto hwStructType = dyn_cast<hw::StructType>(hwType)) {
    if (Value hwValue = unwrapHWStructCast(llvmValue, hwStructType)) {
      castOp.getResult(0).replaceAllUsesWith(hwValue);
      castOp.erase();
      return true;
    }
  }

  OpBuilder builder(castOp);
  DominanceInfo domInfo(castOp->getParentOp());
  constexpr uint64_t kMaxAggregateRebuildBitWidth = 256;
  auto aggregateBitWidth = getHWAggregateBitWidth(hwType);
  bool allowAggregateRebuild =
      aggregateBitWidth && *aggregateBitWidth <= kMaxAggregateRebuildBitWidth;
  if (!allowAggregateRebuild && isInsertValueChainAggregate(llvmValue))
    allowAggregateRebuild = true;
  Value replacement;
  if (auto hwStructType = dyn_cast<hw::StructType>(hwType)) {
    replacement = buildHWStructFromLLVM(llvmValue, hwStructType, builder,
                                        castOp.getLoc(), {}, &domInfo,
                                        &allocaSignals);
  }
  if (!replacement && allowAggregateRebuild)
    replacement =
        buildHWAggregateFromLLVM(llvmValue, hwType, builder, castOp.getLoc(), {});
  auto tryLoadBackedRecovery = [&](LLVM::LoadOp load) -> Value {
    if (!load)
      return {};
    Value recovered = computeHWValueBeforeOp(load, load.getAddr(), hwType,
                                             domInfo, builder, castOp.getLoc(),
                                             &allocaSignals);
    if (!recovered)
      recovered =
          probeLLHDRefForPtr(load.getAddr(), hwType, builder, castOp.getLoc());
    return recovered;
  };

  if (!replacement)
    if (auto load = llvmValue.getDefiningOp<LLVM::LoadOp>())
      replacement = tryLoadBackedRecovery(load);

  if (!replacement)
    if (inputLoad && inputLoad != llvmValue.getDefiningOp<LLVM::LoadOp>())
      replacement = tryLoadBackedRecovery(inputLoad);

  if (!replacement)
    return false;
  if (replacement.getType() != hwType)
    return false;
  castOp.getResult(0).replaceAllUsesWith(replacement);
  castOp.erase();
  return true;
}

void LowerLECLLVMPass::runOnOperation() {
  auto module = getOperation();
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<LLVM::AllocaOp, 8> allocas;
    module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
    DominanceInfo domInfo(module);
    for (LLVM::AllocaOp alloca : allocas)
      changed |= foldSingleBlockAlloca(alloca);
    for (LLVM::AllocaOp alloca : allocas) {
      if (!alloca || !alloca->getParentOp())
        continue;
      changed |= forwardSingleStoreAlloca(alloca, domInfo);
    }
    for (LLVM::AllocaOp alloca : allocas) {
      if (!alloca || !alloca->getParentOp())
        continue;
      changed |= replaceUninitializedAllocaLoadsWithUndef(alloca);
    }
  }
  bool erasedAllocaAfterRef = true;
  while (erasedAllocaAfterRef) {
    erasedAllocaAfterRef = false;
    SmallVector<LLVM::AllocaOp, 8> allocas;
    module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
    for (LLVM::AllocaOp alloca : allocas)
      erasedAllocaAfterRef |= eraseDeadAllocaStores(alloca);
  }

  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      if (region.empty())
        continue;
      Block &entry = region.front();
      for (auto alloca :
           llvm::make_early_inc_range(region.getOps<LLVM::AllocaOp>())) {
        if (alloca->getBlock() == &entry)
          continue;
        bool canHoist = true;
        for (Value operand : alloca->getOperands()) {
          if (auto def = operand.getDefiningOp()) {
            if (def->getParentRegion() != &region)
              continue;
            if (def->getBlock() != &entry) {
              if (def->hasTrait<OpTrait::ConstantLike>())
                def->moveBefore(&entry, entry.begin());
              else {
                canHoist = false;
                break;
              }
            }
          } else if (auto arg = dyn_cast<BlockArgument>(operand)) {
            if (arg.getOwner()->getParent() != &region)
              continue;
            if (arg.getOwner() != &entry) {
              canHoist = false;
              break;
            }
          }
        }
        if (canHoist)
          alloca->moveBefore(&entry, entry.begin());
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
              Value loaded = LLVM::LoadOp::create(builder, branch.getLoc(),
                                                  loadType, ptrValue);
              branch->setOperand(operandIndex, loaded);
            }
          }
        }
      }

      SmallVector<PromotableAllocationOpInterface> allocators;
      region.walk([&](PromotableAllocationOpInterface allocator) {
        allocators.push_back(allocator);
      });
      if (allocators.empty())
        continue;
      OpBuilder builder(&entry, entry.begin());
      DataLayout dataLayout = DataLayout::closest(op);
      DominanceInfo dominance(op);
      (void)tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance);
    }
  });

  DenseMap<Value, Value> allocaSignals;
  SmallVector<UnrealizedConversionCastOp, 8> refCasts;
  module.walk([&](UnrealizedConversionCastOp op) {
    if (op.getInputs().size() == 1 && op.getResults().size() == 1 &&
        isa<LLVM::LLVMPointerType>(op.getInputs().front().getType()) &&
        isa<llhd::RefType>(op.getResult(0).getType()))
      refCasts.push_back(op);
  });
  for (auto castOp : refCasts) {
    if (!castOp->getParentOp())
      continue;
    rewriteAllocaBackedLLHDRef(castOp, allocaSignals);
  }
  bool erasedAlloca = true;
  while (erasedAlloca) {
    erasedAlloca = false;
    SmallVector<LLVM::AllocaOp, 8> allocas;
    module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
    for (LLVM::AllocaOp alloca : allocas)
      erasedAlloca |= eraseDeadAllocaStores(alloca);
  }

  SmallVector<UnrealizedConversionCastOp, 8> casts;
  module.walk([&](UnrealizedConversionCastOp op) {
    if (op.getInputs().size() == 1 && op.getResults().size() == 1 &&
        isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(
            op.getInputs().front().getType()) &&
        isa<hw::StructType, hw::ArrayType>(op.getResult(0).getType()))
      casts.push_back(op);
  });
  for (auto castOp : casts) {
    if (!rewriteLLVMAggregateCast(castOp, allocaSignals)) {
      castOp.emitError(
          "unsupported LLVM aggregate conversion in LEC; add lowering");
      signalPassFailure();
      return;
    }
  }

  // Lower leftover llvm.extractvalue ops from 4-state LLVM structs into
  // comb/hw operations so we can fully eliminate LLVM structs.
  SmallVector<LLVM::ExtractValueOp, 16> extracts;
  module.walk([&](LLVM::ExtractValueOp op) {
    auto llvmStructType =
        dyn_cast<LLVM::LLVMStructType>(op.getContainer().getType());
    if (!llvmStructType)
      return;
    if (!getFourStateStructForLLVM(llvmStructType))
      return;
    extracts.push_back(op);
  });
  if (!extracts.empty()) {
    DominanceInfo domInfo(module);
    for (auto extract : extracts) {
      auto llvmStructType =
          cast<LLVM::LLVMStructType>(extract.getContainer().getType());
      auto hwType = getFourStateStructForLLVM(llvmStructType);
      if (!hwType)
        continue;
      OpBuilder builder(extract);
      ArrayRef<int64_t> path = extract.getPosition();
      Value replacement = findInsertedValue(extract.getContainer(), hwType,
                                            path, &domInfo, builder,
                                            extract.getLoc(), &allocaSignals);
      auto defaultKind = getAggregateDefaultKind(extract.getContainer());
      if (!replacement && defaultKind != AggregateDefaultKind::none) {
        // When extracting from undef/zero containers, provide default values.
        // For undef 4-state structs: value=0, unknown=1 (all bits unknown).
        // For zero containers: value=0, unknown=0.
        auto resultType = dyn_cast<IntegerType>(extract.getType());
        if (resultType) {
          bool isUnknownField = path.size() == 1 && path[0] == 1;
          APInt value(resultType.getWidth(), 0);
          if (defaultKind == AggregateDefaultKind::undef && isUnknownField)
            value = APInt::getAllOnes(resultType.getWidth());
          replacement = hw::ConstantOp::create(
              builder, extract.getLoc(),
              builder.getIntegerAttr(resultType, value));
        }
      }
      if (!replacement)
        continue;
      extract.replaceAllUsesWith(replacement);
      extract.erase();
    }
  }
  bool droppedArgs = true;
  while (droppedArgs) {
    droppedArgs = false;
    module.walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        if (region.empty())
          continue;
        droppedArgs |= dropUnusedBlockArguments(region);
      }
    });
    if (!droppedArgs)
      continue;
    bool erasedAlloca = true;
    while (erasedAlloca) {
      erasedAlloca = false;
      SmallVector<LLVM::AllocaOp, 8> allocas;
      module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
      for (LLVM::AllocaOp alloca : allocas)
        erasedAlloca |= eraseDeadAllocaStores(alloca);
    }
  }

  bool erasedAllocaAfterStruct = true;
  while (erasedAllocaAfterStruct) {
    erasedAllocaAfterStruct = false;
    SmallVector<LLVM::AllocaOp, 8> allocas;
    module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
    for (LLVM::AllocaOp alloca : allocas)
      erasedAllocaAfterStruct |= eraseDeadAllocaStores(alloca);
  }

  auto runCleanup = [&]() {
    auto isCleanupDead = [](Operation *op) {
      if (!op || !op->use_empty())
        return false;
      if (isa<LLVM::UndefOp, LLVM::InsertValueOp, LLVM::AllocaOp,
              LLVM::LoadOp, LLVM::ConstantOp, LLVM::ZeroOp,
              UnrealizedConversionCastOp>(op))
        return true;
      if (op->getNumRegions() != 0)
        return false;
      if (!isMemoryEffectFree(op))
        return false;
      if (op->hasTrait<OpTrait::IsTerminator>())
        return false;
      return true;
    };

    SmallVector<Operation *, 1024> worklist;
    llvm::SmallDenseSet<Operation *, 1024> queued;
    auto enqueueIfDead = [&](Operation *op) {
      if (isCleanupDead(op) && queued.insert(op).second)
        worklist.push_back(op);
    };

    module.walk([&](Operation *op) { enqueueIfDead(op); });

    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      queued.erase(op);
      if (!isCleanupDead(op))
        continue;

      SmallVector<Operation *, 8> operandDefs;
      operandDefs.reserve(op->getNumOperands());
      for (Value operand : op->getOperands())
        if (auto *def = operand.getDefiningOp())
          operandDefs.push_back(def);

      op->erase();

      for (Operation *def : operandDefs)
        enqueueIfDead(def);
    }
  };

  runCleanup();

  // Fold immutable scalar llvm.global loads to constants, then drop dead LLVM
  // declarations/symbols so strict unsupported-op checks only trigger on real
  // residual lowering gaps.
  DenseMap<StringAttr, bool> globalHasStoreCache;
  SmallVector<LLVM::LoadOp, 16> globalLoads;
  module.walk([&](LLVM::LoadOp load) { globalLoads.push_back(load); });
  for (LLVM::LoadOp load : globalLoads) {
    if (!load || !load->getParentOp())
      continue;
    auto access = resolveGlobalLoadAccess(load.getAddr());
    if (!access)
      continue;
    StringAttr globalName = access->addrOf.getGlobalNameAttr().getAttr();
    auto global = module.lookupSymbol<LLVM::GlobalOp>(globalName.getValue());
    if (!global)
      continue;

    bool isImmutable = static_cast<bool>(global.getConstant());
    if (!isImmutable)
      isImmutable = !hasAnyStoreToGlobal(module, globalName, globalHasStoreCache);
    if (!isImmutable)
      continue;

    auto typedAttr =
        extractGlobalScalarLoadConstant(global, load.getType(), access->geps);
    if (!typedAttr)
      continue;
    auto intAttr = dyn_cast<IntegerAttr>(*typedAttr);
    if (!intAttr)
      continue;

    OpBuilder builder(load);
    auto constOp = hw::ConstantOp::create(builder, load.getLoc(), intAttr);
    load.replaceAllUsesWith(constOp.getResult());
    Value loadAddr = load.getAddr();
    load.erase();

    eraseUnusedAddressChain(loadAddr);
  }
  // Lower llvm.intr.ctpop by extracting each bit, zero-extending it to the
  // destination width, and accumulating the sum.
  SmallVector<LLVM::CtPopOp, 16> ctPopOps;
  module.walk([&](LLVM::CtPopOp op) { ctPopOps.push_back(op); });
  for (LLVM::CtPopOp op : ctPopOps) {
    if (!op || !op->getParentOp())
      continue;
    auto intType = dyn_cast<IntegerType>(op.getType());
    if (!intType)
      continue;
    unsigned bitWidth = intType.getWidth();
    if (bitWidth == 0)
      continue;

    OpBuilder builder(op);
    auto zero = hw::ConstantOp::create(
        builder, op.getLoc(),
        builder.getIntegerAttr(intType, APInt(bitWidth, 0)));
    Value result = zero.getResult();

    for (unsigned i = 0; i < bitWidth; ++i) {
      auto bit = comb::ExtractOp::create(builder, op.getLoc(), op.getIn(), i, 1)
                     .getResult();
      Value extended = bit;
      if (bitWidth > 1) {
        auto upperZeros = hw::ConstantOp::create(
            builder, op.getLoc(), builder.getIntegerType(bitWidth - 1), 0);
        extended = comb::ConcatOp::create(builder, op.getLoc(),
                                          upperZeros.getResult(), bit)
                       .getResult();
      }
      result = comb::AddOp::create(builder, op.getLoc(), result, extended)
                   .getResult();
    }

    op.replaceAllUsesWith(result);
    op.erase();
  }

  // Moore random runtime hooks are simulation-only; abstract them to a fixed
  // deterministic value for formal lowering.
  SmallVector<LLVM::CallOp, 8> randomCalls;
  module.walk([&](LLVM::CallOp call) {
    auto callee = call.getCalleeAttr();
    if (!callee)
      return;
    StringRef name = callee.getValue();
    if (name == "__moore_urandom_range" || name == "__moore_urandom")
      randomCalls.push_back(call);
  });
  for (LLVM::CallOp call : randomCalls) {
    if (!call || !call->getParentOp())
      continue;
    if (call.getNumResults() != 1)
      continue;
    auto result = call.getResult();
    auto intType = dyn_cast<IntegerType>(result.getType());
    if (!intType)
      continue;
    OpBuilder builder(call);
    auto zero = hw::ConstantOp::create(
        builder, call.getLoc(), builder.getIntegerAttr(intType, 0));
    result.replaceAllUsesWith(zero);
    call.erase();
  }
  runCleanup();

  SmallVector<LLVM::GlobalOp, 8> deadGlobals;
  module.walk([&](LLVM::GlobalOp global) {
    if (SymbolTable::symbolKnownUseEmpty(global, module))
      deadGlobals.push_back(global);
  });
  for (auto global : deadGlobals)
    global.erase();

  SmallVector<LLVM::LLVMFuncOp, 8> deadDecls;
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal() && SymbolTable::symbolKnownUseEmpty(func, module))
      deadDecls.push_back(func);
  });
  for (auto decl : deadDecls)
    decl.erase();

  bool erasedAllocaFinal = true;
  while (erasedAllocaFinal) {
    erasedAllocaFinal = false;
    SmallVector<LLVM::AllocaOp, 8> allocas;
    module.walk([&](LLVM::AllocaOp op) { allocas.push_back(op); });
    for (LLVM::AllocaOp alloca : allocas)
      erasedAllocaFinal |= eraseDeadAllocaStores(alloca);
  }

  runCleanup();

  Operation *firstLLVM = nullptr;
  bool hasLLHD = false;
  module.walk([&](Operation *op) {
    if (op->getDialect() && op->getDialect()->getNamespace() == "llvm") {
      if (!firstLLVM)
        firstLLVM = op;
    }
    if (isa<llhd::LLHDDialect>(op->getDialect()))
      hasLLHD = true;
  });
  if (firstLLVM && !hasLLHD) {
    llvm::SmallString<256> opBuffer;
    llvm::raw_svector_ostream opStream(opBuffer);
    firstLLVM->print(opStream);
    firstLLVM->emitError() << "LEC LLVM lowering left unsupported LLVM operation: "
                           << firstLLVM->getName().getStringRef() << " :: "
                           << opStream.str();
    signalPassFailure();
  }
}

} // namespace
