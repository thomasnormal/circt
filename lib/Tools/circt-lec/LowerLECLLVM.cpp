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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_LOWERLECLLVM
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {
struct LowerLECLLVMPass
    : public circt::impl::LowerLECLLVMBase<LowerLECLLVMPass> {
  void runOnOperation() override;
};

static bool hasOnlyLoadStoreUsers(Value ptr,
                                  SmallVectorImpl<LLVM::StoreOp> &stores,
                                  SmallVectorImpl<LLVM::LoadOp> &loads) {
  for (Operation *user : ptr.getUsers()) {
    if (auto store = dyn_cast<LLVM::StoreOp>(user)) {
      stores.push_back(store);
      continue;
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(user)) {
      loads.push_back(load);
      continue;
    }
    return false;
  }
  return true;
}

static bool foldTrivialAlloca(LLVM::AllocaOp alloca) {
  Value ptr = alloca.getResult();
  SmallVector<LLVM::StoreOp, 2> stores;
  SmallVector<LLVM::LoadOp, 4> loads;
  if (!hasOnlyLoadStoreUsers(ptr, stores, loads))
    return false;
  if (stores.size() != 1)
    return false;
  LLVM::StoreOp store = stores.front();
  if (store->getBlock() != alloca->getBlock())
    return false;
  for (LLVM::LoadOp load : loads) {
    if (load->getBlock() != alloca->getBlock())
      return false;
    if (!store->isBeforeInBlock(load))
      return false;
  }

  Value storedValue = store.getValue();
  for (LLVM::LoadOp load : loads) {
    load.getResult().replaceAllUsesWith(storedValue);
    load.erase();
  }
  store.erase();
  if (ptr.use_empty())
    alloca.erase();
  return true;
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

static Value unwrapTrivialLoad(Value value) {
  auto load = value.getDefiningOp<LLVM::LoadOp>();
  if (!load)
    return value;
  Value ptr = load.getAddr();
  SmallVector<LLVM::StoreOp, 2> stores;
  SmallVector<LLVM::LoadOp, 4> loads;
  if (!hasOnlyLoadStoreUsers(ptr, stores, loads))
    return value;
  if (stores.size() != 1)
    return value;
  LLVM::StoreOp store = stores.front();
  if (store->getBlock() != load->getBlock())
    return value;
  if (!store->isBeforeInBlock(load))
    return value;
  return store.getValue();
}

static bool rewriteLLVMStructCast(UnrealizedConversionCastOp castOp) {
  if (castOp.getInputs().size() != 1 || castOp.getResults().size() != 1)
    return false;
  auto hwType = dyn_cast<hw::StructType>(castOp.getResult(0).getType());
  if (!hwType)
    return false;
  Value input = castOp.getInputs().front();
  if (!isa<LLVM::LLVMStructType>(input.getType()))
    return false;

  Value llvmValue = unwrapTrivialLoad(input);
  OpBuilder builder(castOp);
  Value replacement =
      buildHWStructFromLLVM(llvmValue, hwType, builder, castOp.getLoc(), {});
  if (!replacement)
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
    for (LLVM::AllocaOp alloca : allocas)
      changed |= foldTrivialAlloca(alloca);
  }

  SmallVector<UnrealizedConversionCastOp, 8> casts;
  module.walk([&](UnrealizedConversionCastOp op) {
    if (op.getInputs().size() == 1 && op.getResults().size() == 1 &&
        isa<LLVM::LLVMStructType>(op.getInputs().front().getType()) &&
        isa<hw::StructType>(op.getResult(0).getType()))
      casts.push_back(op);
  });
  for (auto castOp : casts) {
    if (!rewriteLLVMStructCast(castOp)) {
      castOp.emitError(
          "unsupported LLVM struct conversion in LEC; add lowering");
      signalPassFailure();
      return;
    }
  }

  bool erased = true;
  while (erased) {
    erased = false;
    SmallVector<Operation *, 16> eraseList;
    module.walk([&](Operation *op) {
      if (!op->use_empty())
        return;
      if (isa<LLVM::UndefOp, LLVM::InsertValueOp, LLVM::AllocaOp, LLVM::LoadOp,
              LLVM::ConstantOp>(op))
        eraseList.push_back(op);
    });
    if (!eraseList.empty())
      erased = true;
    for (Operation *op : eraseList)
      op->erase();
  }

  Operation *firstLLVM = nullptr;
  module.walk([&](Operation *op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() == "llvm" && !firstLLVM)
      firstLLVM = op;
  });
  if (firstLLVM) {
    firstLLVM->emitError(
        "LEC LLVM lowering left unsupported LLVM operations");
    signalPassFailure();
  }
}

} // namespace
