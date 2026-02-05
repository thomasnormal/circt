//===- DumpLECUnknowns.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dump backward slices for 4-state unknown outputs to aid LEC debugging.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-lec/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/FourStateUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_DUMPLECUNKNOWNS
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

namespace {

static bool isAllOnesConst(Value value) {
  if (auto constant = value.getDefiningOp<hw::ConstantOp>())
    return constant.getValue().isAllOnes();
  return false;
}

static bool isUnknownExtract(Value value) {
  if (auto extract = value.getDefiningOp<hw::StructExtractOp>())
    return extract.getFieldName() == "unknown";
  if (auto extract = value.getDefiningOp<comb::ExtractOp>())
    return isUnknownExtract(extract.getInput());
  return false;
}

static bool dependsOnInputUnknown(Value value,
                                  DenseMap<Value, bool> &cache,
                                  SmallPtrSetImpl<Value> &visiting) {
  if (!value)
    return false;
  if (auto it = cache.find(value); it != cache.end())
    return it->second;
  if (!visiting.insert(value).second)
    return false;

  bool depends = false;
  if (auto extract = value.getDefiningOp<hw::StructExtractOp>()) {
    if (extract.getFieldName() == "unknown" &&
        isa<BlockArgument>(extract.getInput())) {
      depends = true;
    }
  }

  if (!depends) {
    if (auto *defOp = value.getDefiningOp()) {
      for (Value operand : defOp->getOperands()) {
        if (dependsOnInputUnknown(operand, cache, visiting)) {
          depends = true;
          break;
        }
      }
    }
  }

  visiting.erase(value);
  cache[value] = depends;
  return depends;
}

struct DumpLECUnknownsPass
    : public circt::impl::DumpLECUnknownsBase<DumpLECUnknownsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
      for (auto outputOp : hwModule.getOps<hw::OutputOp>()) {
        for (auto [index, operand] : llvm::enumerate(outputOp.getOperands())) {
          if (!isFourStateStructType(operand.getType()))
            continue;
          auto structTy = cast<hw::StructType>(operand.getType());
          StringAttr unknownField = structTy.getElements()[1].name;
          OpBuilder builder(outputOp);
          auto extract = hw::StructExtractOp::create(
              builder, outputOp.getLoc(), operand, unknownField);

          llvm::SetVector<Operation *> slice;
          llvm::SmallVector<Value, 32> worklist;
          llvm::SmallPtrSet<Value, 32> seenValues;
          llvm::SmallPtrSet<Operation *, 32> seenOps;
          worklist.push_back(extract.getResult());
          seenValues.insert(extract.getResult());
          slice.insert(extract.getOperation());
          while (!worklist.empty()) {
            Value current = worklist.pop_back_val();
            auto *defOp = current.getDefiningOp();
            if (!defOp)
              continue;
            if (seenOps.insert(defOp).second)
              slice.insert(defOp);
            if (auto create = dyn_cast<hw::StructCreateOp>(defOp)) {
              auto structTy = dyn_cast<hw::StructType>(create.getType());
              if (structTy) {
                auto elements = structTy.getElements();
                for (size_t idx = 0; idx < elements.size(); ++idx) {
                  if (elements[idx].name &&
                      elements[idx].name.getValue() == "unknown") {
                    Value operand = create.getOperand(idx);
                    if (seenValues.insert(operand).second)
                      worklist.push_back(operand);
                    break;
                  }
                }
                continue;
              }
            }
            for (Value operand : defOp->getOperands())
              if (seenValues.insert(operand).second)
                worklist.push_back(operand);
          }

          llvm::errs() << "=== Unknown slice: @"
                       << hwModule.getName().str() << " output#"
                       << index << " ===\n";
          unsigned inputUnknownExtracts = 0;
          unsigned unknownAllOnesConsts = 0;
          unsigned unknownXorInversions = 0;
          unsigned inputUnknownInversions = 0;
          DenseMap<Value, bool> dependsCache;
          SmallPtrSet<Value, 16> visiting;
          for (Operation *op : slice) {
            llvm::errs() << "  " << op->getName().getStringRef();
            if (op->getNumOperands()) {
              llvm::errs() << " (";
              llvm::interleave(
                  op->getOperands(),
                  [&](Value operand) {
                    if (auto arg = dyn_cast<BlockArgument>(operand))
                      llvm::errs() << "arg" << arg.getArgNumber();
                    else if (auto *def = operand.getDefiningOp())
                      llvm::errs() << def->getName().getStringRef();
                    else
                      llvm::errs() << "unknown";
                  },
                  [&]() { llvm::errs() << ", "; });
              llvm::errs() << ")";
            }
            llvm::errs() << "\n";
            if (auto extractOp = dyn_cast<hw::StructExtractOp>(op)) {
              if (extractOp.getFieldName() == "unknown" &&
                  isa<BlockArgument>(extractOp.getInput())) {
                ++inputUnknownExtracts;
                auto arg = cast<BlockArgument>(extractOp.getInput());
                llvm::errs() << "    input-unknown arg" << arg.getArgNumber()
                             << " at " << extractOp.getLoc() << "\n";
              }
            }
            if (auto constant = dyn_cast<hw::ConstantOp>(op)) {
              if (constant.getValue().isAllOnes()) {
                ++unknownAllOnesConsts;
                unsigned width = constant.getValue().getBitWidth();
                llvm::errs() << "    const-all-ones width=" << width << " at "
                             << constant.getLoc() << "\n";
              }
            }
            if (auto xorOp = dyn_cast<comb::XorOp>(op)) {
              bool hasAllOnes = false;
              bool hasUnknown = false;
              bool hasInputUnknown = false;
              for (Value input : xorOp.getInputs()) {
                hasAllOnes |= isAllOnesConst(input);
                hasUnknown |= isUnknownExtract(input);
                hasInputUnknown |=
                    dependsOnInputUnknown(input, dependsCache, visiting);
              }
              if (hasAllOnes && hasUnknown) {
                ++unknownXorInversions;
                llvm::errs() << "    unknown-xor-inversion at "
                             << xorOp.getLoc() << "\n";
              }
              if (hasAllOnes && hasInputUnknown) {
                ++inputUnknownInversions;
                llvm::errs() << "    input-unknown-inversion at "
                             << xorOp.getLoc() << "\n";
              }
            }
          }
          llvm::errs() << "  summary: input-unknown-extracts="
                       << inputUnknownExtracts
                       << " const-unknown-ones=" << unknownAllOnesConsts
                       << " unknown-xor-inversions=" << unknownXorInversions
                       << " input-unknown-inversions=" << inputUnknownInversions
                       << "\n";
          llvm::errs() << "=== End unknown slice ===\n";

          extract.erase();
        }
      }
    }
  }
};

} // namespace
