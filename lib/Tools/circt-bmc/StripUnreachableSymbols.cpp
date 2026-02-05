//===- StripUnreachableSymbols.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_STRIPUNREACHABLESYMBOLS
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct StripUnreachableSymbolsPass
    : public circt::impl::StripUnreachableSymbolsBase<
          StripUnreachableSymbolsPass> {
  using StripUnreachableSymbolsBase::StripUnreachableSymbolsBase;

  void runOnOperation() override {
    auto module = getOperation();
    SymbolTable symbolTable(module);
    llvm::SmallVector<Operation *, 8> worklist;
    llvm::SmallPtrSet<Operation *, 32> live;

    auto addSymbol = [&](Operation *sym) {
      if (!sym)
        return;
      if (live.insert(sym).second)
        worklist.push_back(sym);
    };

    if (!entrySymbol.empty())
      addSymbol(symbolTable.lookup(entrySymbol));
    addSymbol(symbolTable.lookup("main"));

    if (worklist.empty())
      return;

    if (!entrySymbol.empty()) {
      llvm::SmallVector<Operation *, 4> ctorsToErase;
      module.walk(
          [&](LLVM::GlobalCtorsOp op) { ctorsToErase.push_back(op); });
      module.walk(
          [&](LLVM::GlobalDtorsOp op) { ctorsToErase.push_back(op); });
      for (auto *op : ctorsToErase)
        op->erase();
    } else {
      module.walk([&](LLVM::GlobalCtorsOp ctorOp) {
        for (auto attr : ctorOp.getCtors()) {
          if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr))
            addSymbol(symbolTable.lookup(symRef.getValue()));
        }
        for (auto attr : ctorOp.getData()) {
          if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr))
            addSymbol(symbolTable.lookup(symRef.getValue()));
        }
      });
      module.walk([&](LLVM::GlobalDtorsOp dtorOp) {
        for (auto attr : dtorOp.getDtors()) {
          if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr))
            addSymbol(symbolTable.lookup(symRef.getValue()));
        }
        for (auto attr : dtorOp.getData()) {
          if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr))
            addSymbol(symbolTable.lookup(symRef.getValue()));
        }
      });
    }

    while (!worklist.empty()) {
      Operation *sym = worklist.pop_back_val();
      auto uses = SymbolTable::getSymbolUses(sym);
      if (!uses)
        return;
      for (const auto &use : *uses) {
        Operation *target =
            SymbolTable::lookupNearestSymbolFrom(use.getUser(),
                                                 use.getSymbolRef());
        addSymbol(target);
      }
    }

    llvm::SmallVector<Operation *, 8> toErase;
    for (Operation &op : module.getBody()->getOperations()) {
      if (isa<SymbolOpInterface>(op) && !live.contains(&op))
        toErase.push_back(&op);
    }
    for (auto *op : toErase)
      op->erase();
  }
};
} // namespace
