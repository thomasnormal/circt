//===- CommutativeValueEquivalence.h - Value equivalence helper -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_COMMUTATIVEVALUEEQUIVALENCE_H
#define CIRCT_SUPPORT_COMMUTATIVEVALUEEQUIVALENCE_H

#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <utility>

namespace circt {

class CommutativeValueEquivalence {
public:
  bool isEquivalent(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == rhs)
      return true;
    if (!lhs || !rhs)
      return false;
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

    if (!mlir::isMemoryEffectFree(lhsOp) || !mlir::isMemoryEffectFree(rhsOp))
      return false;

    if (isCommutativeEquivalent(lhsOp, rhsOp)) {
      memo[key] = true;
      return true;
    }

    auto eq = mlir::OperationEquivalence::isEquivalentTo(
        lhsOp, rhsOp,
        [&](mlir::Value a, mlir::Value b) {
          return isEquivalent(a, b) ? mlir::success() : mlir::failure();
        },
        /*markEquivalent=*/nullptr,
        mlir::OperationEquivalence::IgnoreLocations |
            mlir::OperationEquivalence::IgnoreDiscardableAttrs);

    memo[key] = eq;
    return eq;
  }

private:
  bool isCommutativePredicate(circt::comb::ICmpPredicate pred) const {
    switch (pred) {
    case circt::comb::ICmpPredicate::eq:
    case circt::comb::ICmpPredicate::ne:
    case circt::comb::ICmpPredicate::ceq:
    case circt::comb::ICmpPredicate::cne:
    case circt::comb::ICmpPredicate::weq:
    case circt::comb::ICmpPredicate::wne:
      return true;
    default:
      return false;
    }
  }

  bool operandsEquivalentCommutative(mlir::ValueRange lhs,
                                     mlir::ValueRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    llvm::SmallVector<bool> matched(rhs.size(), false);
    for (mlir::Value lhsVal : lhs) {
      bool found = false;
      for (auto [idx, rhsVal] : llvm::enumerate(rhs)) {
        if (matched[idx])
          continue;
        if (!isEquivalent(lhsVal, rhsVal))
          continue;
        matched[idx] = true;
        found = true;
        break;
      }
      if (!found)
        return false;
    }
    return true;
  }

  void collectAssociativeOperands(mlir::Operation *op,
                                  llvm::SmallVector<mlir::Value> &out) {
    std::function<void(mlir::Value)> addOperand = [&](mlir::Value value) {
      if (auto *def = value.getDefiningOp()) {
        if (def->getName() == op->getName()) {
          for (mlir::Value nested : def->getOperands())
            addOperand(nested);
          return;
        }
      }
      out.push_back(value);
    };
    for (mlir::Value operand : op->getOperands())
      addOperand(operand);
  }

  bool operandsEquivalentAssociative(mlir::Operation *lhsOp,
                                     mlir::Operation *rhsOp) {
    llvm::SmallVector<mlir::Value> lhsOperands;
    llvm::SmallVector<mlir::Value> rhsOperands;
    collectAssociativeOperands(lhsOp, lhsOperands);
    collectAssociativeOperands(rhsOp, rhsOperands);
    return operandsEquivalentCommutative(lhsOperands, rhsOperands);
  }

  bool isCommutativeEquivalent(mlir::Operation *lhsOp,
                               mlir::Operation *rhsOp) {
    if (auto lhs = mlir::dyn_cast<circt::comb::AndOp>(lhsOp)) {
      auto rhs = mlir::dyn_cast<circt::comb::AndOp>(rhsOp);
      if (!rhs)
        return false;
      return operandsEquivalentAssociative(lhsOp, rhsOp);
    }
    if (auto lhs = mlir::dyn_cast<circt::comb::OrOp>(lhsOp)) {
      auto rhs = mlir::dyn_cast<circt::comb::OrOp>(rhsOp);
      if (!rhs)
        return false;
      return operandsEquivalentAssociative(lhsOp, rhsOp);
    }
    if (auto lhs = mlir::dyn_cast<circt::comb::XorOp>(lhsOp)) {
      auto rhs = mlir::dyn_cast<circt::comb::XorOp>(rhsOp);
      if (!rhs)
        return false;
      return operandsEquivalentAssociative(lhsOp, rhsOp);
    }
    if (auto lhs = mlir::dyn_cast<circt::comb::ICmpOp>(lhsOp)) {
      auto rhs = mlir::dyn_cast<circt::comb::ICmpOp>(rhsOp);
      if (!rhs)
        return false;
      if (lhs.getPredicate() != rhs.getPredicate())
        return false;
      if (!isCommutativePredicate(lhs.getPredicate()))
        return false;
      return operandsEquivalentCommutative(lhs.getOperands(),
                                            rhs.getOperands());
    }
    return false;
  }

  llvm::DenseMap<std::pair<mlir::Value, mlir::Value>, bool> memo;
};

} // namespace circt

#endif // CIRCT_SUPPORT_COMMUTATIVEVALUEEQUIVALENCE_H
