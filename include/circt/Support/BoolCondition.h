//===- BoolCondition.h - Boolean condition helper ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BOOLCONDITION_H
#define CIRCT_SUPPORT_BOOLCONDITION_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/PointerIntPair.h"

namespace circt {

/// Tracks a boolean condition as either constant true/false or an SSA value.
class BoolCondition {
public:
  BoolCondition() {}
  BoolCondition(mlir::Value value) : pair(value, 0) {
    if (value) {
      if (mlir::matchPattern(value, mlir::m_One()))
        *this = BoolCondition(true);
      if (mlir::matchPattern(value, mlir::m_Zero()))
        *this = BoolCondition(false);
    }
  }
  BoolCondition(bool konst) : pair(nullptr, konst ? 1 : 2) {}

  explicit operator bool() const {
    return pair.getPointer() != nullptr || pair.getInt() != 0;
  }

  bool isTrue() const { return !pair.getPointer() && pair.getInt() == 1; }
  bool isFalse() const { return !pair.getPointer() && pair.getInt() == 2; }
  mlir::Value getValue() const { return pair.getPointer(); }

  mlir::Value materialize(mlir::OpBuilder &builder,
                          mlir::Location loc) const {
    if (isTrue())
      return circt::hw::ConstantOp::create(builder, loc, llvm::APInt(1, 1));
    if (isFalse())
      return circt::hw::ConstantOp::create(builder, loc, llvm::APInt(1, 0));
    return pair.getPointer();
  }

  BoolCondition orWith(BoolCondition other, mlir::OpBuilder &builder) const {
    if (isTrue() || other.isTrue())
      return true;
    if (isFalse())
      return other;
    if (other.isFalse())
      return *this;
    return builder.createOrFold<circt::comb::OrOp>(
        getValue().getLoc(), getValue(), other.getValue());
  }

  BoolCondition andWith(BoolCondition other, mlir::OpBuilder &builder) const {
    if (isFalse() || other.isFalse())
      return false;
    if (isTrue())
      return other;
    if (other.isTrue())
      return *this;
    return builder.createOrFold<circt::comb::AndOp>(
        getValue().getLoc(), getValue(), other.getValue());
  }

  BoolCondition inverted(mlir::OpBuilder &builder) const {
    if (isTrue())
      return false;
    if (isFalse())
      return true;
    return circt::comb::createOrFoldNot(getValue().getLoc(), getValue(),
                                        builder);
  }

private:
  llvm::PointerIntPair<mlir::Value, 2> pair;
};

} // namespace circt

#endif // CIRCT_SUPPORT_BOOLCONDITION_H
