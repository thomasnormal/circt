//===- I1ValueSimplifier.h - Simplify i1 values ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_I1VALUESIMPLIFIER_H
#define CIRCT_SUPPORT_I1VALUESIMPLIFIER_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace circt {

inline std::optional<bool> getConstI1Value(mlir::Value val) {
  if (auto cst = val.getDefiningOp<hw::ConstantOp>()) {
    if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(cst.getType());
        intTy && intTy.getWidth() == 1)
      return cst.getValue().isAllOnes();
  }
  if (auto cst = val.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(cst.getValue()))
      return boolAttr.getValue();
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
      if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(intAttr.getType());
          intTy && intTy.getWidth() == 1)
        return intAttr.getValue().isAllOnes();
    }
  }
  return std::nullopt;
}

struct SimplifiedI1Value {
  mlir::Value value;
  bool invert = false;
};

inline SimplifiedI1Value simplifyI1Value(mlir::Value value) {
  bool invert = false;
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast->getOperand(0);
        continue;
      }
    }
    if (auto fromClock = value.getDefiningOp<seq::FromClockOp>()) {
      if (auto toClock =
              fromClock.getInput().getDefiningOp<seq::ToClockOp>()) {
        value = toClock.getInput();
        continue;
      }
    }
    if (auto bitcast = value.getDefiningOp<hw::BitcastOp>()) {
      value = bitcast.getInput();
      continue;
    }
    if (auto extract = value.getDefiningOp<hw::StructExtractOp>()) {
      value = extract.getInput();
      continue;
    }
    if (auto extractOp = value.getDefiningOp<comb::ExtractOp>()) {
      value = extractOp.getInput();
      continue;
    }
    if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
      llvm::SmallVector<mlir::Value, 4> nonConst;
      bool constParity = false;
      for (mlir::Value operand : xorOp.getOperands()) {
        if (auto literal = getConstI1Value(operand)) {
          constParity ^= *literal;
          continue;
        }
        nonConst.push_back(operand);
      }
      if (nonConst.empty())
        return {mlir::Value(), invert};
      if (nonConst.size() == 1) {
        invert ^= constParity;
        value = nonConst.front();
        continue;
      }
    }
    if (auto muxOp = value.getDefiningOp<comb::MuxOp>()) {
      if (auto literal = getConstI1Value(muxOp.getCond())) {
        value = *literal ? muxOp.getTrueValue() : muxOp.getFalseValue();
        continue;
      }
      if (muxOp.getTrueValue() == muxOp.getFalseValue()) {
        value = muxOp.getTrueValue();
        continue;
      }
    }
    if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
      mlir::SmallVector<mlir::Value> nonConst;
      bool sawFalse = false;
      for (mlir::Value operand : andOp.getOperands()) {
        if (auto literal = getConstI1Value(operand)) {
          if (!*literal) {
            sawFalse = true;
            break;
          }
          continue;
        }
        nonConst.push_back(operand);
      }
      if (sawFalse || nonConst.empty())
        return {mlir::Value(), invert};
      if (nonConst.size() == 1) {
        value = nonConst.front();
        continue;
      }
    }
    if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
      mlir::SmallVector<mlir::Value> nonConst;
      bool sawTrue = false;
      for (mlir::Value operand : orOp.getOperands()) {
        if (auto literal = getConstI1Value(operand)) {
          if (*literal) {
            sawTrue = true;
            break;
          }
          continue;
        }
        nonConst.push_back(operand);
      }
      if (sawTrue || nonConst.empty())
        return {mlir::Value(), invert};
      if (nonConst.size() == 1) {
        value = nonConst.front();
        continue;
      }
    }
    if (auto icmpOp = value.getDefiningOp<comb::ICmpOp>()) {
      mlir::Value other;
      bool constVal = false;
      if (auto literal = getConstI1Value(icmpOp.getLhs())) {
        constVal = *literal;
        other = icmpOp.getRhs();
      } else if (auto literal = getConstI1Value(icmpOp.getRhs())) {
        constVal = *literal;
        other = icmpOp.getLhs();
      } else {
        break;
      }
      auto otherTy = llvm::dyn_cast<mlir::IntegerType>(other.getType());
      if (!otherTy || otherTy.getWidth() != 1)
        break;
      switch (icmpOp.getPredicate()) {
      case comb::ICmpPredicate::eq:
      case comb::ICmpPredicate::ceq:
      case comb::ICmpPredicate::weq:
        if (!constVal)
          invert = !invert;
        value = other;
        continue;
      case comb::ICmpPredicate::ne:
      case comb::ICmpPredicate::cne:
      case comb::ICmpPredicate::wne:
        if (constVal)
          invert = !invert;
        value = other;
        continue;
      default:
        break;
      }
    }
    break;
  }
  return {value, invert};
}

} // namespace circt

#endif // CIRCT_SUPPORT_I1VALUESIMPLIFIER_H
