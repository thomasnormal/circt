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
#include "llvm/ADT/StringRef.h"
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

inline bool traceI1ValueRoot(mlir::Value value, mlir::BlockArgument &root) {
  if (!value)
    return false;
  if (auto fromClock = value.getDefiningOp<seq::FromClockOp>())
    return traceI1ValueRoot(fromClock.getInput(), root);
  if (auto toClock = value.getDefiningOp<seq::ToClockOp>())
    return traceI1ValueRoot(toClock.getInput(), root);
  if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
      return traceI1ValueRoot(cast->getOperand(0), root);
  }
  if (auto bitcast = value.getDefiningOp<hw::BitcastOp>())
    return traceI1ValueRoot(bitcast.getInput(), root);
  if (auto result = llvm::dyn_cast<mlir::OpResult>(value)) {
    if (auto explode = llvm::dyn_cast<hw::StructExplodeOp>(result.getOwner()))
      return traceI1ValueRoot(explode.getInput(), root);
  }
  if (auto extract = value.getDefiningOp<hw::StructExtractOp>())
    return traceI1ValueRoot(extract.getInput(), root);
  if (auto extractOp = value.getDefiningOp<comb::ExtractOp>())
    return traceI1ValueRoot(extractOp.getInput(), root);
  if (value.getDefiningOp<hw::ConstantOp>() ||
      value.getDefiningOp<mlir::arith::ConstantOp>())
    return true;
  if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    if (!root)
      root = arg;
    return arg == root;
  }
  if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
    for (auto operand : andOp.getOperands())
      if (!traceI1ValueRoot(operand, root))
        return false;
    return true;
  }
  if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
    for (auto operand : orOp.getOperands())
      if (!traceI1ValueRoot(operand, root))
        return false;
    return true;
  }
  if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
    for (auto operand : xorOp.getOperands())
      if (!traceI1ValueRoot(operand, root))
        return false;
    return true;
  }
  if (auto icmpOp = value.getDefiningOp<comb::ICmpOp>()) {
    mlir::Value other;
    if (getConstI1Value(icmpOp.getLhs()))
      other = icmpOp.getRhs();
    else if (getConstI1Value(icmpOp.getRhs()))
      other = icmpOp.getLhs();
    else
      return false;
    auto otherTy = llvm::dyn_cast<mlir::IntegerType>(other.getType());
    if (!otherTy || otherTy.getWidth() != 1)
      return false;
    switch (icmpOp.getPredicate()) {
    case comb::ICmpPredicate::eq:
    case comb::ICmpPredicate::ceq:
    case comb::ICmpPredicate::weq:
    case comb::ICmpPredicate::ne:
    case comb::ICmpPredicate::cne:
    case comb::ICmpPredicate::wne:
      return traceI1ValueRoot(other, root);
    default:
      break;
    }
    return false;
  }
  if (auto concatOp = value.getDefiningOp<comb::ConcatOp>()) {
    for (auto operand : concatOp.getOperands())
      if (!traceI1ValueRoot(operand, root))
        return false;
    return true;
  }
  return false;
}

inline bool isFourStateStructType(mlir::Type type) {
  auto structTy = llvm::dyn_cast<hw::StructType>(type);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (elements[0].name.getValue() != "value")
    return false;
  if (elements[1].name.getValue() != "unknown")
    return false;
  return true;
}

inline mlir::Value getConcatOperandForExtract(comb::ExtractOp extractOp) {
  auto concat = extractOp.getInput().getDefiningOp<comb::ConcatOp>();
  if (!concat)
    return mlir::Value();
  auto extractTy = llvm::dyn_cast<mlir::IntegerType>(extractOp.getType());
  if (!extractTy)
    return mlir::Value();
  unsigned extractWidth = extractTy.getWidth();
  unsigned lowBit = extractOp.getLowBit();
  unsigned offset = 0;
  auto operands = concat.getOperands();
  if (operands.empty())
    return mlir::Value();
  for (size_t index = operands.size(); index-- > 0;) {
    auto operand = operands[index];
    auto operandTy = llvm::dyn_cast<mlir::IntegerType>(operand.getType());
    if (!operandTy)
      return mlir::Value();
    unsigned width = operandTy.getWidth();
    if (lowBit == offset && extractWidth == width)
      return operand;
    offset += width;
  }
  return mlir::Value();
}

inline mlir::Value getStructFieldBase(mlir::Value val,
                                      llvm::StringRef fieldName) {
  if (auto extract = val.getDefiningOp<hw::StructExtractOp>()) {
    auto fieldAttr = extract.getFieldNameAttr();
    if (!fieldAttr || fieldAttr.getValue().empty()) {
      auto structTy =
          llvm::dyn_cast<hw::StructType>(extract.getInput().getType());
      if (structTy) {
        auto idx = extract.getFieldIndex();
        auto elements = structTy.getElements();
        if (idx < elements.size())
          fieldAttr = elements[idx].name;
      }
    }
    if (fieldAttr && fieldAttr.getValue() == fieldName &&
        isFourStateStructType(extract.getInput().getType()))
      return extract.getInput();
  }
  if (auto result = llvm::dyn_cast<mlir::OpResult>(val)) {
    if (auto explode = llvm::dyn_cast<hw::StructExplodeOp>(result.getOwner())) {
      auto structTy =
          llvm::dyn_cast<hw::StructType>(explode.getInput().getType());
      if (structTy && isFourStateStructType(structTy)) {
        auto idx = result.getResultNumber();
        auto elements = structTy.getElements();
        if (idx < elements.size() && elements[idx].name.getValue() == fieldName)
          return explode.getInput();
      }
    }
  }
  if (auto extractOp = val.getDefiningOp<comb::ExtractOp>()) {
    if (auto operand = getConcatOperandForExtract(extractOp))
      return getStructFieldBase(operand, fieldName);
  }
  return mlir::Value();
}

inline mlir::Value stripXorConstTrue(mlir::Value val) {
  auto xorOp = val.getDefiningOp<comb::XorOp>();
  if (!xorOp)
    return mlir::Value();
  mlir::Value nonConst;
  bool parity = false;
  for (mlir::Value operand : xorOp.getOperands()) {
    if (auto literal = getConstI1Value(operand)) {
      parity ^= *literal;
      continue;
    }
    if (nonConst)
      return mlir::Value();
    nonConst = operand;
  }
  if (!nonConst || !parity)
    return mlir::Value();
  return nonConst;
}

inline mlir::Value matchFourStateClockGate(mlir::Value lhs, mlir::Value rhs) {
  auto lhsBase = getStructFieldBase(lhs, "value");
  if (!lhsBase)
    return mlir::Value();
  auto rhsNotUnknown = stripXorConstTrue(rhs);
  if (!rhsNotUnknown)
    return mlir::Value();
  auto rhsBase = getStructFieldBase(rhsNotUnknown, "unknown");
  if (!rhsBase || rhsBase != lhsBase)
    return mlir::Value();
  return lhs;
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
      if (auto fieldAttr = extract.getFieldNameAttr()) {
        if ((fieldAttr.getValue() == "value" ||
             fieldAttr.getValue() == "unknown") &&
            isFourStateStructType(extract.getInput().getType()))
          break;
      }
      value = extract.getInput();
      continue;
    }
    if (auto extractOp = value.getDefiningOp<comb::ExtractOp>()) {
      if (getStructFieldBase(value, "value") ||
          getStructFieldBase(value, "unknown"))
        break;
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
      if (andOp.getNumOperands() == 2) {
        if (auto gated = matchFourStateClockGate(andOp.getOperand(0),
                                                 andOp.getOperand(1))) {
          value = gated;
          continue;
        }
        if (auto gated = matchFourStateClockGate(andOp.getOperand(1),
                                                 andOp.getOperand(0))) {
          value = gated;
          continue;
        }
      }
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
