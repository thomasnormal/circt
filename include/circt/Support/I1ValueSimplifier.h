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
#include "circt/Support/FourStateUtils.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
#include <string>

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

inline mlir::Value unwrapClockToI1(mlir::Value clock, bool &invert) {
  while (clock) {
    if (auto inv = clock.getDefiningOp<seq::ClockInverterOp>()) {
      invert = !invert;
      clock = inv.getInput();
      continue;
    }
    if (auto gate = clock.getDefiningOp<seq::ClockGateOp>()) {
      bool enableOn = false;
      if (auto literal = getConstI1Value(gate.getEnable()))
        enableOn |= *literal;
      if (auto testEnable = gate.getTestEnable()) {
        if (auto literal = getConstI1Value(testEnable))
          enableOn |= *literal;
      }
      if (!enableOn)
        return mlir::Value();
      clock = gate.getInput();
      continue;
    }
    if (auto mux = clock.getDefiningOp<seq::ClockMuxOp>()) {
      if (auto literal = getConstI1Value(mux.getCond())) {
        clock = *literal ? mux.getTrueClock() : mux.getFalseClock();
        continue;
      }
      if (mux.getTrueClock() == mux.getFalseClock()) {
        clock = mux.getTrueClock();
        continue;
      }
      return mlir::Value();
    }
    if (auto div = clock.getDefiningOp<seq::ClockDividerOp>()) {
      if (div.getPow2() == 0) {
        clock = div.getInput();
        continue;
      }
      return mlir::Value();
    }
    if (auto toClock = clock.getDefiningOp<seq::ToClockOp>())
      return toClock.getInput();
    return mlir::Value();
  }
  return mlir::Value();
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
      bool clockInvert = false;
      if (auto unwrapped =
              unwrapClockToI1(fromClock.getInput(), clockInvert)) {
        invert ^= clockInvert;
        value = unwrapped;
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
    if (auto icmpOp = value.getDefiningOp<comb::ICmpOp>()) {
      auto predicate = icmpOp.getPredicate();
      bool isEq = predicate == comb::ICmpPredicate::eq ||
                  predicate == comb::ICmpPredicate::ceq ||
                  predicate == comb::ICmpPredicate::weq;
      bool isNe = predicate == comb::ICmpPredicate::ne ||
                  predicate == comb::ICmpPredicate::cne ||
                  predicate == comb::ICmpPredicate::wne;
      if (!isEq && !isNe)
        break;
      std::optional<bool> literal;
      mlir::Value other;
      if ((literal = getConstI1Value(icmpOp.getLhs()))) {
        other = icmpOp.getRhs();
      } else if ((literal = getConstI1Value(icmpOp.getRhs()))) {
        other = icmpOp.getLhs();
      }
      if (!literal || !other)
        break;
      invert ^= isEq ? !*literal : *literal;
      value = other;
      continue;
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

inline std::optional<std::string> getI1ValueKey(mlir::Value value) {
  if (!value)
    return std::nullopt;
  auto simplified = simplifyI1Value(value);
  value = simplified.value;
  bool invert = simplified.invert;
  if (!value)
    return std::nullopt;

  if (auto literal = getConstI1Value(value)) {
    bool bit = *literal ^ invert;
    return std::string(bit ? "const1" : "const0");
  }

  mlir::BlockArgument root;
  if (traceI1ValueRoot(value, root) && root) {
    std::string key = ("arg" + llvm::Twine(root.getArgNumber())).str();
    if (invert)
      key.append(":inv");
    return key;
  }

  llvm::DenseMap<mlir::Value, llvm::hash_code> memo;
  llvm::SmallPtrSet<mlir::Value, 8> visiting;

  auto combineHashList = [&](llvm::StringRef tag,
                             llvm::ArrayRef<llvm::hash_code> hashes)
      -> llvm::hash_code {
    llvm::hash_code result = llvm::hash_value(tag);
    for (auto hash : hashes)
      result = llvm::hash_combine(result, hash);
    return result;
  };

  std::function<llvm::hash_code(mlir::Value)> hashValue =
      [&](mlir::Value v) -> llvm::hash_code {
    if (!v)
      return llvm::hash_code{};
    if (auto it = memo.find(v); it != memo.end())
      return it->second;
    if (visiting.contains(v))
      return llvm::hash_value(0x9e3779b97f4a7c15ULL);
    visiting.insert(v);

    llvm::hash_code result = llvm::hash_code{};
    if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
      result = llvm::hash_combine("arg", arg.getArgNumber());
    } else if (auto literal = getConstI1Value(v)) {
      result = llvm::hash_combine("const", *literal ? 1 : 0);
    } else if (auto *op = v.getDefiningOp()) {
      if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (cast->getNumOperands() == 1) {
          result = hashValue(cast->getOperand(0));
        }
      } else if (auto fromClock = mlir::dyn_cast<seq::FromClockOp>(op)) {
        result = hashValue(fromClock.getInput());
      } else if (auto toClock = mlir::dyn_cast<seq::ToClockOp>(op)) {
        result = hashValue(toClock.getInput());
      } else if (auto bitcast = mlir::dyn_cast<hw::BitcastOp>(op)) {
        result = hashValue(bitcast.getInput());
      } else if (auto extract = mlir::dyn_cast<hw::StructExtractOp>(op)) {
        auto field = extract.getFieldNameAttr();
        if (field)
          result = llvm::hash_combine("struct_extract", field.getValue(),
                                      hashValue(extract.getInput()));
        else
          result = llvm::hash_combine("struct_extract",
                                      hashValue(extract.getInput()));
      } else if (auto extract = mlir::dyn_cast<comb::ExtractOp>(op)) {
        result = llvm::hash_combine(
            "extract", extract.getLowBit(), hashValue(extract.getInput()));
      } else if (auto andOp = mlir::dyn_cast<comb::AndOp>(op)) {
        llvm::SmallVector<llvm::hash_code> hashes;
        std::function<void(mlir::Value)> collect = [&](mlir::Value operand) {
          if (auto *def = operand.getDefiningOp()) {
            if (def->getName() == op->getName()) {
              for (auto nested : def->getOperands())
                collect(nested);
              return;
            }
          }
          hashes.push_back(hashValue(operand));
        };
        for (auto operand : andOp.getOperands())
          collect(operand);
        llvm::sort(hashes, [](llvm::hash_code a, llvm::hash_code b) {
          return static_cast<size_t>(a) < static_cast<size_t>(b);
        });
        result = combineHashList("and", hashes);
      } else if (auto orOp = mlir::dyn_cast<comb::OrOp>(op)) {
        llvm::SmallVector<llvm::hash_code> hashes;
        std::function<void(mlir::Value)> collect = [&](mlir::Value operand) {
          if (auto *def = operand.getDefiningOp()) {
            if (def->getName() == op->getName()) {
              for (auto nested : def->getOperands())
                collect(nested);
              return;
            }
          }
          hashes.push_back(hashValue(operand));
        };
        for (auto operand : orOp.getOperands())
          collect(operand);
        llvm::sort(hashes, [](llvm::hash_code a, llvm::hash_code b) {
          return static_cast<size_t>(a) < static_cast<size_t>(b);
        });
        result = combineHashList("or", hashes);
      } else if (auto xorOp = mlir::dyn_cast<comb::XorOp>(op)) {
        llvm::SmallVector<llvm::hash_code> hashes;
        std::function<void(mlir::Value)> collect = [&](mlir::Value operand) {
          if (auto *def = operand.getDefiningOp()) {
            if (def->getName() == op->getName()) {
              for (auto nested : def->getOperands())
                collect(nested);
              return;
            }
          }
          hashes.push_back(hashValue(operand));
        };
        for (auto operand : xorOp.getOperands())
          collect(operand);
        llvm::sort(hashes, [](llvm::hash_code a, llvm::hash_code b) {
          return static_cast<size_t>(a) < static_cast<size_t>(b);
        });
        result = combineHashList("xor", hashes);
      } else if (auto icmpOp = mlir::dyn_cast<comb::ICmpOp>(op)) {
        auto lhsHash = hashValue(icmpOp.getLhs());
        auto rhsHash = hashValue(icmpOp.getRhs());
        bool commutative = false;
        switch (icmpOp.getPredicate()) {
        case comb::ICmpPredicate::eq:
        case comb::ICmpPredicate::ne:
        case comb::ICmpPredicate::ceq:
        case comb::ICmpPredicate::cne:
        case comb::ICmpPredicate::weq:
        case comb::ICmpPredicate::wne:
          commutative = true;
          break;
        default:
          break;
        }
        if (commutative &&
            static_cast<size_t>(rhsHash) < static_cast<size_t>(lhsHash))
          std::swap(lhsHash, rhsHash);
        result = llvm::hash_combine("icmp",
                                    static_cast<int>(icmpOp.getPredicate()),
                                    lhsHash, rhsHash);
      } else if (auto muxOp = mlir::dyn_cast<comb::MuxOp>(op)) {
        result = llvm::hash_combine(
            "mux", hashValue(muxOp.getCond()),
            hashValue(muxOp.getTrueValue()),
            hashValue(muxOp.getFalseValue()));
      } else {
        result = llvm::hash_combine(op->getName().getStringRef());
        for (auto operand : op->getOperands())
          result = llvm::hash_combine(result, hashValue(operand));
      }
    }

    visiting.erase(v);
    memo[v] = result;
    return result;
  };

  auto hash = hashValue(value);
  if (invert)
    hash = llvm::hash_combine(hash, 0x1u);
  auto hashValueInt = static_cast<uint64_t>(hash);
  std::string key = "expr:" + llvm::utohexstr(hashValueInt);
  if (invert)
    key.append(":inv");
  return key;
}

} // namespace circt

#endif // CIRCT_SUPPORT_I1VALUESIMPLIFIER_H
