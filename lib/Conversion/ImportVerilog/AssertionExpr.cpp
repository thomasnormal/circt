//===- AssertionExpr.cpp - Slang assertion expression conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "slang/ast/expressions/AssertionExpr.h"
#include "slang/ast/expressions/AssignmentExpressions.h"
#include "slang/ast/expressions/CallExpression.h"
#include "slang/ast/expressions/OperatorExpressions.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "ImportVerilogInternals.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "slang/ast/SystemSubroutine.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <variant>

using namespace circt;
using namespace ImportVerilog;

// NOLINTBEGIN(misc-no-recursion)
namespace {
constexpr const char kDisableIffAttr[] = "sva.disable_iff";
constexpr const char kWeakEventuallyAttr[] = "ltl.weak";
constexpr const char kExplicitClockingAttr[] = "sva.explicit_clocking";

static Value createUnknownOrZeroConstant(Context &context, Location loc,
                                         moore::IntType type) {
  auto &builder = context.builder;
  if (type.getDomain() == moore::Domain::TwoValued)
    return moore::ConstantOp::create(builder, loc, type, 0);
  auto width = type.getWidth();
  if (width == 0)
    return {};
  return moore::ConstantOp::create(
      builder, loc, type,
      FVInt(APInt(width, 0), APInt::getAllOnes(width)));
}

static moore::IntType getSampledSimpleBitVectorType(Context &context,
                                                    const slang::ast::Type &type) {
  Type loweredType = context.convertType(type);
  if (auto refType = dyn_cast<moore::RefType>(loweredType))
    loweredType = refType.getNestedType();
  if (auto intType = dyn_cast<moore::IntType>(loweredType))
    return intType;
  if (auto packedType = dyn_cast<moore::PackedType>(loweredType))
    return packedType.getSimpleBitVector();
  // In bit-vector sampled-value context, strings are converted through
  // moore.string_to_int (32-bit int) by convertToSimpleBitVector.
  if (isa<moore::StringType>(loweredType))
    return moore::IntType::getInt(context.getContext(), 32);
  return {};
}

static Value buildSampledStableComparison(Context &context, Location loc,
                                          Value lhs, Value rhs,
                                          StringRef funcName) {
  auto &builder = context.builder;
  if (!lhs || !rhs || lhs.getType() != rhs.getType()) {
    mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
    return {};
  }

  if (isa<moore::IntType>(lhs.getType()))
    return moore::EqOp::create(builder, loc, lhs, rhs).getResult();

  if (isa<moore::RealType>(lhs.getType()))
    return moore::EqRealOp::create(builder, loc, lhs, rhs).getResult();

  if (isa<moore::EventType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    Value lhsBool = moore::BoolCastOp::create(builder, loc, lhs).getResult();
    Value rhsBool = moore::BoolCastOp::create(builder, loc, rhs).getResult();
    if (lhsBool.getType() != i1Ty)
      lhsBool = context.materializeConversion(i1Ty, lhsBool, false, loc);
    if (rhsBool.getType() != i1Ty)
      rhsBool = context.materializeConversion(i1Ty, rhsBool, false, loc);
    if (!lhsBool || !rhsBool)
      return {};
    return moore::EqOp::create(builder, loc, lhsBool, rhsBool).getResult();
  }

  if (isa<moore::StringType>(lhs.getType()) ||
      isa<moore::FormatStringType>(lhs.getType())) {
    auto strTy = moore::StringType::get(context.getContext());
    lhs = context.materializeConversion(strTy, lhs, false, lhs.getLoc());
    rhs = context.materializeConversion(strTy, rhs, false, rhs.getLoc());
    if (!lhs || !rhs)
      return {};
    return moore::StringCmpOp::create(builder, loc, moore::StringCmpPredicate::eq,
                                      lhs, rhs)
        .getResult();
  }

  if (isa<moore::ChandleType>(lhs.getType())) {
    auto intTy =
        moore::IntType::get(context.getContext(), 64, moore::Domain::TwoValued);
    lhs = context.materializeConversion(intTy, lhs, false, lhs.getLoc());
    rhs = context.materializeConversion(intTy, rhs, false, rhs.getLoc());
    if (!lhs || !rhs)
      return {};
    return moore::EqOp::create(builder, loc, lhs, rhs).getResult();
  }

  if (isa<moore::UnpackedArrayType>(lhs.getType()))
    return moore::UArrayCmpOp::create(builder, loc, moore::UArrayCmpPredicate::eq,
                                      lhs, rhs);

  if (auto openArrayTy = dyn_cast<moore::OpenUnpackedArrayType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);

    Value lhsSize = moore::ArraySizeOp::create(builder, loc, lhs);
    Value rhsSize = moore::ArraySizeOp::create(builder, loc, rhs);
    Value sizeEq = moore::EqOp::create(builder, loc, lhsSize, rhsSize);
    if (sizeEq.getType() != i1Ty)
      sizeEq = context.materializeConversion(i1Ty, sizeEq, false, loc);
    if (!sizeEq)
      return {};

    auto mismatchQueueTy = moore::QueueType::get(openArrayTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, mismatchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, lhs);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(openArrayTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value lhsElem = bodyBlock->getArgument(0);
      Value idx = bodyBlock->getArgument(1);
      Value rhsElem = moore::DynExtractOp::create(builder, loc,
                                                  openArrayTy.getElementType(),
                                                  rhs, idx);
      Value elemEq =
          buildSampledStableComparison(context, loc, lhsElem, rhsElem, funcName);
      if (!elemEq)
        return {};
      if (elemEq.getType() != i1Ty)
        elemEq = context.materializeConversion(i1Ty, elemEq, false, loc);
      if (!elemEq)
        return {};
      Value mismatch = moore::NotOp::create(builder, loc, elemEq).getResult();
      moore::ArrayLocatorYieldOp::create(builder, loc, mismatch);
    }

    Value mismatchCount = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value noMismatch = moore::EqOp::create(builder, loc, mismatchCount, zero);
    if (noMismatch.getType() != i1Ty)
      noMismatch = context.materializeConversion(i1Ty, noMismatch, false, loc);
    if (!noMismatch)
      return {};
    return moore::AndOp::create(builder, loc, sizeEq, noMismatch).getResult();
  }

  if (auto queueTy = dyn_cast<moore::QueueType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);

    Value lhsSize = moore::ArraySizeOp::create(builder, loc, lhs);
    Value rhsSize = moore::ArraySizeOp::create(builder, loc, rhs);
    Value sizeEq = moore::EqOp::create(builder, loc, lhsSize, rhsSize);
    if (sizeEq.getType() != i1Ty)
      sizeEq = context.materializeConversion(i1Ty, sizeEq, false, loc);
    if (!sizeEq)
      return {};

    auto mismatchQueueTy = moore::QueueType::get(queueTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, mismatchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, lhs);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(queueTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value lhsElem = bodyBlock->getArgument(0);
      Value idx = bodyBlock->getArgument(1);
      Value rhsElem = moore::DynExtractOp::create(builder, loc,
                                                  queueTy.getElementType(),
                                                  rhs, idx);
      Value elemEq =
          buildSampledStableComparison(context, loc, lhsElem, rhsElem, funcName);
      if (!elemEq)
        return {};
      if (elemEq.getType() != i1Ty)
        elemEq = context.materializeConversion(i1Ty, elemEq, false, loc);
      if (!elemEq)
        return {};
      Value mismatch = moore::NotOp::create(builder, loc, elemEq).getResult();
      moore::ArrayLocatorYieldOp::create(builder, loc, mismatch);
    }

    Value mismatchCount = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value noMismatch = moore::EqOp::create(builder, loc, mismatchCount, zero);
    if (noMismatch.getType() != i1Ty)
      noMismatch = context.materializeConversion(i1Ty, noMismatch, false, loc);
    if (!noMismatch)
      return {};
    return moore::AndOp::create(builder, loc, sizeEq, noMismatch).getResult();
  }

  if (auto assocTy = dyn_cast<moore::AssocArrayType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    auto idxTy = moore::IntType::getInt(builder.getContext(), 32);

    Value lhsSize = moore::ArraySizeOp::create(builder, loc, lhs);
    Value rhsSize = moore::ArraySizeOp::create(builder, loc, rhs);
    Value sizeEq = moore::EqOp::create(builder, loc, lhsSize, rhsSize);
    if (sizeEq.getType() != i1Ty)
      sizeEq = context.materializeConversion(i1Ty, sizeEq, false, loc);
    if (!sizeEq)
      return {};

    auto buildProjection = [&](moore::QueueType resultTy, bool indexed,
                               Value array) -> Value {
      auto locator = moore::ArrayLocatorOp::create(
          builder, loc, resultTy, moore::LocatorMode::All, indexed, array);
      Block *bodyBlock = &locator.getBody().emplaceBlock();
      bodyBlock->addArgument(assocTy.getElementType(), loc);
      bodyBlock->addArgument(idxTy, loc);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(bodyBlock);
        Value one = moore::ConstantOp::create(builder, loc, i1Ty, 1);
        moore::ArrayLocatorYieldOp::create(builder, loc, one);
      }
      return locator.getResult();
    };

    auto valueQueueTy = moore::QueueType::get(assocTy.getElementType(), 0);
    Value lhsValues = buildProjection(valueQueueTy, /*indexed=*/false, lhs);
    Value rhsValues = buildProjection(valueQueueTy, /*indexed=*/false, rhs);
    Value valuesEq = buildSampledStableComparison(context, loc, lhsValues,
                                                  rhsValues, funcName);
    if (!valuesEq)
      return {};
    if (valuesEq.getType() != i1Ty)
      valuesEq = context.materializeConversion(i1Ty, valuesEq, false, loc);
    if (!valuesEq)
      return {};

    auto keyQueueTy = moore::QueueType::get(assocTy.getIndexType(), 0);
    Value lhsKeys = buildProjection(keyQueueTy, /*indexed=*/true, lhs);
    Value rhsKeys = buildProjection(keyQueueTy, /*indexed=*/true, rhs);
    Value keysEq =
        buildSampledStableComparison(context, loc, lhsKeys, rhsKeys, funcName);
    if (!keysEq)
      return {};
    if (keysEq.getType() != i1Ty)
      keysEq = context.materializeConversion(i1Ty, keysEq, false, loc);
    if (!keysEq)
      return {};

    Value assocEq = moore::AndOp::create(builder, loc, keysEq, valuesEq);
    return moore::AndOp::create(builder, loc, sizeEq, assocEq).getResult();
  }

  if (auto assocTy = dyn_cast<moore::WildcardAssocArrayType>(lhs.getType())) {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    auto buildProjection = [&](Value array) -> Value {
      auto queueTy = moore::QueueType::get(assocTy.getElementType(), 0);
      auto locator = moore::ArrayLocatorOp::create(
          builder, loc, queueTy, moore::LocatorMode::All, /*indexed=*/false,
          array);
      Block *bodyBlock = &locator.getBody().emplaceBlock();
      bodyBlock->addArgument(assocTy.getElementType(), loc);
      bodyBlock->addArgument(i32Ty, loc);
      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(bodyBlock);
        Value one = moore::ConstantOp::create(builder, loc,
                                              moore::IntType::getInt(builder.getContext(), 1), 1);
        moore::ArrayLocatorYieldOp::create(builder, loc, one);
      }
      return locator.getResult();
    };

    Value lhsValues = buildProjection(lhs);
    Value rhsValues = buildProjection(rhs);
    return buildSampledStableComparison(context, loc, lhsValues, rhsValues,
                                        funcName);
  }

  if (auto structTy = dyn_cast<moore::UnpackedStructType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    Value allEqual = moore::ConstantOp::create(builder, loc, i1Ty, 1);
    for (auto member : structTy.getMembers()) {
      auto lhsField = moore::StructExtractOp::create(builder, loc, member.type,
                                                     member.name, lhs);
      auto rhsField = moore::StructExtractOp::create(builder, loc, member.type,
                                                     member.name, rhs);
      auto fieldEq =
          buildSampledStableComparison(context, loc, lhsField, rhsField, funcName);
      if (!fieldEq)
        return {};
      if (fieldEq.getType() != i1Ty)
        fieldEq = context.materializeConversion(i1Ty, fieldEq, false, loc);
      if (!fieldEq)
        return {};
      allEqual = moore::AndOp::create(builder, loc, allEqual, fieldEq).getResult();
    }
    return allEqual;
  }

  if (auto unionTy = dyn_cast<moore::UnpackedUnionType>(lhs.getType())) {
    auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);
    Value allEqual = moore::ConstantOp::create(builder, loc, i1Ty, 1);
    for (auto member : unionTy.getMembers()) {
      auto lhsField = moore::UnionExtractOp::create(builder, loc, member.type,
                                                    member.name, lhs);
      auto rhsField = moore::UnionExtractOp::create(builder, loc, member.type,
                                                    member.name, rhs);
      auto fieldEq =
          buildSampledStableComparison(context, loc, lhsField, rhsField, funcName);
      if (!fieldEq)
        return {};
      if (fieldEq.getType() != i1Ty)
        fieldEq = context.materializeConversion(i1Ty, fieldEq, false, loc);
      if (!fieldEq)
        return {};
      allEqual = moore::AndOp::create(builder, loc, allEqual, fieldEq).getResult();
    }
    return allEqual;
  }

  mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
  return {};
}

/// Build a 1-bit sampled boolean from a sampled operand. For unpacked
/// aggregates, recursively OR-reduce member/element boolean values.
static Value buildSampledBoolean(Context &context, Location loc, Value value,
                                 StringRef funcName) {
  auto &builder = context.builder;
  if (!value) {
    mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
    return {};
  }

  auto i1Ty = moore::IntType::getInt(builder.getContext(), 1);

  if (isa<moore::IntType>(value.getType())) {
    Value boolVal = moore::BoolCastOp::create(builder, loc, value).getResult();
    if (boolVal.getType() != i1Ty)
      boolVal = context.materializeConversion(i1Ty, boolVal, false, loc);
    return boolVal;
  }

  if (auto realTy = dyn_cast<moore::RealType>(value.getType())) {
    auto zeroAttr = realTy.getWidth() == moore::RealWidth::f32
                        ? builder.getF32FloatAttr(0.0)
                        : builder.getF64FloatAttr(0.0);
    Value zero = moore::ConstantRealOp::create(builder, loc, zeroAttr);
    Value nonZero = moore::NeRealOp::create(builder, loc, value, zero);
    if (nonZero.getType() != i1Ty)
      nonZero = context.materializeConversion(i1Ty, nonZero, false, loc);
    return nonZero;
  }

  if (isa<moore::EventType>(value.getType())) {
    Value boolVal = moore::BoolCastOp::create(builder, loc, value).getResult();
    if (boolVal.getType() != i1Ty)
      boolVal = context.materializeConversion(i1Ty, boolVal, false, loc);
    return boolVal;
  }

  if (auto arrayTy = dyn_cast<moore::UnpackedArrayType>(value.getType())) {
    Value anySet = moore::ConstantOp::create(builder, loc, i1Ty, 0);
    auto idxTy = moore::IntType::get(builder.getContext(), 32,
                                     moore::Domain::TwoValued);
    for (unsigned i = 0; i < arrayTy.getSize(); ++i) {
      auto idx = moore::ConstantOp::create(builder, loc, idxTy,
                                           static_cast<int64_t>(i));
      auto elem = moore::DynExtractOp::create(builder, loc, arrayTy.getElementType(),
                                              value, idx);
      auto elemBool = buildSampledBoolean(context, loc, elem, funcName);
      if (!elemBool)
        return {};
      if (elemBool.getType() != i1Ty)
        elemBool = context.materializeConversion(i1Ty, elemBool, false, loc);
      if (!elemBool)
        return {};
      anySet = moore::OrOp::create(builder, loc, anySet, elemBool).getResult();
    }
    return anySet;
  }

  if (auto arrayTy = dyn_cast<moore::OpenUnpackedArrayType>(value.getType())) {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    auto matchQueueTy = moore::QueueType::get(arrayTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, matchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, value);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(arrayTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value elem = bodyBlock->getArgument(0);
      Value elemBool = buildSampledBoolean(context, loc, elem, funcName);
      if (!elemBool)
        return {};
      if (elemBool.getType() != i1Ty)
        elemBool = context.materializeConversion(i1Ty, elemBool, false, loc);
      if (!elemBool)
        return {};
      moore::ArrayLocatorYieldOp::create(builder, loc, elemBool);
    }

    Value count = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value anySet = moore::NeOp::create(builder, loc, count, zero);
    if (anySet.getType() != i1Ty)
      anySet = context.materializeConversion(i1Ty, anySet, false, loc);
    return anySet;
  }

  if (auto queueTy = dyn_cast<moore::QueueType>(value.getType())) {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    auto matchQueueTy = moore::QueueType::get(queueTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, matchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, value);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(queueTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value elem = bodyBlock->getArgument(0);
      Value elemBool = buildSampledBoolean(context, loc, elem, funcName);
      if (!elemBool)
        return {};
      if (elemBool.getType() != i1Ty)
        elemBool = context.materializeConversion(i1Ty, elemBool, false, loc);
      if (!elemBool)
        return {};
      moore::ArrayLocatorYieldOp::create(builder, loc, elemBool);
    }

    Value count = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value anySet = moore::NeOp::create(builder, loc, count, zero);
    if (anySet.getType() != i1Ty)
      anySet = context.materializeConversion(i1Ty, anySet, false, loc);
    return anySet;
  }

  if (auto assocTy = dyn_cast<moore::AssocArrayType>(value.getType())) {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    auto matchQueueTy = moore::QueueType::get(assocTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, matchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, value);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(assocTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value elem = bodyBlock->getArgument(0);
      Value elemBool = buildSampledBoolean(context, loc, elem, funcName);
      if (!elemBool)
        return {};
      if (elemBool.getType() != i1Ty)
        elemBool = context.materializeConversion(i1Ty, elemBool, false, loc);
      if (!elemBool)
        return {};
      moore::ArrayLocatorYieldOp::create(builder, loc, elemBool);
    }

    Value count = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value anySet = moore::NeOp::create(builder, loc, count, zero);
    if (anySet.getType() != i1Ty)
      anySet = context.materializeConversion(i1Ty, anySet, false, loc);
    return anySet;
  }

  if (auto assocTy = dyn_cast<moore::WildcardAssocArrayType>(value.getType())) {
    auto i32Ty = moore::IntType::getInt(builder.getContext(), 32);
    auto matchQueueTy = moore::QueueType::get(assocTy.getElementType(), 0);
    auto locator = moore::ArrayLocatorOp::create(
        builder, loc, matchQueueTy, moore::LocatorMode::All,
        /*indexed=*/false, value);

    Block *bodyBlock = &locator.getBody().emplaceBlock();
    bodyBlock->addArgument(assocTy.getElementType(), loc);
    bodyBlock->addArgument(i32Ty, loc);

    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(bodyBlock);
      Value elem = bodyBlock->getArgument(0);
      Value elemBool = buildSampledBoolean(context, loc, elem, funcName);
      if (!elemBool)
        return {};
      if (elemBool.getType() != i1Ty)
        elemBool = context.materializeConversion(i1Ty, elemBool, false, loc);
      if (!elemBool)
        return {};
      moore::ArrayLocatorYieldOp::create(builder, loc, elemBool);
    }

    Value count = moore::ArraySizeOp::create(builder, loc, locator);
    Value zero = moore::ConstantOp::create(builder, loc, i32Ty, 0);
    Value anySet = moore::NeOp::create(builder, loc, count, zero);
    if (anySet.getType() != i1Ty)
      anySet = context.materializeConversion(i1Ty, anySet, false, loc);
    return anySet;
  }

  if (auto structTy = dyn_cast<moore::UnpackedStructType>(value.getType())) {
    Value anySet = moore::ConstantOp::create(builder, loc, i1Ty, 0);
    for (auto member : structTy.getMembers()) {
      auto field = moore::StructExtractOp::create(builder, loc, member.type,
                                                  member.name, value);
      auto fieldBool = buildSampledBoolean(context, loc, field, funcName);
      if (!fieldBool)
        return {};
      if (fieldBool.getType() != i1Ty)
        fieldBool = context.materializeConversion(i1Ty, fieldBool, false, loc);
      if (!fieldBool)
        return {};
      anySet = moore::OrOp::create(builder, loc, anySet, fieldBool).getResult();
    }
    return anySet;
  }

  if (auto unionTy = dyn_cast<moore::UnpackedUnionType>(value.getType())) {
    Value anySet = moore::ConstantOp::create(builder, loc, i1Ty, 0);
    for (auto member : unionTy.getMembers()) {
      auto field = moore::UnionExtractOp::create(builder, loc, member.type,
                                                 member.name, value);
      auto fieldBool = buildSampledBoolean(context, loc, field, funcName);
      if (!fieldBool)
        return {};
      if (fieldBool.getType() != i1Ty)
        fieldBool = context.materializeConversion(i1Ty, fieldBool, false, loc);
      if (!fieldBool)
        return {};
      anySet = moore::OrOp::create(builder, loc, anySet, fieldBool).getResult();
    }
    return anySet;
  }

  if (isa<moore::StringType, moore::FormatStringType>(value.getType())) {
    Value boolVal = moore::BoolCastOp::create(builder, loc, value).getResult();
    if (boolVal.getType() != i1Ty)
      boolVal = context.materializeConversion(i1Ty, boolVal, false, loc);
    return boolVal;
  }

  auto bitvec = context.convertToSimpleBitVector(value);
  if (!bitvec || !isa<moore::IntType>(bitvec.getType())) {
    mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
    return {};
  }
  Value boolVal = moore::BoolCastOp::create(builder, loc, bitvec).getResult();
  if (boolVal.getType() != i1Ty)
    boolVal = context.materializeConversion(i1Ty, boolVal, false, loc);
  return boolVal;
}

static const slang::ast::SignalEventControl *
getCanonicalSignalEventControl(const slang::ast::TimingControl &ctrl) {
  if (auto *signal = ctrl.as_if<slang::ast::SignalEventControl>()) {
    auto *symRef = signal->expr.getSymbolReference();
    if (symRef && symRef->kind == slang::ast::SymbolKind::ClockingBlock) {
      auto &clockingBlock = symRef->as<slang::ast::ClockingBlockSymbol>();
      return getCanonicalSignalEventControl(clockingBlock.getEvent());
    }
    return signal;
  }
  if (auto *eventList = ctrl.as_if<slang::ast::EventListControl>()) {
    if (eventList->events.size() != 1)
      return nullptr;
    auto *event = *eventList->events.begin();
    if (!event)
      return nullptr;
    return getCanonicalSignalEventControl(*event);
  }
  return nullptr;
}

/// Get the currently active timescale as an integer number of femtoseconds.
static uint64_t getTimeScaleInFemtoseconds(Context &context) {
  static_assert(int(slang::TimeUnit::Seconds) == 0);
  static_assert(int(slang::TimeUnit::Milliseconds) == 1);
  static_assert(int(slang::TimeUnit::Microseconds) == 2);
  static_assert(int(slang::TimeUnit::Nanoseconds) == 3);
  static_assert(int(slang::TimeUnit::Picoseconds) == 4);
  static_assert(int(slang::TimeUnit::Femtoseconds) == 5);

  static_assert(int(slang::TimeScaleMagnitude::One) == 1);
  static_assert(int(slang::TimeScaleMagnitude::Ten) == 10);
  static_assert(int(slang::TimeScaleMagnitude::Hundred) == 100);

  auto exp = static_cast<unsigned>(context.timeScale.base.unit);
  assert(exp <= 5);
  exp = 5 - exp;
  auto scale = static_cast<uint64_t>(context.timeScale.base.magnitude);
  while (exp-- > 0)
    scale *= 1000;
  return scale;
}

static bool isEquivalentTimingControl(const slang::ast::TimingControl &lhs,
                                      const slang::ast::TimingControl &rhs) {
  if (lhs.isEquivalentTo(rhs))
    return true;
  auto *lhsSignal = getCanonicalSignalEventControl(lhs);
  auto *rhsSignal = getCanonicalSignalEventControl(rhs);
  if (!lhsSignal || !rhsSignal)
    return false;
  if (lhsSignal->edge != rhsSignal->edge)
    return false;
  if (!lhsSignal->expr.isEquivalentTo(rhsSignal->expr)) {
    auto *lhsSym = lhsSignal->expr.getSymbolReference();
    auto *rhsSym = rhsSignal->expr.getSymbolReference();
    if (!(lhsSym && rhsSym && lhsSym == rhsSym))
      return false;
  }
  if ((lhsSignal->iffCondition == nullptr) != (rhsSignal->iffCondition == nullptr))
    return false;
  if (lhsSignal->iffCondition &&
      !lhsSignal->iffCondition->isEquivalentTo(*rhsSignal->iffCondition)) {
    auto *lhsIffSym = lhsSignal->iffCondition->getSymbolReference();
    auto *rhsIffSym = rhsSignal->iffCondition->getSymbolReference();
    if (!(lhsIffSym && rhsIffSym && lhsIffSym == rhsIffSym))
      return false;
  }
  return true;
}

static bool containsExplicitClocking(Value value) {
  if (!value)
    return false;
  SmallVector<Operation *, 16> worklist;
  llvm::DenseSet<Operation *> visited;
  if (auto *root = value.getDefiningOp())
    worklist.push_back(root);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!op || !visited.insert(op).second)
      continue;
    if (isa<ltl::ClockOp>(op))
      return true;
    for (Value operand : op->getOperands()) {
      if (auto *def = operand.getDefiningOp())
        worklist.push_back(def);
    }
  }
  return false;
}

struct SequenceLengthBounds {
  uint64_t min = 0;
  std::optional<uint64_t> max;
};

static std::optional<SequenceLengthBounds>
getSequenceLengthBounds(Value seq) {
  if (!seq)
    return std::nullopt;
  if (!isa<ltl::SequenceType>(seq.getType()))
    return SequenceLengthBounds{1, 1};

  if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
    auto inputBounds = getSequenceLengthBounds(delayOp.getInput());
    if (!inputBounds)
      return std::nullopt;
    uint64_t minDelay = delayOp.getDelay();
    std::optional<uint64_t> maxDelay;
    if (auto length = delayOp.getLength())
      maxDelay = minDelay + *length;
    SequenceLengthBounds result;
    result.min = inputBounds->min + minDelay;
    if (inputBounds->max && maxDelay)
      result.max = *inputBounds->max + *maxDelay;
    return result;
  }
  if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>())
    return getSequenceLengthBounds(clockOp.getInput());
  if (auto pastOp = seq.getDefiningOp<ltl::PastOp>())
    return getSequenceLengthBounds(pastOp.getInput());
  if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
    SequenceLengthBounds result;
    result.min = 0;
    result.max = 0;
    for (auto input : concatOp.getInputs()) {
      auto bounds = getSequenceLengthBounds(input);
      if (!bounds)
        return std::nullopt;
      result.min += bounds->min;
      if (result.max && bounds->max)
        *result.max += *bounds->max;
      else
        result.max.reset();
    }
    return result;
  }
  if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
    auto more = repeatOp.getMore();
    auto bounds = getSequenceLengthBounds(repeatOp.getInput());
    if (!bounds)
      return std::nullopt;
    SequenceLengthBounds result;
    result.min = bounds->min * repeatOp.getBase();
    if (bounds->max && more)
      result.max = *bounds->max * (repeatOp.getBase() + *more);
    return result;
  }
  if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
    SequenceLengthBounds result;
    bool initialized = false;
    for (auto input : orOp.getInputs()) {
      auto bounds = getSequenceLengthBounds(input);
      if (!bounds)
        return std::nullopt;
      if (!initialized) {
        result = *bounds;
        initialized = true;
        continue;
      }
      result.min = std::min(result.min, bounds->min);
      if (result.max && bounds->max) {
        result.max = std::max(*result.max, *bounds->max);
      } else if (!result.max || !bounds->max) {
        // Any unbounded operand makes the OR unbounded.
        result.max.reset();
      }
    }
    return result;
  }
  if (auto andOp = seq.getDefiningOp<ltl::AndOp>()) {
    SequenceLengthBounds result;
    bool initialized = false;
    bool sawUnbounded = false;
    for (auto input : andOp.getInputs()) {
      auto bounds = getSequenceLengthBounds(input);
      if (!bounds)
        return std::nullopt;
      if (!initialized) {
        result = *bounds;
        initialized = true;
        if (!bounds->max)
          sawUnbounded = true;
        continue;
      }
      result.min = std::max(result.min, bounds->min);
      if (!bounds->max)
        sawUnbounded = true;
      if (!sawUnbounded && result.max && bounds->max)
        result.max = std::min(*result.max, *bounds->max);
    }
    if (sawUnbounded)
      result.max.reset();
    return result;
  }
  if (auto intersectOp = seq.getDefiningOp<ltl::IntersectOp>()) {
    SequenceLengthBounds result;
    bool initialized = false;
    bool sawUnbounded = false;
    for (auto input : intersectOp.getInputs()) {
      auto bounds = getSequenceLengthBounds(input);
      if (!bounds)
        return std::nullopt;
      if (!initialized) {
        result = *bounds;
        initialized = true;
        if (!bounds->max)
          sawUnbounded = true;
        continue;
      }
      result.min = std::max(result.min, bounds->min);
      if (!bounds->max)
        sawUnbounded = true;
      if (!sawUnbounded && result.max && bounds->max)
        result.max = std::min(*result.max, *bounds->max);
    }
    if (sawUnbounded)
      result.max.reset();
    return result;
  }
  if (auto gotoRepeatOp = seq.getDefiningOp<ltl::GoToRepeatOp>()) {
    auto bounds = getSequenceLengthBounds(gotoRepeatOp.getInput());
    if (!bounds)
      return std::nullopt;
    SequenceLengthBounds result;
    result.min = bounds->min * gotoRepeatOp.getBase();
    // Non-consecutive repetition can insert unbounded gaps between matches.
    result.max.reset();
    return result;
  }
  if (auto nonConsecutiveRepeatOp =
          seq.getDefiningOp<ltl::NonConsecutiveRepeatOp>()) {
    auto bounds = getSequenceLengthBounds(nonConsecutiveRepeatOp.getInput());
    if (!bounds)
      return std::nullopt;
    SequenceLengthBounds result;
    result.min = bounds->min * nonConsecutiveRepeatOp.getBase();
    // Non-consecutive repetition can insert unbounded gaps between matches.
    result.max.reset();
    return result;
  }
  if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
    return getSequenceLengthBounds(firstMatch.getInput());

  return std::nullopt;
}

static Value lowerSampledValueFunctionWithSamplingControl(
    Context &context, const slang::ast::Expression &valueExpr,
    const slang::ast::TimingControl *timingCtrl, StringRef funcName,
    const slang::ast::Expression *enableExpr,
    std::span<const slang::ast::Expression *const> disableExprs, Location loc) {
  auto &builder = context.builder;
  auto *insertionBlock = builder.getInsertionBlock();
  if (!insertionBlock)
    return {};
  auto *parentOp = insertionBlock->getParentOp();
  moore::SVModuleOp module;
  if (parentOp) {
    if (auto direct = dyn_cast<moore::SVModuleOp>(parentOp))
      module = direct;
    else
      module = parentOp->getParentOfType<moore::SVModuleOp>();
  }
  if (!module) {
    mlir::emitWarning(loc)
        << funcName
        << " sampled-value controls are only supported within a module; "
           "returning 0 as a placeholder";
    auto resultType =
        moore::IntType::get(builder.getContext(), 1, moore::Domain::FourValued);
    return moore::ConstantOp::create(builder, loc, resultType, 0);
  }

  bool isRose = funcName == "$rose";
  bool isFell = funcName == "$fell";
  bool isStable = funcName == "$stable";
  bool isChanged = funcName == "$changed";
  Type loweredType = context.convertType(*valueExpr.type);
  if (auto refType = dyn_cast<moore::RefType>(loweredType))
    loweredType = refType.getNestedType();
  bool isUnpackedAggregateStableType =
      isa<moore::UnpackedArrayType, moore::OpenUnpackedArrayType,
          moore::QueueType, moore::AssocArrayType,
          moore::WildcardAssocArrayType, moore::UnpackedStructType,
          moore::UnpackedUnionType>(loweredType);
  bool isUnpackedAggregateEdgeType =
      isa<moore::UnpackedArrayType, moore::OpenUnpackedArrayType,
          moore::QueueType, moore::AssocArrayType,
          moore::WildcardAssocArrayType, moore::UnpackedStructType,
          moore::UnpackedUnionType>(loweredType);
  bool isUnpackedAggregateStableSample =
      (isStable || isChanged) && isUnpackedAggregateStableType;
  bool isUnpackedAggregateEdgeSample =
      (isRose || isFell) && isUnpackedAggregateEdgeType;
  bool isRealStableSample =
      (isStable || isChanged) && isa<moore::RealType>(loweredType);
  bool isStringStableSample =
      (isStable || isChanged) &&
      isa<moore::StringType, moore::FormatStringType>(loweredType);
  bool isEventStableSample =
      (isStable || isChanged) && isa<moore::EventType>(loweredType);
  bool isRealEdgeSample =
      (isRose || isFell) && isa<moore::RealType>(loweredType);
  bool isStringEdgeSample =
      (isRose || isFell) &&
      isa<moore::StringType, moore::FormatStringType>(loweredType);
  bool isEventEdgeSample =
      (isRose || isFell) && isa<moore::EventType>(loweredType);
  auto intType = getSampledSimpleBitVectorType(context, *valueExpr.type);
  if (!isUnpackedAggregateStableSample && !isUnpackedAggregateEdgeSample &&
      !isRealStableSample && !isStringStableSample && !isEventStableSample &&
      !isRealEdgeSample && !isStringEdgeSample && !isEventEdgeSample &&
      !intType) {
    mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
    return {};
  }

  bool boolCast = isRose || isFell;
  moore::IntType sampleType;
  moore::UnpackedType sampledStorageType;
  moore::IntType resultType;
  if (isUnpackedAggregateStableSample || isRealStableSample ||
      isStringStableSample) {
    sampledStorageType = cast<moore::UnpackedType>(loweredType);
    resultType = moore::IntType::getInt(builder.getContext(), 1);
  } else if (isEventStableSample) {
    sampleType = moore::IntType::getInt(builder.getContext(), 1);
    sampledStorageType = sampleType;
    resultType = sampleType;
  } else if (isUnpackedAggregateEdgeSample || isRealEdgeSample ||
             isStringEdgeSample || isEventEdgeSample) {
    sampleType = moore::IntType::getInt(builder.getContext(), 1);
    sampledStorageType = sampleType;
    resultType = sampleType;
  } else {
    sampleType = boolCast
                     ? moore::IntType::get(builder.getContext(), 1,
                                           intType.getDomain())
                     : intType;
    sampledStorageType = sampleType;
    resultType =
        moore::IntType::get(builder.getContext(), 1, sampleType.getDomain());
  }

  Value prevVar;
  Value resultVar;
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *moduleBlock = module.getBody();
    if (moduleBlock->mightHaveTerminator()) {
      if (auto *terminator = moduleBlock->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(moduleBlock);
    } else {
      builder.setInsertionPointToEnd(moduleBlock);
    }

    Value prevInit;
    if (!isUnpackedAggregateStableSample && !isRealStableSample &&
        !isStringStableSample) {
      prevInit = createUnknownOrZeroConstant(context, loc, sampleType);
      if (!prevInit)
        return {};
    }
    Value resultInit = moore::ConstantOp::create(builder, loc, resultType, 0);

    prevVar = moore::VariableOp::create(builder, loc,
                                        moore::RefType::get(sampledStorageType),
                                        StringAttr{}, prevInit);
    resultVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(resultType), StringAttr{},
        resultInit);

    auto proc =
        moore::ProcedureOp::create(builder, loc, moore::ProcedureKind::Always);
    builder.setInsertionPointToEnd(&proc.getBody().emplaceBlock());
    if (timingCtrl && failed(context.convertTimingControl(*timingCtrl)))
      return {};

    Value current = context.convertRvalueExpression(valueExpr);
    if (!current)
      return {};
    if (!isUnpackedAggregateStableSample && !isUnpackedAggregateEdgeSample &&
        !isRealStableSample && !isStringStableSample && !isEventStableSample &&
        !isRealEdgeSample && !isStringEdgeSample && !isEventEdgeSample &&
        !isa<moore::IntType>(current.getType()))
      current = context.convertToSimpleBitVector(current);
    if (!current)
      return {};
    if (isUnpackedAggregateStableSample || isRealStableSample ||
        isStringStableSample) {
      if (!isa<moore::UnpackedType>(current.getType())) {
        mlir::emitError(loc)
            << "unsupported sampled value type for " << funcName;
        return {};
      }
      if (current.getType() != sampledStorageType) {
        mlir::emitError(loc)
            << "unsupported sampled value type for " << funcName;
        return {};
      }
    } else if (isEventStableSample) {
      if (!isa<moore::EventType>(current.getType()) ||
          current.getType() != loweredType) {
        mlir::emitError(loc)
            << "unsupported sampled value type for " << funcName;
        return {};
      }
      current = buildSampledBoolean(context, loc, current, funcName);
      if (!current)
        return {};
      if (current.getType() != sampleType)
        current = context.materializeConversion(sampleType, current,
                                                /*isSigned=*/false, loc);
      if (!current)
        return {};
    } else if (isUnpackedAggregateEdgeSample || isRealEdgeSample ||
               isStringEdgeSample || isEventEdgeSample) {
      if (!isa<moore::UnpackedType>(current.getType()) ||
          current.getType() != loweredType) {
        mlir::emitError(loc)
            << "unsupported sampled value type for " << funcName;
        return {};
      }
      current = buildSampledBoolean(context, loc, current, funcName);
      if (!current)
        return {};
      if (current.getType() != sampleType)
        current = context.materializeConversion(sampleType, current,
                                                /*isSigned=*/false, loc);
      if (!current)
        return {};
    } else {
      auto currentType = dyn_cast_or_null<moore::IntType>(current.getType());
      if (!currentType) {
        mlir::emitError(loc)
            << "unsupported sampled value type for " << funcName;
        return {};
      }
      if (boolCast)
        current = moore::BoolCastOp::create(builder, loc, current);
      if (current.getType() != sampleType)
        current = context.materializeConversion(sampleType, current,
                                                /*isSigned=*/false, loc);
    }

    Value enable;
    bool hasEnable = false;
    if (enableExpr) {
      enable = context.convertRvalueExpression(*enableExpr);
      if (!enable)
        return {};
      enable = context.convertToBool(enable);
      if (!enable)
        return {};
      hasEnable = true;
    }
    Value disable;
    bool hasDisable = false;
    for (auto *disableExpr : disableExprs) {
      if (!disableExpr)
        continue;
      Value term = context.convertRvalueExpression(*disableExpr);
      if (!term)
        return {};
      term = context.convertToBool(term);
      if (!term)
        return {};
      if (!hasDisable) {
        disable = term;
        hasDisable = true;
      } else {
        disable = moore::OrOp::create(builder, loc, disable, term);
      }
    }
    auto selectWithControl = [&](Value onUpdate, Value onHold,
                                 Value onReset) -> Value {
      if (!hasEnable && !hasDisable)
        return onUpdate;
      if (hasDisable) {
        auto conditional = moore::ConditionalOp::create(builder, loc,
                                                        onUpdate.getType(),
                                                        disable);
        auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
        auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(&trueBlock);
          moore::YieldOp::create(builder, loc, onReset);
          builder.setInsertionPointToStart(&falseBlock);
          if (hasEnable) {
            auto inner = moore::ConditionalOp::create(
                builder, loc, onUpdate.getType(), enable);
            auto &innerTrue = inner.getTrueRegion().emplaceBlock();
            auto &innerFalse = inner.getFalseRegion().emplaceBlock();
            {
              OpBuilder::InsertionGuard innerGuard(builder);
              builder.setInsertionPointToStart(&innerTrue);
              moore::YieldOp::create(builder, loc, onUpdate);
              builder.setInsertionPointToStart(&innerFalse);
              moore::YieldOp::create(builder, loc, onHold);
            }
            moore::YieldOp::create(builder, loc, inner.getResult());
          } else {
            moore::YieldOp::create(builder, loc, onUpdate);
          }
        }
        return conditional.getResult();
      }
      auto conditional = moore::ConditionalOp::create(
          builder, loc, onUpdate.getType(), enable);
      auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
      auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&trueBlock);
        moore::YieldOp::create(builder, loc, onUpdate);
        builder.setInsertionPointToStart(&falseBlock);
        moore::YieldOp::create(builder, loc, onHold);
      }
      return conditional.getResult();
    };

    Value prev = moore::ReadOp::create(builder, loc, prevVar);
    Value result;
    if (isStable || isChanged) {
      Value stable = buildSampledStableComparison(context, loc, current, prev,
                                                  funcName);
      if (!stable)
        return {};
      result = stable;
      if (isChanged)
        result = moore::NotOp::create(builder, loc, stable).getResult();
    } else {
      if (isRose) {
        auto notPrev = moore::NotOp::create(builder, loc, prev).getResult();
        result = moore::AndOp::create(builder, loc, current, notPrev)
                     .getResult();
      } else {
        auto notCurrent =
            moore::NotOp::create(builder, loc, current).getResult();
        result = moore::AndOp::create(builder, loc, notCurrent, prev)
                     .getResult();
      }
    }

    Value disabledValue;
    if (isUnpackedAggregateStableSample) {
      disabledValue = createUnknownOrZeroConstant(context, loc, resultType);
      if (!disabledValue)
        return {};
    } else {
      disabledValue = createUnknownOrZeroConstant(context, loc, resultType);
      if (!disabledValue)
        return {};
    }
    Value resultValue =
        selectWithControl(result, disabledValue, disabledValue);
    moore::BlockingAssignOp::create(builder, loc, resultVar, resultValue);
    Value resetPrev = prevInit ? prevInit : prev;
    Value nextPrev = selectWithControl(current, prev, resetPrev);
    moore::BlockingAssignOp::create(builder, loc, prevVar, nextPrev);
    moore::ReturnOp::create(builder, loc);
  }

  return moore::ReadOp::create(builder, loc, resultVar);
}

static Value lowerSampledValueFunctionWithClocking(
    Context &context, const slang::ast::Expression &valueExpr,
    const slang::ast::TimingControl &timingCtrl, StringRef funcName,
    const slang::ast::Expression *enableExpr,
    std::span<const slang::ast::Expression *const> disableExprs, Location loc) {
  return lowerSampledValueFunctionWithSamplingControl(
      context, valueExpr, &timingCtrl, funcName, enableExpr, disableExprs, loc);
}

static Value lowerPastWithSamplingControl(
    Context &context, const slang::ast::Expression &valueExpr,
    const slang::ast::TimingControl *timingCtrl, int64_t delay,
    const slang::ast::Expression *enableExpr,
    std::span<const slang::ast::Expression *const> disableExprs, Location loc) {
  auto &builder = context.builder;
  auto *insertionBlock = builder.getInsertionBlock();
  if (!insertionBlock)
    return {};
  auto *parentOp = insertionBlock->getParentOp();
  moore::SVModuleOp module;
  if (parentOp) {
    if (auto direct = dyn_cast<moore::SVModuleOp>(parentOp))
      module = direct;
    else
      module = parentOp->getParentOfType<moore::SVModuleOp>();
  }
  if (!module) {
    mlir::emitWarning(loc)
        << "$past sampled-value controls are only supported within a module; "
           "returning 0 as a placeholder";
    auto resultType =
        moore::IntType::get(builder.getContext(), 1, moore::Domain::FourValued);
    return moore::ConstantOp::create(builder, loc, resultType, 0);
  }
  if (delay < 0) {
    mlir::emitError(loc) << "$past delay must be non-negative";
    return {};
  }

  Type originalType = context.convertType(*valueExpr.type);
  if (auto refType = dyn_cast<moore::RefType>(originalType))
    originalType = refType.getNestedType();
  auto originalUnpacked = dyn_cast<moore::UnpackedType>(originalType);
  bool isUnpackedAggregateSample =
      isa<moore::UnpackedArrayType>(originalType) ||
      isa<moore::OpenUnpackedArrayType>(originalType) ||
      isa<moore::QueueType>(originalType) ||
      isa<moore::AssocArrayType>(originalType) ||
      isa<moore::WildcardAssocArrayType>(originalType) ||
      isa<moore::UnpackedStructType>(originalType) ||
      isa<moore::UnpackedUnionType>(originalType);
  bool isRealSample = isa<moore::RealType>(originalType);
  bool isStringSample =
      isa<moore::StringType>(originalType) ||
      isa<moore::FormatStringType>(originalType);
  bool isTimeSample = isa<moore::TimeType>(originalType);
  auto intType = getSampledSimpleBitVectorType(context, *valueExpr.type);
  if (!isUnpackedAggregateSample && !isRealSample && !isStringSample &&
      !isTimeSample &&
      !intType) {
    mlir::emitError(loc)
        << "unsupported $past value type with sampled-value controls (input "
           "type: "
        << originalType << ")";
    return {};
  }
  moore::UnpackedType storageType =
      (isUnpackedAggregateSample || isRealSample || isStringSample ||
       isTimeSample)
          ? cast<moore::UnpackedType>(originalType)
          : cast<moore::UnpackedType>(intType);

  int64_t historyDepth = std::max<int64_t>(delay, 1);

  SmallVector<Value, 4> historyVars;
  Value resultVar;
  {
    OpBuilder::InsertionGuard guard(builder);
    auto *moduleBlock = module.getBody();
    if (moduleBlock->mightHaveTerminator()) {
      if (auto *terminator = moduleBlock->getTerminator())
        builder.setInsertionPoint(terminator);
      else
        builder.setInsertionPointToEnd(moduleBlock);
    } else {
      builder.setInsertionPointToEnd(moduleBlock);
    }

    Value init;
    if (!isUnpackedAggregateSample && !isRealSample && !isStringSample &&
        !isTimeSample) {
      init = createUnknownOrZeroConstant(context, loc, intType);
      if (!init)
        return {};
    }

    for (int64_t i = 0; i < historyDepth; ++i) {
      historyVars.push_back(moore::VariableOp::create(
          builder, loc, moore::RefType::get(storageType), StringAttr{}, init));
    }
    resultVar = moore::VariableOp::create(
        builder, loc, moore::RefType::get(storageType), StringAttr{}, init);

    auto proc =
        moore::ProcedureOp::create(builder, loc, moore::ProcedureKind::Always);
    builder.setInsertionPointToEnd(&proc.getBody().emplaceBlock());
    if (timingCtrl && failed(context.convertTimingControl(*timingCtrl)))
      return {};

    Value current = context.convertRvalueExpression(valueExpr);
    if (!current)
      return {};
    if (!isUnpackedAggregateSample && !isRealSample && !isStringSample &&
        !isTimeSample &&
        !isa<moore::IntType>(current.getType()))
      current = context.convertToSimpleBitVector(current);
    if (!current)
      return {};
    if (isUnpackedAggregateSample || isRealSample || isStringSample ||
        isTimeSample) {
      if (!isa<moore::UnpackedType>(current.getType()) ||
          current.getType() != storageType) {
        mlir::emitError(loc)
            << "unsupported $past value type with sampled-value controls "
               "(current type: "
            << current.getType() << ", storage type: " << storageType << ")";
        return {};
      }
    } else {
      auto currentType = dyn_cast_or_null<moore::IntType>(current.getType());
      if (!currentType) {
        mlir::emitError(loc)
            << "unsupported $past value type with sampled-value controls "
               "(current type: "
            << current.getType() << ")";
        return {};
      }
      if (current.getType() != intType)
        current = context.materializeConversion(intType, current,
                                                /*isSigned=*/false, loc);
    }

    Value enable;
    bool hasEnable = false;
    if (enableExpr) {
      enable = context.convertRvalueExpression(*enableExpr);
      if (!enable)
        return {};
      enable = context.convertToBool(enable);
      if (!enable)
        return {};
      hasEnable = true;
    }
    Value disable;
    bool hasDisable = false;
    for (auto *disableExpr : disableExprs) {
      if (!disableExpr)
        continue;
      Value term = context.convertRvalueExpression(*disableExpr);
      if (!term)
        return {};
      term = context.convertToBool(term);
      if (!term)
        return {};
      if (!hasDisable) {
        disable = term;
        hasDisable = true;
      } else {
        disable = moore::OrOp::create(builder, loc, disable, term);
      }
    }
    auto selectWithControl = [&](Value onUpdate, Value onHold,
                                 Value onReset) -> Value {
      if (!hasEnable && !hasDisable)
        return onUpdate;
      if (hasDisable) {
        auto conditional = moore::ConditionalOp::create(builder, loc,
                                                        onUpdate.getType(),
                                                        disable);
        auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
        auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
        {
          OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(&trueBlock);
          moore::YieldOp::create(builder, loc, onReset);
          builder.setInsertionPointToStart(&falseBlock);
          if (hasEnable) {
            auto inner = moore::ConditionalOp::create(
                builder, loc, onUpdate.getType(), enable);
            auto &innerTrue = inner.getTrueRegion().emplaceBlock();
            auto &innerFalse = inner.getFalseRegion().emplaceBlock();
            {
              OpBuilder::InsertionGuard innerGuard(builder);
              builder.setInsertionPointToStart(&innerTrue);
              moore::YieldOp::create(builder, loc, onUpdate);
              builder.setInsertionPointToStart(&innerFalse);
              moore::YieldOp::create(builder, loc, onHold);
            }
            moore::YieldOp::create(builder, loc, inner.getResult());
          } else {
            moore::YieldOp::create(builder, loc, onUpdate);
          }
        }
        return conditional.getResult();
      }
      auto conditional = moore::ConditionalOp::create(
          builder, loc, onUpdate.getType(), enable);
      auto &trueBlock = conditional.getTrueRegion().emplaceBlock();
      auto &falseBlock = conditional.getFalseRegion().emplaceBlock();
      {
        OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&trueBlock);
        moore::YieldOp::create(builder, loc, onUpdate);
        builder.setInsertionPointToStart(&falseBlock);
        moore::YieldOp::create(builder, loc, onHold);
      }
      return conditional.getResult();
    };

    Value pastValue = current;
    if (delay > 0)
      pastValue = moore::ReadOp::create(builder, loc, historyVars.back());
    Value disabledValue = pastValue;
    if (!isUnpackedAggregateSample && !isRealSample && !isStringSample &&
        !isTimeSample) {
      disabledValue = createUnknownOrZeroConstant(context, loc, intType);
      if (!disabledValue)
        return {};
    }
    Value resultValue =
        selectWithControl(pastValue, disabledValue, disabledValue);
    moore::BlockingAssignOp::create(builder, loc, resultVar, resultValue);

    auto selectHistoryUpdate = [&](Value onTrue, Value onHold) -> Value {
      Value reset = init ? init : onHold;
      return selectWithControl(onTrue, onHold, reset);
    };

    for (int64_t i = historyDepth - 1; i > 0; --i) {
      Value prev = moore::ReadOp::create(builder, loc, historyVars[i]);
      Value prevPrev = moore::ReadOp::create(builder, loc, historyVars[i - 1]);
      Value next = selectHistoryUpdate(prevPrev, prev);
      moore::BlockingAssignOp::create(builder, loc, historyVars[i], next);
    }
    Value prev0 = moore::ReadOp::create(builder, loc, historyVars[0]);
    Value next0 = selectHistoryUpdate(current, prev0);
    moore::BlockingAssignOp::create(builder, loc, historyVars[0], next0);
    moore::ReturnOp::create(builder, loc);
  }

  Value result = moore::ReadOp::create(builder, loc, resultVar);
  if (originalUnpacked && result.getType() != originalUnpacked)
    result = context.materializeConversion(originalUnpacked, result,
                                           /*isSigned=*/false, loc);
  return result;
}

static Value lowerPastWithClocking(Context &context,
                                   const slang::ast::Expression &valueExpr,
                                   const slang::ast::TimingControl &timingCtrl,
                                   int64_t delay,
                                   const slang::ast::Expression *enableExpr,
                                   std::span<const slang::ast::Expression *const>
                                       disableExprs,
                                   Location loc) {
  return lowerPastWithSamplingControl(context, valueExpr, &timingCtrl, delay,
                                      enableExpr, disableExprs, loc);
}

struct AssertionExprVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;

  AssertionExprVisitor(Context &context, Location loc)
      : context(context), loc(loc), builder(context.builder) {}

  /// Helper to convert a range (min, optional max) to MLIR integer attributes
  std::pair<mlir::IntegerAttr, mlir::IntegerAttr>
  convertRangeToAttrs(uint32_t min,
                      std::optional<uint32_t> max = std::nullopt) {
    auto minAttr = builder.getI64IntegerAttr(min);
    mlir::IntegerAttr rangeAttr;
    if (max.has_value()) {
      rangeAttr = builder.getI64IntegerAttr(max.value() - min);
    }
    return {minAttr, rangeAttr};
  }

  LogicalResult
  handleMatchItems(std::span<const slang::ast::Expression *const> matchItems) {
    for (auto *item : matchItems) {
      if (!item)
        continue;
      switch (item->kind) {
      case slang::ast::ExpressionKind::Assignment: {
        auto &assign = item->as<slang::ast::AssignmentExpression>();
        auto *sym = assign.left().getSymbolReference();
        auto *local =
            sym ? sym->as_if<slang::ast::LocalAssertionVarSymbol>() : nullptr;
        if (!local) {
          mlir::emitError(loc, "match item assignment must target a local "
                               "assertion variable");
          return failure();
        }
        Value rhs;
        if (assign.isCompound()) {
          auto *binding = context.lookupAssertionLocalVarBinding(local);
          if (!binding) {
            mlir::emitError(loc, "local assertion variable referenced before "
                                 "assignment");
            return failure();
          }
          auto offset = context.getAssertionSequenceOffset();
          if (offset < binding->offset) {
            mlir::emitError(loc, "local assertion variable referenced before "
                                 "assignment time");
            return failure();
          }
          Value lhs;
          if (offset == binding->offset) {
            lhs = binding->value;
          } else {
            if (!isa<moore::UnpackedType>(binding->value.getType())) {
              mlir::emitError(loc, "unsupported local assertion variable type");
              return failure();
            }
            lhs = moore::PastOp::create(builder, loc, binding->value,
                                        static_cast<int64_t>(offset -
                                                             binding->offset))
                      .getResult();
          }
          if (!lhs)
            return failure();
          auto lhsUnpacked = dyn_cast<moore::UnpackedType>(lhs.getType());
          if (!lhsUnpacked) {
            mlir::emitError(loc, "unsupported match item assignment type")
                << lhs.getType();
            return failure();
          }
          auto lhsRef = moore::VariableOp::create(
              builder, loc, moore::RefType::get(lhsUnpacked), StringAttr{}, lhs);
          context.lvalueStack.push_back(lhsRef);
          rhs = context.convertRvalueExpression(assign.right());
          context.lvalueStack.pop_back();
          if (!rhs)
            return failure();
        }
        if (!rhs) {
          rhs = context.convertRvalueExpression(assign.right());
          if (!rhs)
            return failure();
        }
        if (!isa<moore::UnpackedType>(rhs.getType())) {
          mlir::emitError(loc, "unsupported match item assignment type")
              << rhs.getType();
          return failure();
        }
        context.setAssertionLocalVarBinding(
            local, rhs, context.getAssertionSequenceOffset());
        break;
      }
      case slang::ast::ExpressionKind::UnaryOp: {
        auto &unary = item->as<slang::ast::UnaryExpression>();
        using slang::ast::UnaryOperator;
        bool isInc = false;
        switch (unary.op) {
        case UnaryOperator::Preincrement:
        case UnaryOperator::Postincrement:
          isInc = true;
          break;
        case UnaryOperator::Predecrement:
        case UnaryOperator::Postdecrement:
          break;
        default:
          mlir::emitError(loc, "unsupported match item unary operator");
          return failure();
        }
        auto *sym = unary.operand().getSymbolReference();
        auto *local =
            sym ? sym->as_if<slang::ast::LocalAssertionVarSymbol>() : nullptr;
        if (!local) {
          mlir::emitError(loc, "match item unary operator must target a local "
                               "assertion variable");
          return failure();
        }
        auto base = context.convertRvalueExpression(unary.operand());
        if (!base)
          return failure();
        auto intType = dyn_cast<moore::IntType>(base.getType());
        Value updated;
        if (intType) {
          auto one = moore::ConstantOp::create(builder, loc, intType, 1);
          updated =
              isInc ? moore::AddOp::create(builder, loc, base, one).getResult()
                    : moore::SubOp::create(builder, loc, base, one).getResult();
        } else if (isa<moore::TimeType>(base.getType())) {
          auto realTy =
              moore::RealType::get(context.getContext(), moore::RealWidth::f64);
          auto realBase =
              context.materializeConversion(realTy, base, /*isSigned=*/false,
                                            loc);
          if (!realBase)
            return failure();
          auto oneAttr = builder.getFloatAttr(
              builder.getF64Type(), getTimeScaleInFemtoseconds(context));
          auto one = moore::ConstantRealOp::create(builder, loc, oneAttr);
          auto realUpdated = isInc
                                 ? moore::AddRealOp::create(builder, loc, realBase,
                                                            one)
                                       .getResult()
                                 : moore::SubRealOp::create(builder, loc, realBase,
                                                            one)
                                       .getResult();
          updated = context.materializeConversion(
              moore::TimeType::get(context.getContext()), realUpdated,
              /*isSigned=*/false, loc);
          if (!updated)
            return failure();
        } else if (auto realType = dyn_cast<moore::RealType>(base.getType())) {
          FloatAttr oneAttr;
          if (realType.getWidth() == moore::RealWidth::f32)
            oneAttr = builder.getFloatAttr(builder.getF32Type(), 1.0);
          else
            oneAttr = builder.getFloatAttr(builder.getF64Type(), 1.0);
          auto one = moore::ConstantRealOp::create(builder, loc, oneAttr);
          updated = isInc
                        ? moore::AddRealOp::create(builder, loc, base, one)
                              .getResult()
                        : moore::SubRealOp::create(builder, loc, base, one)
                              .getResult();
        } else {
          mlir::emitError(loc,
                          "match item unary operator requires int, real, or time "
                          "local assertion variable");
          return failure();
        }
        context.setAssertionLocalVarBinding(
            local, updated, context.getAssertionSequenceOffset());
        break;
      }
      case slang::ast::ExpressionKind::Call: {
        auto &call = item->as<slang::ast::CallExpression>();
        if (auto *sysInfo =
                std::get_if<slang::ast::CallExpression::SystemCallInfo>(
                    &call.subroutine)) {
          StringRef name = sysInfo->subroutine->name;
          auto getFd = [&]() -> Value {
            auto args = call.arguments();
            if (args.empty()) {
              mlir::emitError(loc) << name << " requires a file descriptor";
              return {};
            }
            Value fd = context.convertRvalueExpression(*args[0]);
            if (!fd)
              return {};
            auto intTy = moore::IntType::getInt(builder.getContext(), 32);
            if (fd.getType() != intTy)
              fd = context.materializeConversion(intTy, fd, /*isSigned=*/false,
                                                 loc);
            return fd;
          };
          auto emitSeverity = [&](moore::Severity severity) {
            auto msg = moore::FormatLiteralOp::create(builder, loc, name.str());
            moore::SeverityBIOp::create(builder, loc, severity, msg);
          };
          if (name == "$info") {
            emitSeverity(moore::Severity::Info);
            break;
          }
          if (name == "$warning") {
            emitSeverity(moore::Severity::Warning);
            break;
          }
          if (name == "$error") {
            emitSeverity(moore::Severity::Error);
            break;
          }
          if (name == "$fatal") {
            emitSeverity(moore::Severity::Fatal);
            moore::FinishBIOp::create(builder, loc, 1);
            break;
          }
          if (name == "$monitoron") {
            moore::MonitorOnBIOp::create(builder, loc);
            break;
          }
          if (name == "$monitoroff") {
            moore::MonitorOffBIOp::create(builder, loc);
            break;
          }
          if (name == "$printtimescale") {
            moore::PrintTimescaleBIOp::create(builder, loc);
            break;
          }
          if (name == "$fflush") {
            Value fd = getFd();
            if (!fd)
              return failure();
            moore::FFlushBIOp::create(builder, loc, fd);
            break;
          }
          if (name == "$fclose") {
            Value fd = getFd();
            if (!fd)
              return failure();
            moore::FCloseBIOp::create(builder, loc, fd);
            break;
          }
          bool isFWriteLike = false;
          bool appendNewlineFWrite = false;
          StringRef fwriteSuffix = name;
          if (fwriteSuffix.consume_front("$fdisplay")) {
            isFWriteLike = true;
            appendNewlineFWrite = true;
          } else if (fwriteSuffix.consume_front("$fwrite")) {
            isFWriteLike = true;
          }
          if (isFWriteLike) {
            if (!fwriteSuffix.empty() && fwriteSuffix != "b" &&
                fwriteSuffix != "o" && fwriteSuffix != "h")
              isFWriteLike = false;
          }
          if (isFWriteLike) {
            Value fd = getFd();
            if (!fd)
              return failure();
            std::string marker = name.str();
            if (appendNewlineFWrite)
              marker.push_back('\n');
            auto msg = moore::FormatLiteralOp::create(builder, loc, marker);
            moore::FWriteBIOp::create(builder, loc, fd, msg);
            break;
          }
          bool isFStrobeLike = false;
          StringRef fstrobeSuffix = name;
          if (fstrobeSuffix.consume_front("$fstrobe"))
            isFStrobeLike = true;
          if (isFStrobeLike) {
            if (!fstrobeSuffix.empty() && fstrobeSuffix != "b" &&
                fstrobeSuffix != "o" && fstrobeSuffix != "h")
              isFStrobeLike = false;
          }
          if (isFStrobeLike) {
            Value fd = getFd();
            if (!fd)
              return failure();
            auto msg = moore::FormatLiteralOp::create(builder, loc, name.str());
            moore::FStrobeBIOp::create(builder, loc, fd, msg);
            break;
          }
          bool isFMonitorLike = false;
          StringRef fmonitorSuffix = name;
          if (fmonitorSuffix.consume_front("$fmonitor"))
            isFMonitorLike = true;
          if (isFMonitorLike) {
            if (!fmonitorSuffix.empty() && fmonitorSuffix != "b" &&
                fmonitorSuffix != "o" && fmonitorSuffix != "h")
              isFMonitorLike = false;
          }
          if (isFMonitorLike) {
            Value fd = getFd();
            if (!fd)
              return failure();
            auto msg = moore::FormatLiteralOp::create(builder, loc, name.str());
            moore::FMonitorBIOp::create(builder, loc, fd, msg);
            break;
          }
          bool isMonitorLike = false;
          StringRef monitorSuffix = name;
          if (monitorSuffix.consume_front("$monitor"))
            isMonitorLike = true;
          if (isMonitorLike) {
            if (monitorSuffix.empty() || monitorSuffix == "b" ||
                monitorSuffix == "o" || monitorSuffix == "h") {
              auto msg = moore::FormatLiteralOp::create(builder, loc, name.str());
              moore::MonitorBIOp::create(builder, loc, msg);
              break;
            }
          }
          bool isStrobeLike = false;
          StringRef strobeSuffix = name;
          if (strobeSuffix.consume_front("$strobe"))
            isStrobeLike = true;
          if (isStrobeLike) {
            if (strobeSuffix.empty() || strobeSuffix == "b" ||
                strobeSuffix == "o" || strobeSuffix == "h") {
              std::string marker = name.str();
              marker.push_back('\n');
              auto msg = moore::FormatLiteralOp::create(builder, loc, marker);
              moore::DisplayBIOp::create(builder, loc, msg);
              break;
            }
          }
          bool isDisplayLike = false;
          bool appendNewline = false;
          StringRef suffix = name;
          if (suffix.consume_front("$display")) {
            isDisplayLike = true;
            appendNewline = true;
          } else if (suffix.consume_front("$write")) {
            isDisplayLike = true;
          }
          if (isDisplayLike) {
            if (!suffix.empty() && suffix != "b" && suffix != "o" &&
                suffix != "h")
              isDisplayLike = false;
          }
          if (isDisplayLike) {
            std::string marker = name.str();
            if (appendNewline)
              marker.push_back('\n');
            auto msg = moore::FormatLiteralOp::create(builder, loc, marker);
            moore::DisplayBIOp::create(builder, loc, msg);
            break;
          }
          auto callLoc = context.convertLocation(call.sourceRange);
          mlir::emitRemark(callLoc)
              << "ignoring system subroutine `" << name
              << "` in assertion match items";
          break;
        }
        if (!context.convertRvalueExpression(call))
          return failure();
        break;
      }
      default:
        mlir::emitError(loc, "unsupported match item expression");
        return failure();
      }
    }
    return success();
  }

  /// Add repetition operation to a sequence
  Value createRepetition(Location loc,
                         const slang::ast::SequenceRepetition &repetition,
                         Value &inputSequence) {
    // Extract cycle range
    auto [minRepetitions, repetitionRange] =
        convertRangeToAttrs(repetition.range.min, repetition.range.max);

    using slang::ast::SequenceRepetition;

    switch (repetition.kind) {
    case SequenceRepetition::Consecutive:
      return ltl::RepeatOp::create(builder, loc, inputSequence, minRepetitions,
                                   repetitionRange);
    case SequenceRepetition::Nonconsecutive:
      return ltl::NonConsecutiveRepeatOp::create(
          builder, loc, inputSequence, minRepetitions, repetitionRange);
    case SequenceRepetition::GoTo:
      return ltl::GoToRepeatOp::create(builder, loc, inputSequence,
                                       minRepetitions, repetitionRange);
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::SimpleAssertionExpr &expr) {
    // Handle expression
    auto value = context.convertRvalueExpression(expr.expr);
    if (!value)
      return {};
    auto loc = context.convertLocation(expr.expr.sourceRange);
    auto valueType = value.getType();
    // For assertion instances the value is already the expected type, convert
    // boolean value
    if (!mlir::isa<ltl::SequenceType, ltl::PropertyType>(valueType)) {
      // Multi-bit values (e.g., packed structs used in boolean context) need
      // to be reduced to 1-bit via BoolCast (!=0) before converting to i1.
      value = context.convertToBool(value);
      value = context.convertToI1(value);
    }
    if (!value)
      return {};

    // Handle repetition
    // The optional repetition is empty, return the converted expression
    if (!expr.repetition.has_value()) {
      return value;
    }

    // There is a repetition, embed the expression into the kind of given
    // repetition
    return createRepetition(loc, expr.repetition.value(), value);
  }

  Value visit(const slang::ast::SequenceWithMatchExpr &expr) {
    auto value =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!value)
      return {};
    if (expr.repetition.has_value()) {
      value = createRepetition(loc, expr.repetition.value(), value);
      if (!value)
        return {};
    }
    if (failed(handleMatchItems(expr.matchItems)))
      return {};
    return value;
  }

  Value visit(const slang::ast::SequenceConcatExpr &expr) {
    // Create a sequence of delayed operations, combined with a concat operation
    assert(!expr.elements.empty());

    SmallVector<Value> sequenceElements;
    uint64_t savedOffset = context.getAssertionSequenceOffset();
    uint64_t currentOffset = savedOffset;

    for (auto it = expr.elements.begin(); it != expr.elements.end(); ++it) {
      const auto &concatElement = *it;

      // Adjust inter-element delays to account for concat's cycle alignment.
      // For ##N between elements, concat already advances one cycle, so
      // subtract one when possible to align with SVA timing. The first element
      // delay is relative to the sequence start and should not be adjusted.
      uint32_t minDelay = concatElement.delay.min;
      std::optional<uint32_t> maxDelay = concatElement.delay.max;
      uint32_t ltlMinDelay = minDelay;
      std::optional<uint32_t> ltlMaxDelay = maxDelay;
      if (it != expr.elements.begin() && ltlMinDelay > 0) {
        --ltlMinDelay;
        if (ltlMaxDelay.has_value() && ltlMaxDelay.value() > 0)
          --ltlMaxDelay.value();
      }
      // Sequence offsets track the effective cycle position of each element.
      // Concat always advances one cycle between elements, so ##0 still moves
      // by one in the lowered LTL; reflect that when computing local-var pasts.
      uint32_t offsetDelay = minDelay;
      if (it != expr.elements.begin() && offsetDelay == 0)
        offsetDelay = 1;
      currentOffset += offsetDelay;
      context.setAssertionSequenceOffset(currentOffset);

      Value sequenceValue =
          context.convertAssertionExpression(*concatElement.sequence, loc,
                                             /*applyDefaults=*/false);
      if (!sequenceValue)
        return {};

      Type valueType = sequenceValue.getType();
      // Sequence concatenation requires sequence types (i1 or !ltl.sequence).
      // Property types (from $rose, $fell, $changed, $stable) cannot be used
      // directly in sequence contexts.
      if (mlir::isa<ltl::PropertyType>(valueType)) {
        mlir::emitError(loc, "property type cannot be used in sequence "
                             "concatenation; consider restructuring the "
                             "assertion to use the property as a consequent");
        return {};
      }

      auto [delayMin, delayRange] =
          convertRangeToAttrs(ltlMinDelay, ltlMaxDelay);
      auto delayedSequence = ltl::DelayOp::create(builder, loc, sequenceValue,
                                                  delayMin, delayRange);
      sequenceElements.push_back(delayedSequence);
    }

    context.setAssertionSequenceOffset(savedOffset);
    return builder.createOrFold<ltl::ConcatOp>(loc, sequenceElements);
  }

  Value visit(const slang::ast::FirstMatchAssertionExpr &expr) {
    auto sequenceValue =
        context.convertAssertionExpression(expr.seq, loc, /*applyDefaults=*/false);
    if (!sequenceValue)
      return {};
    if (failed(handleMatchItems(expr.matchItems)))
      return {};
    return ltl::FirstMatchOp::create(builder, loc, sequenceValue);
  }

  Value visit(const slang::ast::UnaryAssertionExpr &expr) {
    auto value =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!value)
      return {};
    using slang::ast::UnaryAssertionOperator;
    auto shiftPropertyBy =
        [&](Value property, uint64_t delayCycles,
            bool requireFiniteProgress = false) -> Value {
      if (delayCycles == 0)
        return property;
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto delayedTrue =
          ltl::DelayOp::create(builder, loc, trueVal,
                               builder.getI64IntegerAttr(delayCycles),
                               builder.getI64IntegerAttr(0));
      auto shifted =
          ltl::ImplicationOp::create(builder, loc, delayedTrue, property);
      if (!requireFiniteProgress)
        return shifted;
      return ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{delayedTrue, shifted});
    };
    auto makeEventually = [&](Value property, bool isWeak) -> Value {
      auto eventually = ltl::EventuallyOp::create(builder, loc, property);
      if (isWeak)
        eventually->setAttr(kWeakEventuallyAttr, builder.getUnitAttr());
      return eventually;
    };
    auto lowerAlwaysProperty = [&](Value property, bool isStrongAlways) -> Value {
      auto neg = ltl::NotOp::create(builder, loc, property);
      auto eventually = makeEventually(neg, /*isWeak=*/!isStrongAlways);
      return ltl::NotOp::create(builder, loc, eventually);
    };
    auto requireStrongFiniteProgress = [&](Value temporalExpr) -> Value {
      auto eventually = makeEventually(temporalExpr, /*isWeak=*/false);
      return ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{temporalExpr, eventually});
    };
    switch (expr.op) {
    case UnaryAssertionOperator::Not:
      return ltl::NotOp::create(builder, loc, value);
    case UnaryAssertionOperator::SEventually:
      if (expr.range.has_value()) {
        if (isa<ltl::PropertyType>(value.getType())) {
          auto minDelay = expr.range.value().min;
          if (!expr.range.value().max.has_value()) {
            return makeEventually(
                shiftPropertyBy(value, minDelay, /*requireFiniteProgress=*/true),
                                  /*isWeak=*/false);
          }
          auto maxDelay = expr.range.value().max.value();
          SmallVector<Value, 4> shifted;
          shifted.reserve(maxDelay - minDelay + 1);
          for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
               ++delayCycles)
            shifted.push_back(shiftPropertyBy(value, delayCycles,
                                              /*requireFiniteProgress=*/true));
          if (shifted.size() == 1)
            return shifted.front();
          return ltl::OrOp::create(builder, loc, shifted);
        }
        auto minDelay = builder.getI64IntegerAttr(expr.range.value().min);
        auto lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(
              expr.range.value().max.value() - expr.range.value().min);
        }
        auto delayed = ltl::DelayOp::create(builder, loc, value, minDelay,
                                            lengthAttr);
        return requireStrongFiniteProgress(delayed);
      }
      return ltl::EventuallyOp::create(builder, loc, value);
    case UnaryAssertionOperator::Eventually: {
      if (expr.range.has_value()) {
        if (isa<ltl::PropertyType>(value.getType())) {
          auto minDelay = expr.range.value().min;
          if (!expr.range.value().max.has_value()) {
            mlir::emitError(loc)
                << "unbounded eventually range on property expressions is not "
                   "yet supported";
            return {};
          }
          auto maxDelay = expr.range.value().max.value();
          SmallVector<Value, 4> shifted;
          shifted.reserve(maxDelay - minDelay + 1);
          for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
               ++delayCycles)
            shifted.push_back(shiftPropertyBy(value, delayCycles));
          if (shifted.size() == 1)
            return shifted.front();
          return ltl::OrOp::create(builder, loc, shifted);
        }
        auto minDelay = builder.getI64IntegerAttr(expr.range.value().min);
        auto lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(
              expr.range.value().max.value() - expr.range.value().min);
        }
        return ltl::DelayOp::create(builder, loc, value, minDelay,
                                    lengthAttr);
      }
      auto eventually = ltl::EventuallyOp::create(builder, loc, value);
      eventually->setAttr(kWeakEventuallyAttr, builder.getUnitAttr());
      return eventually;
    }
    case UnaryAssertionOperator::Always: {
      if (isa<ltl::PropertyType>(value.getType())) {
        if (expr.range.has_value()) {
          auto minDelay = expr.range.value().min;
          if (!expr.range.value().max.has_value()) {
            auto shifted = shiftPropertyBy(value, minDelay);
            return lowerAlwaysProperty(shifted, /*isStrongAlways=*/false);
          }
          auto maxDelay = expr.range.value().max.value();
          SmallVector<Value, 4> shifted;
          shifted.reserve(maxDelay - minDelay + 1);
          for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
               ++delayCycles)
            shifted.push_back(shiftPropertyBy(value, delayCycles));
          if (shifted.size() == 1)
            return shifted.front();
          return ltl::AndOp::create(builder, loc, shifted);
        }
        return lowerAlwaysProperty(value, /*isStrongAlways=*/false);
      }
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      return ltl::RepeatOp::create(builder, loc, value, attr.first,
                                   attr.second);
    }
    case UnaryAssertionOperator::NextTime: {
      if (isa<ltl::PropertyType>(value.getType())) {
        uint64_t minDelay = 1;
        uint64_t maxDelay = 1;
        if (expr.range.has_value()) {
          minDelay = expr.range.value().min;
          maxDelay = expr.range.value().max.value_or(minDelay);
        }
        SmallVector<Value, 4> shifted;
        shifted.reserve(maxDelay - minDelay + 1);
        for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
             ++delayCycles)
          shifted.push_back(shiftPropertyBy(value, delayCycles));
        if (shifted.size() == 1)
          return shifted.front();
        return ltl::OrOp::create(builder, loc, shifted);
      }
      auto minRepetitions = builder.getI64IntegerAttr(1);
      mlir::IntegerAttr lengthAttr = builder.getI64IntegerAttr(0);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
        lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(expr.range.value().max.value() -
                                                 expr.range.value().min);
        }
      }
      return ltl::DelayOp::create(builder, loc, value, minRepetitions,
                                  lengthAttr);
    }
    case UnaryAssertionOperator::SNextTime: {
      if (isa<ltl::PropertyType>(value.getType())) {
        uint64_t minDelay = 1;
        uint64_t maxDelay = 1;
        if (expr.range.has_value()) {
          minDelay = expr.range.value().min;
          maxDelay = expr.range.value().max.value_or(minDelay);
        }
        SmallVector<Value, 4> shifted;
        shifted.reserve(maxDelay - minDelay + 1);
        for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
             ++delayCycles)
          shifted.push_back(shiftPropertyBy(value, delayCycles,
                                            /*requireFiniteProgress=*/true));
        if (shifted.size() == 1)
          return shifted.front();
        return ltl::OrOp::create(builder, loc, shifted);
      }
      auto minRepetitions = builder.getI64IntegerAttr(1);
      mlir::IntegerAttr lengthAttr = builder.getI64IntegerAttr(0);
      if (expr.range.has_value()) {
        minRepetitions = builder.getI64IntegerAttr(expr.range.value().min);
        lengthAttr = mlir::IntegerAttr{};
        if (expr.range.value().max.has_value()) {
          lengthAttr = builder.getI64IntegerAttr(expr.range.value().max.value() -
                                                 expr.range.value().min);
        }
      }
      auto shifted = ltl::DelayOp::create(builder, loc, value, minRepetitions,
                                          lengthAttr);
      return requireStrongFiniteProgress(shifted);
    }
    case UnaryAssertionOperator::SAlways: {
      if (isa<ltl::PropertyType>(value.getType())) {
        if (expr.range.has_value()) {
          if (!expr.range.value().max.has_value()) {
            mlir::emitError(loc)
                << "unbounded s_always range on property expressions is not "
                   "yet supported";
            return {};
          }
          auto minDelay = expr.range.value().min;
          auto maxDelay = expr.range.value().max.value();
          SmallVector<Value, 4> shifted;
          shifted.reserve(maxDelay - minDelay + 1);
          for (uint64_t delayCycles = minDelay; delayCycles <= maxDelay;
               ++delayCycles)
            shifted.push_back(shiftPropertyBy(value, delayCycles,
                                              /*requireFiniteProgress=*/true));
          if (shifted.size() == 1)
            return shifted.front();
          return ltl::AndOp::create(builder, loc, shifted);
        }
        return lowerAlwaysProperty(value, /*isStrongAlways=*/true);
      }
      std::pair<mlir::IntegerAttr, mlir::IntegerAttr> attr = {
          builder.getI64IntegerAttr(0), mlir::IntegerAttr{}};
      if (expr.range.has_value()) {
        attr =
            convertRangeToAttrs(expr.range.value().min, expr.range.value().max);
      }
      auto repeated =
          ltl::RepeatOp::create(builder, loc, value, attr.first, attr.second);
      return requireStrongFiniteProgress(repeated);
    }
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::BinaryAssertionExpr &expr) {
    auto lhs =
        context.convertAssertionExpression(expr.left, loc, /*applyDefaults=*/false);
    auto rhs =
        context.convertAssertionExpression(expr.right, loc, /*applyDefaults=*/false);
    if (!lhs || !rhs)
      return {};
    SmallVector<Value, 2> operands = {lhs, rhs};
    using slang::ast::BinaryAssertionOperator;
    switch (expr.op) {
    case BinaryAssertionOperator::And:
      return ltl::AndOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Or:
      return ltl::OrOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Intersect:
      return ltl::IntersectOp::create(builder, loc, operands);
    case BinaryAssertionOperator::Throughout: {
      auto minAttr = builder.getI64IntegerAttr(0);
      mlir::IntegerAttr moreAttr;
      if (auto bounds = getSequenceLengthBounds(rhs)) {
        minAttr = builder.getI64IntegerAttr(bounds->min);
        if (bounds->max && *bounds->max >= bounds->min)
          moreAttr = builder.getI64IntegerAttr(*bounds->max - bounds->min);
      }
      auto lhsRepeat =
          ltl::RepeatOp::create(builder, loc, lhs, minAttr, moreAttr);
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{lhsRepeat, rhs});
    }
    case BinaryAssertionOperator::Within: {
      auto constOne =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      auto oneRepeat = ltl::RepeatOp::create(builder, loc, constOne,
                                             builder.getI64IntegerAttr(0),
                                             mlir::IntegerAttr{});
      auto repeatDelay = ltl::DelayOp::create(builder, loc, oneRepeat,
                                              builder.getI64IntegerAttr(1),
                                              builder.getI64IntegerAttr(0));
      auto lhsDelay =
          ltl::DelayOp::create(builder, loc, lhs, builder.getI64IntegerAttr(1),
                               builder.getI64IntegerAttr(0));
      auto combined = ltl::ConcatOp::create(
          builder, loc, SmallVector<Value, 3>{repeatDelay, lhsDelay, constOne});
      return ltl::IntersectOp::create(builder, loc,
                                      SmallVector<Value, 2>{combined, rhs});
    }
    case BinaryAssertionOperator::Iff: {
      auto ored = ltl::OrOp::create(builder, loc, operands);
      auto notOred = ltl::NotOp::create(builder, loc, ored);
      auto anded = ltl::AndOp::create(builder, loc, operands);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notOred, anded});
    }
    case BinaryAssertionOperator::Until:
      return ltl::UntilOp::create(builder, loc, operands);
    case BinaryAssertionOperator::UntilWith: {
      auto untilOp = ltl::UntilOp::create(builder, loc, operands);
      auto andOp = ltl::AndOp::create(builder, loc, operands);
      auto notUntil = ltl::NotOp::create(builder, loc, untilOp);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notUntil, andOp});
    }
    case BinaryAssertionOperator::Implies: {
      auto notLhs = ltl::NotOp::create(builder, loc, lhs);
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{notLhs, rhs});
    }
    case BinaryAssertionOperator::OverlappedImplication: {
      // The antecedent of an implication must be a sequence type (i1 or
      // !ltl.sequence), not a property type. Property types from $rose, $fell,
      // $changed, $stable cannot be used directly as antecedents.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as implication "
                             "antecedent; consider restructuring the assertion "
                             "to use the property as a consequent");
        return {};
      }
      return ltl::ImplicationOp::create(builder, loc, operands);
    }
    case BinaryAssertionOperator::NonOverlappedImplication: {
      // The antecedent of an implication must be a sequence type (i1 or
      // !ltl.sequence), not a property type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as implication "
                             "antecedent; consider restructuring the assertion "
                             "to use the property as a consequent");
        return {};
      }
      if (isa<ltl::PropertyType>(rhs.getType())) {
        // Use past-shifted antecedent to avoid concat+delay true in BMC.
        // ltl.past only accepts i1, so use delay+concat for sequences.
        Value pastAntecedent;
        if (lhs.getType().isInteger(1)) {
          pastAntecedent = ltl::PastOp::create(builder, loc, lhs, 1).getResult();
        } else {
          // For sequences, use delay to shift the antecedent back.
          auto constOne =
              hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
          auto lhsDelay = ltl::DelayOp::create(
              builder, loc, lhs, builder.getI64IntegerAttr(1),
              builder.getI64IntegerAttr(0));
          pastAntecedent = ltl::ConcatOp::create(
              builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
        }
        return ltl::ImplicationOp::create(
            builder, loc, SmallVector<Value, 2>{pastAntecedent, rhs});
      }
      auto ltlSeqType = ltl::SequenceType::get(builder.getContext());
      auto delayedRhs = ltl::DelayOp::create(
          builder, loc, ltlSeqType, rhs, builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(0));
      return ltl::ImplicationOp::create(builder, loc,
                                        SmallVector<Value, 2>{lhs, delayedRhs});
    }
    case BinaryAssertionOperator::OverlappedFollowedBy: {
      // The antecedent of an implication must be a sequence type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as followed-by "
                             "antecedent; consider restructuring the assertion");
        return {};
      }
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, notRhs});
      return ltl::NotOp::create(builder, loc, implication);
    }
    case BinaryAssertionOperator::NonOverlappedFollowedBy: {
      // The antecedent of an implication must be a sequence type.
      if (isa<ltl::PropertyType>(lhs.getType())) {
        mlir::emitError(loc, "property type cannot be used as followed-by "
                             "antecedent; consider restructuring the assertion");
        return {};
      }
      auto notRhs = ltl::NotOp::create(builder, loc, rhs);
      if (isa<ltl::PropertyType>(notRhs.getType())) {
        auto constOne =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
        auto lhsDelay = ltl::DelayOp::create(
            builder, loc, lhs, builder.getI64IntegerAttr(1),
            builder.getI64IntegerAttr(0));
        auto antecedent = ltl::ConcatOp::create(
            builder, loc, SmallVector<Value, 2>{lhsDelay, constOne});
        auto implication = ltl::ImplicationOp::create(
            builder, loc, SmallVector<Value, 2>{antecedent, notRhs});
        return ltl::NotOp::create(builder, loc, implication);
      }
      auto ltlSeqType = ltl::SequenceType::get(builder.getContext());
      auto delayedRhs = ltl::DelayOp::create(
          builder, loc, ltlSeqType, notRhs, builder.getI64IntegerAttr(1),
          builder.getI64IntegerAttr(0));
      auto implication = ltl::ImplicationOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, delayedRhs});
      return ltl::NotOp::create(builder, loc, implication);
    }
    case BinaryAssertionOperator::SUntil: {
      // Strong until: a U b AND eventually b.
      auto untilOp = ltl::UntilOp::create(builder, loc, operands);
      auto eventuallyRhs =
          ltl::EventuallyOp::create(builder, loc, rhs);
      return ltl::AndOp::create(builder, loc,
                                SmallVector<Value, 2>{untilOp, eventuallyRhs});
    }
    case BinaryAssertionOperator::SUntilWith: {
      // Strong until-with: require overlap at termination and eventual b.
      auto andOp = ltl::AndOp::create(builder, loc, operands);
      auto untilWith = ltl::UntilOp::create(
          builder, loc, SmallVector<Value, 2>{lhs, andOp});
      auto eventuallyAnd =
          ltl::EventuallyOp::create(builder, loc, andOp);
      return ltl::AndOp::create(builder, loc,
                                SmallVector<Value, 2>{untilWith, eventuallyAnd});
    }
    }
    llvm_unreachable("All enum values handled in switch");
  }

  Value visit(const slang::ast::ClockingAssertionExpr &expr) {
    auto *previousTiming = context.currentAssertionTimingControl;
    auto timingGuard = llvm::make_scope_exit(
        [&] { context.currentAssertionTimingControl = previousTiming; });
    context.currentAssertionTimingControl = &expr.clocking;
    auto assertionExpr =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!assertionExpr)
      return {};
    return context.convertLTLTimingControl(expr.clocking, assertionExpr);
  }

  Value visit(const slang::ast::ConditionalAssertionExpr &expr) {
    auto condition = context.convertRvalueExpression(expr.condition);
    condition = context.convertToBool(condition);
    condition = context.convertToI1(condition);
    if (!condition)
      return {};

    auto ifExpr =
        context.convertAssertionExpression(expr.ifExpr, loc, /*applyDefaults=*/false);
    if (!ifExpr)
      return {};

    auto notCond = ltl::NotOp::create(builder, loc, condition);

    if (expr.elseExpr) {
      auto elseExpr = context.convertAssertionExpression(*expr.elseExpr, loc,
                                                         /*applyDefaults=*/false);
      if (!elseExpr)
        return {};
      auto condAndIf =
          ltl::AndOp::create(builder, loc, SmallVector<Value, 2>{condition, ifExpr});
      auto notCondAndElse =
          ltl::AndOp::create(builder, loc, SmallVector<Value, 2>{notCond, elseExpr});
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{condAndIf, notCondAndElse});
    }

    return ltl::OrOp::create(builder, loc,
                             SmallVector<Value, 2>{notCond, ifExpr});
  }

  Value visit(const slang::ast::CaseAssertionExpr &expr) {
    auto selector = context.convertRvalueExpression(expr.expr);
    if (!selector)
      return {};
    bool selectorIsString = isa<moore::StringType, moore::FormatStringType>(
        selector.getType());
    if (selectorIsString) {
      auto strTy = moore::StringType::get(context.getContext());
      selector =
          context.materializeConversion(strTy, selector, /*isSigned=*/false, loc);
      if (!selector)
        return {};
    } else {
      selector = context.convertToSimpleBitVector(selector);
      if (!selector)
        return {};
    }

    Value result;
    if (expr.defaultCase)
      result = context.convertAssertionExpression(*expr.defaultCase, loc,
                                                  /*applyDefaults=*/false);
    if (expr.defaultCase && !result)
      return {};

    for (auto itemIt = expr.items.rbegin(); itemIt != expr.items.rend();
         ++itemIt) {
      auto body = context.convertAssertionExpression(*itemIt->body, loc,
                                                     /*applyDefaults=*/false);
      if (!body)
        return {};

      Value groupCond;
      for (auto *caseExpr : itemIt->expressions) {
        if (!caseExpr)
          continue;
        auto caseValue = context.convertRvalueExpression(*caseExpr);
        if (!caseValue)
          return {};
        Value match;
        if (selectorIsString) {
          auto strTy = moore::StringType::get(context.getContext());
          caseValue = context.materializeConversion(
              strTy, caseValue, /*isSigned=*/false,
              context.convertLocation(caseExpr->sourceRange));
          if (!caseValue)
            return {};
          match = moore::StringCmpOp::create(builder, loc,
                                             moore::StringCmpPredicate::eq,
                                             selector, caseValue);
        } else {
          caseValue = context.convertToSimpleBitVector(caseValue);
          if (!caseValue)
            return {};
          if (caseValue.getType() != selector.getType()) {
            caseValue = context.materializeConversion(
                selector.getType(), caseValue, /*isSigned=*/false,
                context.convertLocation(caseExpr->sourceRange));
            if (!caseValue)
              return {};
          }
          match = moore::CaseEqOp::create(builder, loc, selector, caseValue);
        }
        match = context.convertToBool(match);
        match = context.convertToI1(match);
        if (!match)
          return {};
        if (!groupCond)
          groupCond = match;
        else
          groupCond = ltl::OrOp::create(builder, loc,
                                        SmallVector<Value, 2>{groupCond, match});
      }
      if (!groupCond)
        continue;

      auto condAndBody = ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{groupCond, body});
      if (!result) {
        result = condAndBody;
        continue;
      }

      auto notCond = ltl::NotOp::create(builder, loc, groupCond);
      auto notCondAndElse = ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{notCond, result});
      result = ltl::OrOp::create(builder, loc,
                                 SmallVector<Value, 2>{condAndBody,
                                                       notCondAndElse});
    }

    if (!result)
      result = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    return result;
  }

  Value visit(const slang::ast::StrongWeakAssertionExpr &expr) {
    auto value =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!value)
      return {};
    // Distinguish strong/weak wrappers by adding an explicit finite-progress
    // obligation for strong(...) via eventually(...). Weak(...) remains
    // safety-only in the current lowering.
    if (expr.strength == slang::ast::StrongWeakAssertionExpr::Strong) {
      auto eventually = ltl::EventuallyOp::create(builder, loc, value);
      return ltl::AndOp::create(
          builder, loc, SmallVector<Value, 2>{value, eventually});
    }
    return value;
  }

  Value visit(const slang::ast::DisableIffAssertionExpr &expr) {
    auto disableCond = context.convertRvalueExpression(expr.condition);
    // IEEE 1800 allows general integral truthy expressions in `disable iff`.
    // Normalize to a boolean first, then to builtin i1 for LTL lowering.
    disableCond = context.convertToBool(disableCond);
    disableCond = context.convertToI1(disableCond);
    if (!disableCond)
      return {};

    context.pushAssertionDisableExpr(&expr.condition);
    auto disableGuard =
        llvm::make_scope_exit([&] { context.popAssertionDisableExpr(); });
    auto assertionExpr =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!assertionExpr)
      return {};

    // Approximate disable iff by treating the property as vacuously true when
    // the disable condition holds.
    auto orOp = ltl::OrOp::create(
        builder, loc, SmallVector<Value, 2>{disableCond, assertionExpr});
    orOp->setAttr(kDisableIffAttr, builder.getUnitAttr());
    return orOp.getResult();
  }

  Value visit(const slang::ast::AbortAssertionExpr &expr) {
    auto condition = context.convertRvalueExpression(expr.condition);
    condition = context.convertToBool(condition);
    condition = context.convertToI1(condition);
    if (!condition)
      return {};

    if (expr.isSync) {
      const slang::ast::TimingControl *clockingCtrl = nullptr;
      if (context.currentAssertionClock)
        clockingCtrl = context.currentAssertionClock;
      if (!clockingCtrl && context.currentAssertionTimingControl)
        clockingCtrl = context.currentAssertionTimingControl;
      if (!clockingCtrl && context.currentScope) {
        if (auto *clocking =
                context.compilation.getDefaultClocking(*context.currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
      if (clockingCtrl) {
        condition = context.convertLTLTimingControl(*clockingCtrl, condition);
        if (!condition)
          return {};
      }
    }

    auto assertionExpr =
        context.convertAssertionExpression(expr.expr, loc, /*applyDefaults=*/false);
    if (!assertionExpr)
      return {};

    if (expr.action == slang::ast::AbortAssertionExpr::Accept) {
      // Approximate accept_on as vacuous success when the abort condition is
      // true; for sync_accept_on, condition is sampled on the property clock.
      return ltl::OrOp::create(builder, loc,
                               SmallVector<Value, 2>{condition, assertionExpr});
    }

    // Approximate reject_on as forcing failure when the abort condition is
    // true; for sync_reject_on, condition is sampled on the property clock.
    auto notCondition = ltl::NotOp::create(builder, loc, condition);
    return ltl::AndOp::create(builder, loc,
                              SmallVector<Value, 2>{notCondition, assertionExpr});
  }

  /// Emit an error for all other expressions.
  template <typename T>
  Value visit(T &&node) {
    mlir::emitError(loc, "unsupported expression: ")
        << slang::ast::toString(node.kind);
    return {};
  }

  Value visitInvalid(const slang::ast::AssertionExpr &expr) {
    // Slang wraps assertion expressions in InvalidAssertionExpr when they
    // appear in dead generate blocks (e.g., `if (ICache)` defaulting to 0).
    // Try to unwrap and convert the child; if that also fails (unresolvable
    // hierarchical refs in dead code), return {} so the caller can skip the
    // assertion entirely.
    if (auto *invalid =
            expr.as_if<slang::ast::InvalidAssertionExpr>()) {
      if (invalid->child) {
        auto result = invalid->child->visit(*this);
        if (result)
          return result;
      }
      return {};
    }
    mlir::emitError(loc, "invalid expression");
    return {};
  }
};
} // namespace

FailureOr<Value> Context::convertAssertionSystemCallArity1(
    const slang::ast::SystemSubroutine &subroutine, Location loc, Value value) {

  auto systemCallRes =
      llvm::StringSwitch<std::function<FailureOr<Value>()>>(subroutine.name)
          // Note: $rose/$fell/$stable/$changed are handled in
          // convertAssertionCallExpression to keep them usable in sequences.
          .Case("$fell",
                [&]() -> Value {
                  return {};
                })
          // Translate $rose to x[0] ∧ ¬x[-1]
          .Case("$rose",
                [&]() -> Value {
                  return {};
                })
          // Translate $stable to ( x[0] ∧ x[-1] ) ⋁ ( ¬x[0] ∧ ¬x[-1] )
          .Case("$stable",
                [&]() -> Value {
                  return {};
                })
          // Translate $changed to ¬$stable(x).
          .Case("$changed",
                [&]() -> Value {
                  return {};
                })
          // $sampled is handled in convertAssertionCallExpression.
          .Case("$sampled", [&]() -> Value { return {}; })
          // Note: $past is handled separately in convertAssertionCallExpression
          // using moore::PastOp to preserve the type for comparisons.
          .Default([&]() -> Value { return {}; });
  return systemCallRes();
}

Value Context::convertAssertionCallExpression(
    const slang::ast::CallExpression &expr,
    const slang::ast::CallExpression::SystemCallInfo &info, Location loc) {

  const auto &subroutine = *info.subroutine;
  auto args = expr.arguments();

  // Normalize _gclk sampled value function variants (IEEE 1800-2017 §16.9.3)
  // to their base equivalents. The global clock semantics are equivalent for
  // elaboration purposes since the test assertions already specify a clock.
  std::string normalizedName =
      llvm::StringSwitch<std::string>(subroutine.name)
          .Case("$rose_gclk", "$rose")
          .Case("$fell_gclk", "$fell")
          .Case("$stable_gclk", "$stable")
          .Case("$changed_gclk", "$changed")
          .Case("$past_gclk", "$past")
          .Case("$future_gclk", "$future")
          .Case("$rising_gclk", "$rose")
          .Case("$falling_gclk", "$fell")
          .Case("$steady_gclk", "$stable")
          .Case("$changing_gclk", "$changed")
          .Default(std::string(subroutine.name));
  StringRef funcName(normalizedName);
  bool isGlobalClockVariant = StringRef(subroutine.name).ends_with("_gclk");

  FailureOr<Value> result;
  Value value;
  Value boolVal;

  if (funcName == "$future") {
    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};
    value = this->convertToBool(value);
    value = this->convertToI1(value);
    if (!value)
      return {};
    auto delayed = ltl::DelayOp::create(builder, loc, value,
                                        builder.getI64IntegerAttr(1),
                                        builder.getI64IntegerAttr(0));
    if (isGlobalClockVariant && inAssertionExpr && !currentAssertionClock &&
        !currentAssertionTimingControl && currentScope) {
      if (auto *clocking =
              compilation.getGlobalClockingAndNoteUse(*currentScope)) {
        if (auto *clockBlock =
                clocking->as_if<slang::ast::ClockingBlockSymbol>()) {
          return convertLTLTimingControl(clockBlock->getEvent(), delayed);
        }
      }
    }
    return delayed;
  }

  if (funcName == "$rose" || funcName == "$fell" ||
      funcName == "$stable" || funcName == "$changed") {
    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};
    bool isAggregateSample =
        (funcName == "$stable" || funcName == "$changed") &&
        (isa<moore::UnpackedArrayType>(value.getType()) ||
         isa<moore::OpenUnpackedArrayType>(value.getType()) ||
         isa<moore::QueueType>(value.getType()) ||
         isa<moore::AssocArrayType>(value.getType()) ||
         isa<moore::WildcardAssocArrayType>(value.getType()) ||
         isa<moore::UnpackedStructType>(value.getType()) ||
         isa<moore::UnpackedUnionType>(value.getType()));
    bool isAggregateEdgeSample =
        (funcName == "$rose" || funcName == "$fell") &&
        (isa<moore::UnpackedArrayType>(value.getType()) ||
         isa<moore::OpenUnpackedArrayType>(value.getType()) ||
         isa<moore::QueueType>(value.getType()) ||
         isa<moore::AssocArrayType>(value.getType()) ||
         isa<moore::WildcardAssocArrayType>(value.getType()) ||
         isa<moore::UnpackedStructType>(value.getType()) ||
         isa<moore::UnpackedUnionType>(value.getType()));
    bool isRealSample = isa<moore::RealType>(value.getType());
    bool isStringStableSample =
        (funcName == "$stable" || funcName == "$changed") &&
        isa<moore::StringType, moore::FormatStringType>(value.getType());
    bool isEventStableSample =
        (funcName == "$stable" || funcName == "$changed") &&
        isa<moore::EventType>(value.getType());
    bool isStringEdgeSample =
        (funcName == "$rose" || funcName == "$fell") &&
        isa<moore::StringType, moore::FormatStringType>(value.getType());
    bool isEventEdgeSample =
        (funcName == "$rose" || funcName == "$fell") &&
        isa<moore::EventType>(value.getType());
    if (!isAggregateSample && !isAggregateEdgeSample &&
        !isRealSample && !isStringStableSample && !isEventStableSample &&
        !isStringEdgeSample && !isEventEdgeSample &&
        (isa<moore::UnpackedArrayType>(value.getType()) ||
         isa<moore::OpenUnpackedArrayType>(value.getType()) ||
         isa<moore::QueueType>(value.getType()) ||
         isa<moore::AssocArrayType>(value.getType()) ||
         isa<moore::WildcardAssocArrayType>(value.getType()) ||
         isa<moore::UnpackedStructType>(value.getType()) ||
         isa<moore::UnpackedUnionType>(value.getType()))) {
      mlir::emitError(loc) << "unsupported sampled value type for " << funcName;
      return {};
    }

    if (!isAggregateSample && !isAggregateEdgeSample &&
        !isRealSample && !isStringStableSample && !isEventStableSample &&
        !isStringEdgeSample && !isEventEdgeSample &&
        !isa<moore::IntType>(value.getType()))
      value = convertToSimpleBitVector(value);
    if (!value)
      return {};
    if (!isAggregateSample && !isAggregateEdgeSample &&
        !isRealSample && !isStringStableSample && !isEventStableSample &&
        !isStringEdgeSample && !isEventEdgeSample &&
        !isa<moore::IntType>(value.getType())) {
      mlir::emitError(loc) << "unsupported sampled value type for "
                           << funcName;
      return {};
    }
    const slang::ast::TimingControl *clockingCtrl = nullptr;
    bool inferredImplicitClocking = false;
    bool hasClockingArg =
        args.size() > 1 &&
        args[1]->kind == slang::ast::ExpressionKind::ClockingEvent;
    if (hasClockingArg) {
      if (auto *clockExpr =
              args[1]->as_if<slang::ast::ClockingEventExpression>()) {
        clockingCtrl = &clockExpr->timingControl;
      } else if (!inAssertionExpr) {
        auto resultType = moore::IntType::getInt(builder.getContext(), 1);
        mlir::emitWarning(loc)
            << funcName
            << " with explicit clocking is not yet lowered outside assertions; "
               "returning 0 as a placeholder";
        return moore::ConstantOp::create(builder, loc, resultType, 0);
      }
    }

    const slang::ast::Expression *enableExpr = nullptr;
    SmallVector<const slang::ast::Expression *, 4> disableExprs;
    if (inAssertionExpr) {
      if (currentScope) {
        if (auto *defaultDisable =
                compilation.getDefaultDisable(*currentScope))
          disableExprs.push_back(defaultDisable);
      }
      for (auto *expr : getAssertionDisableExprs())
        disableExprs.push_back(expr);
    }

    if (inAssertionExpr && !clockingCtrl) {
      if (currentAssertionClock)
        clockingCtrl = currentAssertionClock;
      if (!clockingCtrl && currentAssertionTimingControl)
        clockingCtrl = currentAssertionTimingControl;
      if (!clockingCtrl && currentScope) {
        if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
      if (!clockingCtrl && isGlobalClockVariant && currentScope) {
        if (auto *clocking =
                compilation.getGlobalClockingAndNoteUse(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
    }
    if (!inAssertionExpr && !clockingCtrl && isGlobalClockVariant &&
        currentScope) {
      if (auto *clocking =
              compilation.getGlobalClockingAndNoteUse(*currentScope)) {
        if (auto *clockBlock =
                clocking->as_if<slang::ast::ClockingBlockSymbol>())
          clockingCtrl = &clockBlock->getEvent();
      }
    }
    if (!inAssertionExpr && !clockingCtrl && currentScope) {
      if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
        if (auto *clockBlock =
                clocking->as_if<slang::ast::ClockingBlockSymbol>()) {
          clockingCtrl = &clockBlock->getEvent();
          inferredImplicitClocking = true;
        }
      }
    }

    // The helper-procedure lowering introduces explicit sampled state. For
    // implicit assertion clocks without disable/enable controls, prefer direct
    // past-based lowering to avoid extra-cycle skew in temporal combinations
    // such as non-overlap implication with $rose/$fell.
    //
    // Also allow direct lowering when the sampled-value explicit clocking
    // argument is structurally equivalent to the enclosing assertion clock,
    // since both sample on the same event stream.
    bool explicitClockMatchesAssertionClock = false;
    if (hasClockingArg && inAssertionExpr && clockingCtrl) {
      if (currentAssertionClock &&
          isEquivalentTimingControl(*clockingCtrl, *currentAssertionClock))
        explicitClockMatchesAssertionClock = true;
      else if (currentAssertionTimingControl &&
               isEquivalentTimingControl(*clockingCtrl,
                                         *currentAssertionTimingControl))
        explicitClockMatchesAssertionClock = true;
    }
    bool forceClockedHelperForUnclockedGclk =
        isGlobalClockVariant && clockingCtrl && inAssertionExpr &&
        !currentAssertionClock && !currentAssertionTimingControl;
    bool needsClockedHelper =
        enableExpr || !disableExprs.empty() ||
        (hasClockingArg && !explicitClockMatchesAssertionClock) ||
        forceClockedHelperForUnclockedGclk;
    if (inAssertionExpr && needsClockedHelper) {
      auto sampled = lowerSampledValueFunctionWithSamplingControl(
          *this, *args[0], clockingCtrl, funcName, enableExpr, disableExprs,
          loc);
      if (!sampled)
        return {};
      if (forceClockedHelperForUnclockedGclk) {
        sampled = convertToBool(sampled);
        sampled = convertToI1(sampled);
        if (!sampled)
          return {};
        return convertLTLTimingControl(*clockingCtrl, sampled);
      }
      return sampled;
    }

    if (!inAssertionExpr && clockingCtrl &&
        (hasClockingArg || inferredImplicitClocking || isGlobalClockVariant)) {
      if (clockingCtrl) {
        return lowerSampledValueFunctionWithClocking(
            *this, *args[0], *clockingCtrl, funcName, nullptr,
            std::span<const slang::ast::Expression *const>{}, loc);
      }
      auto resultType = moore::IntType::getInt(builder.getContext(), 1);
      mlir::emitWarning(loc)
          << funcName
          << " with explicit clocking is not yet lowered outside assertions; "
             "returning 0 as a placeholder";
      return moore::ConstantOp::create(builder, loc, resultType, 0);
    }

    if (funcName == "$stable" || funcName == "$changed") {
      Value sampled = value;
      Value past;
      if (inAssertionExpr) {
        // Sampled-value semantics: compare the sampled value at this edge with
        // the sampled value from the previous edge.
        sampled = value;
        past =
            moore::PastOp::create(builder, loc, value, /*delay=*/1).getResult();
      } else {
        past =
            moore::PastOp::create(builder, loc, value, /*delay=*/1).getResult();
      }
      Value stable =
          buildSampledStableComparison(*this, loc, sampled, past, funcName);
      if (!stable)
        return {};
      Value resultVal = stable;
      if (funcName == "$changed")
        resultVal = moore::NotOp::create(builder, loc, stable).getResult();
      return resultVal;
    }

    Value current = buildSampledBoolean(*this, loc, value, funcName);
    if (!current)
      return {};
    Value sampled = current;
    Value past;
    if (inAssertionExpr) {
      // Sampled-value semantics: use the sampled current value and sampled past.
      sampled = current;
      past =
          moore::PastOp::create(builder, loc, current, /*delay=*/1).getResult();
    } else {
      past =
          moore::PastOp::create(builder, loc, current, /*delay=*/1).getResult();
    }
    Value resultVal;
    if (funcName == "$rose") {
      auto notPast = moore::NotOp::create(builder, loc, past).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, sampled, notPast).getResult();
    } else {
      auto notCurrent =
          moore::NotOp::create(builder, loc, sampled).getResult();
      resultVal =
          moore::AndOp::create(builder, loc, notCurrent, past).getResult();
    }
    return resultVal;
  }

  // Handle $past specially - it returns the past value with preserved type
  // so that comparisons like `$past(val) == 0` work correctly.
  if (funcName == "$past") {
    // Get the delay (numTicks) from the second argument if present.
    // Default to 1 if empty or not provided.
    int64_t delay = 1;
    const slang::ast::TimingControl *clockingCtrl = nullptr;
    bool hasClockingArg = false;
    const slang::ast::Expression *enableExpr = nullptr;
    SmallVector<const slang::ast::Expression *, 4> disableExprs;
    if (args.size() > 1 &&
        args[1]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[1]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (auto *clockExpr =
                args[1]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
        hasClockingArg = true;
      } else {
        auto cv = evaluateConstant(*args[1]);
        if (cv.isInteger()) {
          auto intVal = cv.integer().as<int64_t>();
          if (intVal)
            delay = *intVal;
        }
      }
    }
    if (!clockingCtrl && args.size() > 2 &&
        args[2]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[2]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (auto *clockExpr =
                args[2]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
        hasClockingArg = true;
      } else {
        enableExpr = args[2];
      }
    }
    if (args.size() > 3 &&
        args[3]->kind != slang::ast::ExpressionKind::EmptyArgument) {
      if (args[3]->kind == slang::ast::ExpressionKind::ClockingEvent) {
        if (clockingCtrl) {
          mlir::emitError(loc) << "multiple $past clocking events";
          return {};
        }
        if (auto *clockExpr =
                args[3]->as_if<slang::ast::ClockingEventExpression>())
          clockingCtrl = &clockExpr->timingControl;
        hasClockingArg = true;
      } else if (!enableExpr) {
        enableExpr = args[3];
      } else {
        mlir::emitError(loc) << "too many $past arguments";
        return {};
      }
    }
    auto maybeSetImplicitClocking = [&]() {
      if (!clockingCtrl && currentAssertionClock)
        clockingCtrl = currentAssertionClock;
      if (!clockingCtrl && currentAssertionTimingControl)
        clockingCtrl = currentAssertionTimingControl;
      if (!clockingCtrl && currentScope) {
        if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
      if (!clockingCtrl && isGlobalClockVariant && currentScope) {
        if (auto *clocking =
                compilation.getGlobalClockingAndNoteUse(*currentScope)) {
          if (auto *clockBlock =
                  clocking->as_if<slang::ast::ClockingBlockSymbol>())
            clockingCtrl = &clockBlock->getEvent();
        }
      }
    };
    if (!clockingCtrl)
      maybeSetImplicitClocking();
    if (inAssertionExpr) {
      if (currentScope) {
        if (auto *defaultDisable =
                compilation.getDefaultDisable(*currentScope))
          disableExprs.push_back(defaultDisable);
      }
      for (auto *expr : getAssertionDisableExprs())
        disableExprs.push_back(expr);
    }
    if (clockingCtrl) {
      if (!inAssertionExpr)
        return lowerPastWithClocking(*this, *args[0], *clockingCtrl, delay,
                                     enableExpr, disableExprs, loc);
      bool explicitClockMatchesAssertionClock = false;
      if (hasClockingArg) {
        if (currentAssertionClock &&
            isEquivalentTimingControl(*clockingCtrl, *currentAssertionClock))
          explicitClockMatchesAssertionClock = true;
        else if (currentAssertionTimingControl &&
                 isEquivalentTimingControl(*clockingCtrl,
                                           *currentAssertionTimingControl))
          explicitClockMatchesAssertionClock = true;
      }
      bool forceClockedHelperForUnclockedGclk =
          isGlobalClockVariant && clockingCtrl && !currentAssertionClock &&
          !currentAssertionTimingControl;
      bool needsClockedHelper =
          enableExpr || !disableExprs.empty() ||
          (hasClockingArg && !explicitClockMatchesAssertionClock) ||
          forceClockedHelperForUnclockedGclk;
      if (needsClockedHelper) {
        auto sampled = lowerPastWithClocking(*this, *args[0], *clockingCtrl,
                                             delay, enableExpr, disableExprs,
                                             loc);
        if (!sampled)
          return {};
        if (forceClockedHelperForUnclockedGclk) {
          sampled = convertToBool(sampled);
          sampled = convertToI1(sampled);
          if (!sampled)
            return {};
          return convertLTLTimingControl(*clockingCtrl, sampled);
        }
        return sampled;
      }
    }
    if (enableExpr || !disableExprs.empty()) {
      auto sampled = lowerPastWithSamplingControl(
          *this, *args[0], /*timingCtrl=*/nullptr, delay, enableExpr,
          disableExprs, loc);
      if (!sampled)
        return {};
      return sampled;
    }

    value = this->convertRvalueExpression(*args[0]);
    if (!value)
      return {};

    // Always use moore::PastOp to preserve the type for comparisons.
    // $past(val) returns the sampled past value with the same type as val, so
    // that comparisons like `$past(val) == 0` work correctly.
    return moore::PastOp::create(builder, loc, value, delay).getResult();
  }

  switch (args.size()) {
  case (1):
    value = this->convertRvalueExpression(*args[0]);

    // $sampled returns the sampled value of the expression.
    if (funcName == "$sampled") {
      if (inAssertionExpr)
        return moore::PastOp::create(builder, loc, value, /*delay=*/0)
            .getResult();
      return value;
    }

    boolVal = builder.createOrFold<moore::ToBuiltinBoolOp>(loc, value);
    if (!boolVal)
      return {};
    result = this->convertAssertionSystemCallArity1(subroutine, loc, boolVal);
    break;

  default:
    break;
  }

  if (failed(result))
    return {};
  if (*result)
    return *result;

  mlir::emitError(loc) << "unsupported system call `" << funcName << "`";
  return {};
}

Value Context::convertAssertionExpression(const slang::ast::AssertionExpr &expr,
                                          Location loc, bool applyDefaults) {
  bool prevInAssertionExpr = inAssertionExpr;
  if (!prevInAssertionExpr) {
    pushAssertionLocalVarScope();
    pushAssertionSequenceOffset(0);
  }
  inAssertionExpr = true;
  AssertionExprVisitor visitor{*this, loc};
  auto value = expr.visit(visitor);
  inAssertionExpr = prevInAssertionExpr;
  if (!prevInAssertionExpr) {
    popAssertionSequenceOffset();
    popAssertionLocalVarScope();
  }
  if (!value || !applyDefaults)
    return value;

  if (currentScope &&
      (isa<ltl::PropertyType, ltl::SequenceType>(value.getType()) ||
       value.getType().isInteger(1))) {
    if (auto *disableExpr = compilation.getDefaultDisable(*currentScope)) {
      auto disableVal = convertRvalueExpression(*disableExpr);
      disableVal = convertToBool(disableVal);
      disableVal = convertToI1(disableVal);
      if (disableVal) {
        auto orOp = ltl::OrOp::create(
            builder, loc, SmallVector<Value, 2>{disableVal, value});
        orOp->setAttr(kDisableIffAttr, builder.getUnitAttr());
        value = orOp.getResult();
      }
    }

    bool hasExplicitClockAttr =
        value && value.getDefiningOp() &&
        value.getDefiningOp()->hasAttr(kExplicitClockingAttr);
    if (!hasExplicitClockAttr && !containsExplicitClocking(value)) {
      if (auto *clocking = compilation.getDefaultClocking(*currentScope)) {
        if (auto *clockBlock =
                clocking->as_if<slang::ast::ClockingBlockSymbol>()) {
          value = convertLTLTimingControl(clockBlock->getEvent(), value);
        }
      }
    }
  }

  return value;
}
// NOLINTEND(misc-no-recursion)

/// Helper function to convert a value to an i1 value.
Value Context::convertToI1(Value value) {
  if (!value)
    return {};

  // If the value is already an i1 (e.g., from $sampled), return it directly.
  if (value.getType().isInteger(1))
    return value;

  auto type = dyn_cast<moore::IntType>(value.getType());
  if (!type || type.getBitSize() != 1) {
    mlir::emitError(value.getLoc(), "expected a 1-bit integer");
    return {};
  }

  return moore::ToBuiltinBoolOp::create(builder, value.getLoc(), value);
}
