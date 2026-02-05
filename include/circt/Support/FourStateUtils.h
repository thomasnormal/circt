//===- FourStateUtils.h - 4-state helpers ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_FOURSTATEUTILS_H
#define CIRCT_SUPPORT_FOURSTATEUTILS_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace circt {

inline bool isFourStateStructType(mlir::Type type) {
  auto structTy = llvm::dyn_cast<hw::StructType>(type);
  if (!structTy)
    return false;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return false;
  if (!elements[0].name || !elements[1].name)
    return false;
  if (elements[0].name.getValue() != "value")
    return false;
  if (elements[1].name.getValue() != "unknown")
    return false;
  return true;
}

inline std::optional<unsigned> getFourStateValueWidth(mlir::Type type) {
  auto structTy = llvm::dyn_cast<hw::StructType>(type);
  if (!structTy)
    return std::nullopt;
  auto elements = structTy.getElements();
  if (elements.size() != 2)
    return std::nullopt;
  if (!elements[0].name || !elements[1].name)
    return std::nullopt;
  if (elements[0].name.getValue() != "value")
    return std::nullopt;
  if (elements[1].name.getValue() != "unknown")
    return std::nullopt;
  auto valueWidth = hw::getBitWidth(elements[0].type);
  auto unknownWidth = hw::getBitWidth(elements[1].type);
  if (valueWidth <= 0 || unknownWidth <= 0 || valueWidth != unknownWidth)
    return std::nullopt;
  return static_cast<unsigned>(valueWidth);
}

inline mlir::Value resolveFourStatePair(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value lhs,
                                        mlir::Value rhs) {
  if (!lhs || !rhs)
    return mlir::Value();
  if (lhs.getType() != rhs.getType())
    return mlir::Value();
  auto widthOpt = getFourStateValueWidth(lhs.getType());
  if (!widthOpt)
    return mlir::Value();
  auto structTy = llvm::cast<hw::StructType>(lhs.getType());
  unsigned width = *widthOpt;

  auto valueField = structTy.getElements()[0].name;
  auto unknownField = structTy.getElements()[1].name;
  mlir::Value lhsVal =
      builder.createOrFold<hw::StructExtractOp>(loc, lhs, valueField);
  mlir::Value lhsUnk =
      builder.createOrFold<hw::StructExtractOp>(loc, lhs, unknownField);
  mlir::Value rhsVal =
      builder.createOrFold<hw::StructExtractOp>(loc, rhs, valueField);
  mlir::Value rhsUnk =
      builder.createOrFold<hw::StructExtractOp>(loc, rhs, unknownField);

  mlir::Value ones = hw::ConstantOp::create(
                         builder, loc, llvm::APInt::getAllOnes(width))
                         .getResult();
  mlir::Value lhsKnown = builder.createOrFold<comb::XorOp>(loc, lhsUnk, ones);
  mlir::Value rhsKnown = builder.createOrFold<comb::XorOp>(loc, rhsUnk, ones);
  mlir::Value valueDiff =
      builder.createOrFold<comb::XorOp>(loc, lhsVal, rhsVal);
  mlir::Value knownBoth =
      builder.createOrFold<comb::AndOp>(loc, lhsKnown, rhsKnown);
  mlir::Value conflict =
      builder.createOrFold<comb::AndOp>(loc, valueDiff, knownBoth);
  mlir::Value unknownOut =
      builder
          .createOrFold<comb::OrOp>(loc,
                                    llvm::ArrayRef<mlir::Value>{
                                        lhsUnk, rhsUnk, conflict},
                                    /*twoState=*/false)
          ;

  mlir::Value lhsKnownVal =
      builder.createOrFold<comb::AndOp>(loc, lhsVal, lhsKnown);
  mlir::Value rhsKnownVal =
      builder.createOrFold<comb::AndOp>(loc, rhsVal, rhsKnown);
  mlir::Value valueOut =
      builder
          .createOrFold<comb::OrOp>(loc,
                                    llvm::ArrayRef<mlir::Value>{
                                        lhsKnownVal, rhsKnownVal},
                                    /*twoState=*/false)
          ;

  return builder.createOrFold<hw::StructCreateOp>(
      loc, structTy, mlir::ValueRange{valueOut, unknownOut});
}

inline mlir::Value resolveFourStateValues(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return mlir::Value();
  mlir::Value current = values.front();
  for (mlir::Value next : values.drop_front()) {
    mlir::Value resolved = resolveFourStatePair(builder, loc, current, next);
    if (!resolved)
      return mlir::Value();
    current = resolved;
  }
  return current;
}

inline mlir::Value
resolveFourStateValuesWithEnable(mlir::OpBuilder &builder, mlir::Location loc,
                                 llvm::ArrayRef<mlir::Value> values,
                                 llvm::ArrayRef<mlir::Value> enables) {
  if (values.empty())
    return mlir::Value();
  if (values.size() != enables.size())
    return mlir::Value();
  auto widthOpt = getFourStateValueWidth(values.front().getType());
  if (!widthOpt)
    return mlir::Value();
  unsigned width = *widthOpt;
  for (mlir::Value value : values) {
    if (value.getType() != values.front().getType())
      return mlir::Value();
  }
  for (mlir::Value enable : enables) {
    auto intTy = llvm::dyn_cast<mlir::IntegerType>(enable.getType());
    if (!intTy || intTy.getWidth() != 1)
      return mlir::Value();
  }

  auto structTy = llvm::cast<hw::StructType>(values.front().getType());
  auto valueField = structTy.getElements()[0].name;
  auto unknownField = structTy.getElements()[1].name;

  mlir::Value zeros = hw::ConstantOp::create(builder, loc,
                                             llvm::APInt::getZero(width))
                          .getResult();
  mlir::Value ones = hw::ConstantOp::create(builder, loc,
                                            llvm::APInt::getAllOnes(width))
                         .getResult();
  mlir::Value knownOnes = zeros;
  mlir::Value knownZeros = zeros;
  mlir::Value unknownOut = zeros;
  mlir::Value anyEnabled =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0).getResult();

  for (auto [value, enable] : llvm::zip(values, enables)) {
    anyEnabled = comb::OrOp::create(builder, loc, anyEnabled, enable);
    mlir::Value enableVec =
        comb::ReplicateOp::create(builder, loc, enable, width);
    mlir::Value val =
        builder.createOrFold<hw::StructExtractOp>(loc, value, valueField);
    mlir::Value unk =
        builder.createOrFold<hw::StructExtractOp>(loc, value, unknownField);
    mlir::Value known = builder.createOrFold<comb::XorOp>(loc, unk, ones);
    mlir::Value enabledKnown =
        builder.createOrFold<comb::AndOp>(loc, known, enableVec);
    mlir::Value enabledUnk =
        builder.createOrFold<comb::AndOp>(loc, unk, enableVec);
    mlir::Value enabledVal =
        builder.createOrFold<comb::AndOp>(loc, val, enabledKnown);
    knownOnes = builder.createOrFold<comb::OrOp>(loc, knownOnes, enabledVal);

    mlir::Value valNot = builder.createOrFold<comb::XorOp>(loc, val, ones);
    mlir::Value enabledZero =
        builder.createOrFold<comb::AndOp>(loc, valNot, enabledKnown);
    knownZeros =
        builder.createOrFold<comb::OrOp>(loc, knownZeros, enabledZero);

    unknownOut = comb::OrOp::create(builder, loc,
                                    llvm::ArrayRef<mlir::Value>{
                                        unknownOut, enabledUnk},
                                    /*twoState=*/false);
  }

  mlir::Value conflict =
      builder.createOrFold<comb::AndOp>(loc, knownOnes, knownZeros);
  unknownOut =
      comb::OrOp::create(builder, loc,
                         llvm::ArrayRef<mlir::Value>{unknownOut, conflict},
                         /*twoState=*/false);

  mlir::Value anyEnabledVec =
      comb::ReplicateOp::create(builder, loc, anyEnabled, width);
  mlir::Value notAnyEnabled =
      builder.createOrFold<comb::XorOp>(loc, anyEnabledVec, ones);
  unknownOut = comb::OrOp::create(
      builder, loc,
      llvm::ArrayRef<mlir::Value>{unknownOut, notAnyEnabled},
      /*twoState=*/false);
  mlir::Value valueOut =
      builder.createOrFold<comb::AndOp>(loc, knownOnes, anyEnabledVec);

  return builder.createOrFold<hw::StructCreateOp>(
      loc, structTy, mlir::ValueRange{valueOut, unknownOut});
}

inline mlir::Value resolveFourStateValuesWithStrength(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> values, llvm::ArrayRef<mlir::Value> enables,
    llvm::ArrayRef<unsigned> strength0, llvm::ArrayRef<unsigned> strength1,
    unsigned highZStrength = 4) {
  if (values.empty())
    return mlir::Value();
  if (values.size() != enables.size() || values.size() != strength0.size() ||
      values.size() != strength1.size())
    return mlir::Value();
  auto widthOpt = getFourStateValueWidth(values.front().getType());
  if (!widthOpt)
    return mlir::Value();
  unsigned width = *widthOpt;
  for (mlir::Value value : values) {
    if (value.getType() != values.front().getType())
      return mlir::Value();
  }
  for (mlir::Value enable : enables) {
    auto intTy = llvm::dyn_cast<mlir::IntegerType>(enable.getType());
    if (!intTy || intTy.getWidth() != 1)
      return mlir::Value();
  }

  auto structTy = llvm::cast<hw::StructType>(values.front().getType());
  auto valueField = structTy.getElements()[0].name;
  auto unknownField = structTy.getElements()[1].name;

  mlir::Value zeros = hw::ConstantOp::create(builder, loc,
                                             llvm::APInt::getZero(width))
                          .getResult();
  mlir::Value ones = hw::ConstantOp::create(builder, loc,
                                            llvm::APInt::getAllOnes(width))
                         .getResult();

  constexpr unsigned kStrengthLevels = 4;
  llvm::SmallVector<mlir::Value, kStrengthLevels> onesByStrength(
      kStrengthLevels, zeros);
  llvm::SmallVector<mlir::Value, kStrengthLevels> zerosByStrength(
      kStrengthLevels, zeros);
  llvm::SmallVector<mlir::Value, kStrengthLevels> unknownByStrength(
      kStrengthLevels, zeros);

  auto addToBucket = [&](llvm::SmallVectorImpl<mlir::Value> &buckets,
                         unsigned strength, mlir::Value mask) {
    if (strength >= kStrengthLevels || strength == highZStrength)
      return;
    buckets[strength] =
        builder.createOrFold<comb::OrOp>(loc, buckets[strength], mask);
  };

  for (auto [value, enable, s0, s1] :
       llvm::zip(values, enables, strength0, strength1)) {
    mlir::Value enableVec =
        comb::ReplicateOp::create(builder, loc, enable, width);
    mlir::Value val =
        builder.createOrFold<hw::StructExtractOp>(loc, value, valueField);
    mlir::Value unk =
        builder.createOrFold<hw::StructExtractOp>(loc, value, unknownField);
    mlir::Value known = builder.createOrFold<comb::XorOp>(loc, unk, ones);
    mlir::Value knownEnabled =
        builder.createOrFold<comb::AndOp>(loc, known, enableVec);
    mlir::Value onesMask =
        builder.createOrFold<comb::AndOp>(loc, val, knownEnabled);
    mlir::Value valNot = builder.createOrFold<comb::XorOp>(loc, val, ones);
    mlir::Value zerosMask =
        builder.createOrFold<comb::AndOp>(loc, valNot, knownEnabled);
    mlir::Value unknownMask =
        builder.createOrFold<comb::AndOp>(loc, unk, enableVec);

    addToBucket(onesByStrength, s1, onesMask);
    addToBucket(zerosByStrength, s0, zerosMask);

    if (s0 == s1) {
      addToBucket(unknownByStrength, s0, unknownMask);
    } else {
      addToBucket(unknownByStrength, s0, unknownMask);
      addToBucket(unknownByStrength, s1, unknownMask);
    }
  }

  mlir::Value unresolved = ones;
  mlir::Value valueOut = zeros;
  mlir::Value unknownOut = zeros;

  auto bitNot = [&](mlir::Value v) {
    return builder.createOrFold<comb::XorOp>(loc, v, ones);
  };

  for (unsigned strength = 0; strength < kStrengthLevels; ++strength) {
    mlir::Value onesMask = onesByStrength[strength];
    mlir::Value zerosMask = zerosByStrength[strength];
    mlir::Value unkMask = unknownByStrength[strength];
    mlir::Value driveMask =
        comb::OrOp::create(builder, loc,
                           llvm::ArrayRef<mlir::Value>{onesMask, zerosMask,
                                                       unkMask},
                           /*twoState=*/false);
    mlir::Value conflict =
        comb::OrOp::create(builder, loc,
                           llvm::ArrayRef<mlir::Value>{
                               builder.createOrFold<comb::AndOp>(loc, onesMask,
                                                                 zerosMask),
                               unkMask},
                           /*twoState=*/false);
    mlir::Value conflictActive =
        builder.createOrFold<comb::AndOp>(loc, conflict, unresolved);
    unknownOut = comb::OrOp::create(
        builder, loc,
        llvm::ArrayRef<mlir::Value>{unknownOut, conflictActive},
        /*twoState=*/false);

    mlir::Value onesNoConflict =
        builder.createOrFold<comb::AndOp>(loc, onesMask, bitNot(conflict));
    mlir::Value resolvedOnes =
        builder.createOrFold<comb::AndOp>(loc, onesNoConflict, unresolved);
    valueOut =
        builder.createOrFold<comb::OrOp>(loc, valueOut, resolvedOnes);

    unresolved =
        builder.createOrFold<comb::AndOp>(loc, unresolved, bitNot(driveMask));
  }

  unknownOut = comb::OrOp::create(
      builder, loc, llvm::ArrayRef<mlir::Value>{unknownOut, unresolved},
      /*twoState=*/false);
  return builder.createOrFold<hw::StructCreateOp>(
      loc, structTy, mlir::ValueRange{valueOut, unknownOut});
}

} // namespace circt

#endif // CIRCT_SUPPORT_FOURSTATEUTILS_H
