//===- TwoStateUtils.h - 2-state helpers -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_TWOSTATEUTILS_H
#define CIRCT_SUPPORT_TWOSTATEUTILS_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/APInt.h"
#include <optional>

namespace circt {

inline std::optional<unsigned> getTwoStateValueWidth(mlir::Type type) {
  auto width = hw::getBitWidth(type);
  if (width <= 0)
    return std::nullopt;
  return static_cast<unsigned>(width);
}

inline mlir::Value resolveTwoStateValuesWithEnable(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> values,
    llvm::ArrayRef<mlir::Value> enables, mlir::Value unknownValue) {
  if (values.empty() || values.size() != enables.size())
    return mlir::Value();
  mlir::Type valueType = values.front().getType();
  auto widthOpt = getTwoStateValueWidth(valueType);
  if (!widthOpt)
    return mlir::Value();
  if (unknownValue.getType() != valueType)
    return mlir::Value();
  for (mlir::Value value : values)
    if (value.getType() != valueType)
      return mlir::Value();
  for (mlir::Value enable : enables) {
    auto intTy = llvm::dyn_cast<mlir::IntegerType>(enable.getType());
    if (!intTy || intTy.getWidth() != 1)
      return mlir::Value();
  }

  unsigned width = *widthOpt;
  mlir::Value zeros = hw::ConstantOp::create(builder, loc,
                                             llvm::APInt::getZero(width))
                          .getResult();
  mlir::Value ones = hw::ConstantOp::create(builder, loc,
                                            llvm::APInt::getAllOnes(width))
                         .getResult();
  mlir::Value knownOnes = zeros;
  mlir::Value knownZeros = zeros;
  mlir::Value anyEnabled =
      hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0).getResult();

  for (auto [value, enable] : llvm::zip(values, enables)) {
    mlir::Value enableVec =
        builder.createOrFold<comb::ReplicateOp>(loc, enable, width);
    anyEnabled =
        builder.createOrFold<comb::OrOp>(loc, anyEnabled, enable, true);

    mlir::Value enabledOnes =
        builder.createOrFold<comb::AndOp>(loc, value, enableVec, true);
    mlir::Value valueNot =
        builder.createOrFold<comb::XorOp>(loc, value, ones, true);
    mlir::Value enabledZeros =
        builder.createOrFold<comb::AndOp>(loc, valueNot, enableVec, true);

    knownOnes =
        builder.createOrFold<comb::OrOp>(loc, knownOnes, enabledOnes, true);
    knownZeros =
        builder.createOrFold<comb::OrOp>(loc, knownZeros, enabledZeros, true);
  }

  mlir::Value conflict =
      builder.createOrFold<comb::AndOp>(loc, knownOnes, knownZeros, true);
  mlir::Value anyEnabledVec =
      builder.createOrFold<comb::ReplicateOp>(loc, anyEnabled, width);
  mlir::Value notAnyEnabled =
      builder.createOrFold<comb::XorOp>(loc, anyEnabledVec, ones, true);
  mlir::Value unknownMask = builder.createOrFold<comb::OrOp>(
      loc, conflict, notAnyEnabled, true);
  mlir::Value notUnknown =
      builder.createOrFold<comb::XorOp>(loc, unknownMask, ones, true);

  mlir::Value maskedResolved =
      builder.createOrFold<comb::AndOp>(loc, knownOnes, notUnknown, true);
  mlir::Value maskedUnknown =
      builder.createOrFold<comb::AndOp>(loc, unknownValue, unknownMask, true);
  return builder.createOrFold<comb::OrOp>(
      loc, llvm::ArrayRef<mlir::Value>{maskedResolved, maskedUnknown}, true);
}

inline mlir::Value resolveTwoStateValuesWithStrength(
    mlir::OpBuilder &builder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> values,
    llvm::ArrayRef<mlir::Value> enables, llvm::ArrayRef<unsigned> strength0,
    llvm::ArrayRef<unsigned> strength1, mlir::Value unknownValue,
    unsigned highZStrength = 4) {
  if (values.empty())
    return mlir::Value();
  if (values.size() != enables.size() || values.size() != strength0.size() ||
      values.size() != strength1.size())
    return mlir::Value();
  mlir::Type valueType = values.front().getType();
  auto widthOpt = getTwoStateValueWidth(valueType);
  if (!widthOpt)
    return mlir::Value();
  if (unknownValue.getType() != valueType)
    return mlir::Value();
  for (mlir::Value value : values) {
    if (value.getType() != values.front().getType())
      return mlir::Value();
  }
  for (mlir::Value enable : enables) {
    auto intTy = llvm::dyn_cast<mlir::IntegerType>(enable.getType());
    if (!intTy || intTy.getWidth() != 1)
      return mlir::Value();
  }

  unsigned width = *widthOpt;
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

  auto addToBucket = [&](llvm::SmallVectorImpl<mlir::Value> &buckets,
                         unsigned strength, mlir::Value mask) {
    if (strength >= kStrengthLevels || strength == highZStrength)
      return;
    buckets[strength] = builder.createOrFold<comb::OrOp>(
        loc, buckets[strength], mask, true);
  };

  for (auto [value, enable, s0, s1] :
       llvm::zip(values, enables, strength0, strength1)) {
    mlir::Value enableVec =
        builder.createOrFold<comb::ReplicateOp>(loc, enable, width);
    mlir::Value onesMask =
        builder.createOrFold<comb::AndOp>(loc, value, enableVec, true);
    mlir::Value valueNot =
        builder.createOrFold<comb::XorOp>(loc, value, ones, true);
    mlir::Value zerosMask =
        builder.createOrFold<comb::AndOp>(loc, valueNot, enableVec, true);

    addToBucket(onesByStrength, s1, onesMask);
    addToBucket(zerosByStrength, s0, zerosMask);
  }

  mlir::Value unresolved = ones;
  mlir::Value valueOut = zeros;
  mlir::Value unknownMask = zeros;

  auto bitNot = [&](mlir::Value v) {
    return builder.createOrFold<comb::XorOp>(loc, v, ones, true);
  };

  for (unsigned strength = 0; strength < kStrengthLevels; ++strength) {
    mlir::Value onesMask = onesByStrength[strength];
    mlir::Value zerosMask = zerosByStrength[strength];
    mlir::Value driveMask = builder.createOrFold<comb::OrOp>(
        loc, llvm::ArrayRef<mlir::Value>{onesMask, zerosMask}, true);
    mlir::Value conflict =
        builder.createOrFold<comb::AndOp>(loc, onesMask, zerosMask, true);
    mlir::Value conflictActive =
        builder.createOrFold<comb::AndOp>(loc, conflict, unresolved, true);
    unknownMask = builder.createOrFold<comb::OrOp>(loc, unknownMask,
                                                   conflictActive, true);

    mlir::Value onesNoConflict = builder.createOrFold<comb::AndOp>(
        loc, onesMask, bitNot(conflict), true);
    mlir::Value resolvedOnes = builder.createOrFold<comb::AndOp>(
        loc, onesNoConflict, unresolved, true);
    valueOut =
        builder.createOrFold<comb::OrOp>(loc, valueOut, resolvedOnes, true);

    unresolved = builder.createOrFold<comb::AndOp>(
        loc, unresolved, bitNot(driveMask), true);
  }

  unknownMask =
      builder.createOrFold<comb::OrOp>(loc, unknownMask, unresolved, true);
  mlir::Value notUnknown = bitNot(unknownMask);
  mlir::Value maskedResolved =
      builder.createOrFold<comb::AndOp>(loc, valueOut, notUnknown, true);
  mlir::Value maskedUnknown = builder.createOrFold<comb::AndOp>(
      loc, unknownValue, unknownMask, true);
  return builder.createOrFold<comb::OrOp>(
      loc, llvm::ArrayRef<mlir::Value>{maskedResolved, maskedUnknown}, true);
}

} // namespace circt

#endif // CIRCT_SUPPORT_TWOSTATEUTILS_H
