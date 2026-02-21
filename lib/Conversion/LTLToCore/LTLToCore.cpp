//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LTLSequenceNFA.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <optional>

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
constexpr const char kWeakEventuallyAttr[] = "ltl.weak";

using ltl::NFABuilder;
struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = seq::createConstantInitialValue(
        rewriter, op->getLoc(), rewriter.getIntegerAttr(i1, 0));

    // Generate the constant used to negate the reset value
    Value constOne = hw::ConstantOp::create(rewriter, op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        comb::OrOp::create(rewriter, op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(seq::CompRegOp::create(
        rewriter, op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset = comb::XorOp::create(rewriter, op.getLoc(),
                                         adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

struct PropertyResult {
  Value safety;
  Value finalCheck;
};

constexpr const char kDisableIffAttr[] = "sva.disable_iff";

struct LTLPropertyLowerer {
  OpBuilder &builder;
  Location loc;
  Value disable;
  DenseMap<Value, Value> posedgeTicks;
  DenseMap<Value, Value> negedgeTicks;
  DenseMap<Value, Value> bothedgeTicks;
  // For assumes, we skip the warmup period since constraints should apply from
  // cycle 0. Assertions need warmup to avoid false failures during sequence
  // startup. This matches Yosys behavior with `-early -assume`.
  bool skipWarmup;

  LTLPropertyLowerer(OpBuilder &builder, Location loc, bool skipWarmup = false)
      : builder(builder), loc(loc), disable(), skipWarmup(skipWarmup) {}

  Value getClockTick(Value clockSignal, ltl::ClockEdge edge, Value outerClock) {
    if (!outerClock)
      return {};

    if (isa<seq::ClockType>(clockSignal.getType()))
      clockSignal = seq::FromClockOp::create(builder, loc, clockSignal);

    DenseMap<Value, Value> *cache = nullptr;
    switch (edge) {
    case ltl::ClockEdge::Pos:
      cache = &posedgeTicks;
      break;
    case ltl::ClockEdge::Neg:
      cache = &negedgeTicks;
      break;
    case ltl::ClockEdge::Both:
      cache = &bothedgeTicks;
      break;
    }
    if (auto it = cache->find(clockSignal); it != cache->end())
      return it->second;

    auto one = hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    if (auto toClock = outerClock.getDefiningOp<seq::ToClockOp>()) {
      Value outerSignal = toClock.getInput();
      if (edge == ltl::ClockEdge::Pos && outerSignal == clockSignal)
        return one;
      if (edge == ltl::ClockEdge::Neg) {
        if (auto xorOp = outerSignal.getDefiningOp<comb::XorOp>()) {
          auto inputs = xorOp.getInputs();
          if (inputs.size() == 2) {
            auto isOne = [](Value value) {
              if (auto cst = value.getDefiningOp<hw::ConstantOp>())
                return cst.getValue().isOne();
              return false;
            };
            auto lhs = inputs[0];
            auto rhs = inputs[1];
            if ((lhs == clockSignal && isOne(rhs)) ||
                (rhs == clockSignal && isOne(lhs)))
              return one;
          }
        }
      }
    }
    auto prev = shiftValue(clockSignal, 1, outerClock);
    Value tick;
    switch (edge) {
    case ltl::ClockEdge::Pos: {
      auto notPrev = comb::XorOp::create(builder, loc, prev, one);
      tick = comb::AndOp::create(builder, loc,
                                 SmallVector<Value, 2>{clockSignal, notPrev},
                                 true);
      break;
    }
    case ltl::ClockEdge::Neg: {
      auto notSignal = comb::XorOp::create(builder, loc, clockSignal, one);
      tick = comb::AndOp::create(builder, loc,
                                 SmallVector<Value, 2>{notSignal, prev}, true);
      break;
    }
    case ltl::ClockEdge::Both:
      tick = comb::XorOp::create(builder, loc, clockSignal, prev);
      break;
    }

    (*cache)[clockSignal] = tick;
    return tick;
  }

  std::optional<uint64_t> getSequenceMaxLength(Value seq) {
    if (!seq)
      return 0;
    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>())
      return getSequenceMaxLength(clockOp.getInput());
    if (auto pastOp = seq.getDefiningOp<ltl::PastOp>())
      return getSequenceMaxLength(pastOp.getInput());
    if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
      auto innerMax = getSequenceMaxLength(delayOp.getInput());
      if (!innerMax)
        return std::nullopt;
      uint64_t maxDelay = delayOp.getDelay();
      if (auto length = delayOp.getLength())
        maxDelay += *length;
      return *innerMax + maxDelay;
    }
    if (!isa<ltl::SequenceType>(seq.getType()))
      return 1;
    if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
      uint64_t total = 0;
      for (auto input : concatOp.getInputs()) {
        auto maxLen = getSequenceMaxLength(input);
        if (!maxLen)
          return std::nullopt;
        total += *maxLen;
      }
      return total;
    }
    if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
      auto innerMax = getSequenceMaxLength(repeatOp.getInput());
      if (!innerMax)
        return std::nullopt;
      if (!repeatOp.getMore())
        return std::nullopt;
      uint64_t maxCount = repeatOp.getBase() + *repeatOp.getMore();
      return *innerMax * maxCount;
    }
    if (seq.getDefiningOp<ltl::GoToRepeatOp>())
      return std::nullopt;
    if (seq.getDefiningOp<ltl::NonConsecutiveRepeatOp>())
      return std::nullopt;
    if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
      uint64_t maxLen = 0;
      for (auto input : orOp.getInputs()) {
        auto maxInput = getSequenceMaxLength(input);
        if (!maxInput)
          return std::nullopt;
        maxLen = std::max(maxLen, *maxInput);
      }
      return maxLen;
    }
    if (auto andOp = seq.getDefiningOp<ltl::AndOp>()) {
      uint64_t maxLen = 0;
      for (auto input : andOp.getInputs()) {
        auto maxInput = getSequenceMaxLength(input);
        if (!maxInput)
          return std::nullopt;
        maxLen = std::max(maxLen, *maxInput);
      }
      return maxLen;
    }
    if (auto intersectOp = seq.getDefiningOp<ltl::IntersectOp>()) {
      std::optional<uint64_t> maxLen;
      for (auto input : intersectOp.getInputs()) {
        auto maxInput = getSequenceMaxLength(input);
        if (!maxInput)
          continue;
        if (!maxLen || *maxInput < *maxLen)
          maxLen = *maxInput;
      }
      return maxLen;
    }
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
      return getSequenceMaxLength(firstMatch.getInput());

    return std::nullopt;
  }

  std::optional<std::pair<uint64_t, uint64_t>>
  getSequenceLengthBounds(Value seq) {
    if (!seq)
      return std::make_pair<uint64_t, uint64_t>(0, 0);
    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>())
      return getSequenceLengthBounds(clockOp.getInput());
    if (auto pastOp = seq.getDefiningOp<ltl::PastOp>())
      return getSequenceLengthBounds(pastOp.getInput());
    if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
      auto inputBounds = getSequenceLengthBounds(delayOp.getInput());
      if (!inputBounds)
        return std::nullopt;
      uint64_t minDelay = delayOp.getDelay();
      if (auto length = delayOp.getLength()) {
        uint64_t maxDelay = minDelay + *length;
        return std::make_pair(inputBounds->first + minDelay,
                              inputBounds->second + maxDelay);
      }
      return std::nullopt;
    }
    if (!isa<ltl::SequenceType>(seq.getType()))
      return std::make_pair<uint64_t, uint64_t>(1, 1);
    if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
      uint64_t minLen = 0;
      uint64_t maxLen = 0;
      for (auto input : concatOp.getInputs()) {
        auto bounds = getSequenceLengthBounds(input);
        if (!bounds)
          return std::nullopt;
        minLen += bounds->first;
        maxLen += bounds->second;
      }
      return std::make_pair(minLen, maxLen);
    }
    if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
      auto more = repeatOp.getMore();
      if (!more)
        return std::nullopt;
      auto bounds = getSequenceLengthBounds(repeatOp.getInput());
      if (!bounds)
        return std::nullopt;
      uint64_t minLen = bounds->first * repeatOp.getBase();
      uint64_t maxLen = bounds->second * (repeatOp.getBase() + *more);
      return std::make_pair(minLen, maxLen);
    }
    if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
      std::optional<std::pair<uint64_t, uint64_t>> result;
      for (auto input : orOp.getInputs()) {
        auto bounds = getSequenceLengthBounds(input);
        if (!bounds)
          return std::nullopt;
        if (!result) {
          result = *bounds;
        } else {
          result->first = std::min(result->first, bounds->first);
          result->second = std::max(result->second, bounds->second);
        }
      }
      return result;
    }
    if (auto andOp = seq.getDefiningOp<ltl::AndOp>()) {
      std::optional<std::pair<uint64_t, uint64_t>> result;
      for (auto input : andOp.getInputs()) {
        auto bounds = getSequenceLengthBounds(input);
        if (!bounds)
          return std::nullopt;
        if (!result) {
          result = *bounds;
        } else {
          result->first = std::max(result->first, bounds->first);
          result->second = std::min(result->second, bounds->second);
        }
      }
      if (result && result->first > result->second)
        return std::nullopt;
      return result;
    }
    if (auto intersectOp = seq.getDefiningOp<ltl::IntersectOp>()) {
      std::optional<uint64_t> minLen;
      std::optional<uint64_t> maxLen;
      for (auto input : intersectOp.getInputs()) {
        auto bounds = getSequenceLengthBounds(input);
        if (!bounds)
          return std::nullopt;
        if (!minLen) {
          minLen = bounds->first;
          maxLen = bounds->second;
        } else {
          minLen = std::max(*minLen, bounds->first);
          maxLen = std::min(*maxLen, bounds->second);
        }
      }
      if (!minLen || !maxLen || *minLen > *maxLen)
        return std::nullopt;
      return std::make_pair(*minLen, *maxLen);
    }
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
      return getSequenceLengthBounds(firstMatch.getInput());

    return std::nullopt;
  }

  std::optional<uint64_t> getSequenceMinLength(Value seq) {
    if (!seq)
      return 0;
    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>())
      return getSequenceMinLength(clockOp.getInput());
    if (auto pastOp = seq.getDefiningOp<ltl::PastOp>())
      return getSequenceMinLength(pastOp.getInput());
    if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
      auto inputMin = getSequenceMinLength(delayOp.getInput());
      if (!inputMin)
        return std::nullopt;
      return *inputMin + delayOp.getDelay();
    }
    if (!isa<ltl::SequenceType>(seq.getType()))
      return 1;
    if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
      uint64_t minLen = 0;
      for (auto input : concatOp.getInputs()) {
        auto inputMin = getSequenceMinLength(input);
        if (!inputMin)
          return std::nullopt;
        minLen += *inputMin;
      }
      return minLen;
    }
    if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
      auto inputMin = getSequenceMinLength(repeatOp.getInput());
      if (!inputMin)
        return std::nullopt;
      return *inputMin * repeatOp.getBase();
    }
    if (auto gotoOp = seq.getDefiningOp<ltl::GoToRepeatOp>()) {
      auto inputMin = getSequenceMinLength(gotoOp.getInput());
      if (!inputMin)
        return std::nullopt;
      return *inputMin * gotoOp.getBase();
    }
    if (auto nonConsecutiveOp = seq.getDefiningOp<ltl::NonConsecutiveRepeatOp>()) {
      auto inputMin = getSequenceMinLength(nonConsecutiveOp.getInput());
      if (!inputMin)
        return std::nullopt;
      return *inputMin * nonConsecutiveOp.getBase();
    }
    if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
      std::optional<uint64_t> minLen;
      for (auto input : orOp.getInputs()) {
        auto inputMin = getSequenceMinLength(input);
        if (!inputMin)
          return std::nullopt;
        if (!minLen || *inputMin < *minLen)
          minLen = *inputMin;
      }
      return minLen;
    }
    if (auto andOp = seq.getDefiningOp<ltl::AndOp>()) {
      std::optional<uint64_t> minLen;
      for (auto input : andOp.getInputs()) {
        auto inputMin = getSequenceMinLength(input);
        if (!inputMin)
          return std::nullopt;
        if (!minLen || *inputMin > *minLen)
          minLen = *inputMin;
      }
      return minLen;
    }
    if (auto intersectOp = seq.getDefiningOp<ltl::IntersectOp>()) {
      std::optional<uint64_t> minLen;
      for (auto input : intersectOp.getInputs()) {
        auto inputMin = getSequenceMinLength(input);
        if (!inputMin)
          return std::nullopt;
        if (!minLen || *inputMin > *minLen)
          minLen = *inputMin;
      }
      return minLen;
    }
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
      return getSequenceMinLength(firstMatch.getInput());
    return std::nullopt;
  }

  Value shiftAges(Value input, Value zeroBit, unsigned width, Value zeroBits) {
    if (width == 1)
      return zeroBits;
    auto lowBits = comb::ExtractOp::create(builder, loc, input, 0, width - 1);
    return comb::ConcatOp::create(builder, loc,
                                  SmallVector<Value, 2>{lowBits, zeroBit});
  }

  std::pair<Value, Value> getResetPair(Value resetVal) {
    if (disable)
      return {disable, resetVal};
    return {Value(), Value()};
  }

  std::optional<bool> getI1Constant(Value value) {
    if (!value)
      return std::nullopt;
    if (auto hwConst = value.getDefiningOp<hw::ConstantOp>()) {
      APInt bits = hwConst.getValue();
      if (bits.getBitWidth() == 1)
        return bits.isOne();
    }
    if (auto arithConst = value.getDefiningOp<arith::ConstantOp>()) {
      if (auto boolAttr = dyn_cast<BoolAttr>(arithConst.getValue()))
        return boolAttr.getValue();
      if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
        if (intAttr.getType().isInteger(1))
          return intAttr.getValue().isOne();
      }
    }
    return std::nullopt;
  }

  PropertyResult lowerDisableIff(ltl::OrOp orOp, Value clock,
                                 ltl::ClockEdge edge) {
    auto inputs = orOp.getInputs();
    if (inputs.size() != 2) {
      orOp.emitError("disable iff expects two inputs");
      return {Value(), {}};
    }
    Value disableInput;
    Value propertyInput;
    auto type0 = inputs[0].getType();
    auto type1 = inputs[1].getType();
    bool input0IsBool = type0.isInteger(1);
    bool input1IsBool = type1.isInteger(1);
    bool input0IsProp = isa<ltl::PropertyType, ltl::SequenceType>(type0);
    bool input1IsProp = isa<ltl::PropertyType, ltl::SequenceType>(type1);
    if (input0IsBool && input1IsProp) {
      disableInput = inputs[0];
      propertyInput = inputs[1];
    } else if (input1IsBool && input0IsProp) {
      disableInput = inputs[1];
      propertyInput = inputs[0];
    } else {
      orOp.emitError("disable iff expects i1 and property inputs");
      return {Value(), {}};
    }

    Value savedDisable = disable;
    if (auto disableConst = getI1Constant(disableInput)) {
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      if (*disableConst)
        return {trueVal, trueVal};
    }
    Value combinedDisable = disableInput;
    if (disable) {
      combinedDisable = comb::OrOp::create(
          builder, loc, SmallVector<Value, 2>{disable, disableInput}, true);
    }
    disable = combinedDisable;
    auto result = lowerProperty(propertyInput, clock, edge);
    disable = savedDisable;
    if (!result.safety || !result.finalCheck) {
      orOp.emitError("invalid property lowering");
      return {Value(), {}};
    }

    auto safety = comb::OrOp::create(
        builder, loc, SmallVector<Value, 2>{disableInput, result.safety},
        true);
    auto finalCheck = comb::OrOp::create(
        builder, loc, SmallVector<Value, 2>{disableInput, result.finalCheck},
        true);
    return {safety, finalCheck};
  }

  Value lowerFirstMatchSequence(Value seq, Value clock, ltl::ClockEdge edge,
                                uint64_t maxLen) {
    static_cast<void>(edge);
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }
    if (maxLen == 0) {
      return hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    NFABuilder nfa(trueVal);
    auto fragment = nfa.build(seq, loc, builder);
    nfa.eliminateEpsilon();

    size_t numStates = nfa.states.size();
    if (numStates == 0)
      return trueVal;

    unsigned ageWidth = maxLen + 1;
    auto ageType = builder.getIntegerType(ageWidth);
    auto zeroBits =
        hw::ConstantOp::create(builder, loc, APInt(ageWidth, 0));
    auto oneBits =
        hw::ConstantOp::create(builder, loc, APInt(ageWidth, 1));
    auto allOnes =
        hw::ConstantOp::create(builder, loc, APInt::getAllOnes(ageWidth));
    auto zeroBit =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);

    SmallVector<Value, 8> stateRegs;
    SmallVector<Backedge, 8> nextStates;
    BackedgeBuilder bb(builder, loc);
    for (size_t i = 0; i < numStates; ++i) {
      auto next = bb.get(ageType);
      nextStates.push_back(next);
      auto powerOn =
          seq::createConstantInitialValue(builder, zeroBits.getOperation());
      auto [reset, resetVal] = getResetPair(zeroBits);
      auto reg = seq::CompRegOp::create(builder, loc, next, clock, reset,
                                        resetVal,
                                        builder.getStringAttr("ltl_state"),
                                        powerOn);
      stateRegs.push_back(reg);
    }

    SmallVector<Value, 8> currentStates(stateRegs.begin(), stateRegs.end());
    currentStates[fragment.start] =
        comb::OrOp::create(builder, loc,
                           SmallVector<Value, 2>{currentStates[fragment.start],
                                                 oneBits},
                           true);

    SmallVector<SmallVector<Value, 8>, 8> nextInputs(numStates);
    for (size_t from = 0; from < numStates; ++from) {
      auto shifted =
          shiftAges(currentStates[from], zeroBit, ageWidth, zeroBits);
      DenseMap<size_t, Value> maskedByCond;
      for (auto &tr : nfa.states[from].transitions) {
        if (tr.isEpsilon)
          continue;
        auto it = maskedByCond.find(tr.condIndex);
        Value masked;
        if (it != maskedByCond.end()) {
          masked = it->second;
        } else {
          auto mask = comb::ReplicateOp::create(builder, loc,
                                                nfa.conditions[tr.condIndex],
                                                ageWidth);
          masked = comb::AndOp::create(builder, loc, shifted, mask);
          maskedByCond.insert({tr.condIndex, masked});
        }
        nextInputs[tr.to].push_back(masked);
      }
    }

    SmallVector<Value, 8> nextVals(numStates, zeroBits);
    for (size_t i = 0; i < numStates; ++i) {
      if (nextInputs[i].empty())
        continue;
      if (nextInputs[i].size() == 1) {
        nextVals[i] = nextInputs[i].front();
        continue;
      }
      nextVals[i] = comb::OrOp::create(builder, loc, nextInputs[i], true);
    }

    SmallVector<Value, 8> accepting;
    for (size_t i = 0; i < numStates; ++i) {
      if (nfa.states[i].accepting)
        accepting.push_back(nextVals[i]);
    }
    if (accepting.empty())
      return zeroBit;
    Value matchBits = accepting.front();
    if (accepting.size() > 1)
      matchBits = comb::OrOp::create(builder, loc, accepting, true);
    auto match =
        comb::ICmpOp::create(builder, loc, comb::ICmpPredicate::ne, matchBits,
                             zeroBits);
    auto notMatchBits =
        comb::XorOp::create(builder, loc, matchBits, allOnes);
    for (size_t i = 0; i < numStates; ++i) {
      auto kept = comb::AndOp::create(builder, loc, nextVals[i], notMatchBits);
      nextStates[i].setValue(kept);
    }

    return match;
  }

  Value lowerFirstMatchSequenceUnbounded(Value seq, Value clock,
                                         ltl::ClockEdge edge) {
    static_cast<void>(edge);
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    auto falseVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    NFABuilder nfa(trueVal);
    auto fragment = nfa.build(seq, loc, builder);
    nfa.eliminateEpsilon();

    size_t numStates = nfa.states.size();
    if (numStates == 0)
      return trueVal;

    SmallVector<Value, 8> stateRegs;
    SmallVector<Backedge, 8> nextStates;
    BackedgeBuilder bb(builder, loc);
    for (size_t i = 0; i < numStates; ++i) {
      auto next = bb.get(builder.getI1Type());
      nextStates.push_back(next);
      auto powerOn =
          seq::createConstantInitialValue(builder, falseVal.getOperation());
      auto [reset, resetVal] = getResetPair(falseVal);
      auto reg = seq::CompRegOp::create(builder, loc, next, clock, reset,
                                        resetVal,
                                        builder.getStringAttr("ltl_state"),
                                        powerOn);
      stateRegs.push_back(reg);
    }

    SmallVector<Value, 8> currentStates(stateRegs.begin(), stateRegs.end());
    currentStates[fragment.start] =
        comb::OrOp::create(builder, loc,
                           SmallVector<Value, 2>{currentStates[fragment.start],
                                                 trueVal},
                           true);

    SmallVector<SmallVector<Value, 8>, 8> nextInputs(numStates);
    for (size_t from = 0; from < numStates; ++from) {
      DenseMap<size_t, Value> maskedByCond;
      for (auto &tr : nfa.states[from].transitions) {
        if (tr.isEpsilon)
          continue;
        auto it = maskedByCond.find(tr.condIndex);
        Value masked;
        if (it != maskedByCond.end()) {
          masked = it->second;
        } else {
          masked = comb::AndOp::create(
              builder, loc,
              SmallVector<Value, 2>{currentStates[from],
                                    nfa.conditions[tr.condIndex]},
              true);
          maskedByCond.insert({tr.condIndex, masked});
        }
        nextInputs[tr.to].push_back(masked);
      }
    }

    SmallVector<Value, 8> nextVals(numStates, falseVal);
    for (size_t i = 0; i < numStates; ++i) {
      if (nextInputs[i].empty())
        continue;
      if (nextInputs[i].size() == 1) {
        nextVals[i] = nextInputs[i].front();
        continue;
      }
      nextVals[i] = comb::OrOp::create(builder, loc, nextInputs[i], true);
    }

    SmallVector<Value, 8> accepting;
    for (size_t i = 0; i < numStates; ++i) {
      if (nfa.states[i].accepting)
        accepting.push_back(nextVals[i]);
    }
    if (accepting.empty())
      return falseVal;
    Value match = accepting.front();
    if (accepting.size() > 1)
      match = comb::OrOp::create(builder, loc, accepting, true);
    auto notMatch = comb::XorOp::create(builder, loc, match, trueVal);
    for (size_t i = 0; i < numStates; ++i) {
      auto kept = comb::AndOp::create(
          builder, loc, SmallVector<Value, 2>{nextVals[i], notMatch}, true);
      nextStates[i].setValue(kept);
    }
    return match;
  }

  PropertyResult lowerProperty(Value prop, Value clock, ltl::ClockEdge edge) {
    if (!prop)
      return {Value(), {}};
    if (!isa<ltl::PropertyType, ltl::SequenceType>(prop.getType())) {
      // For simple i1 values, the finalCheck is always true (no liveness
      // requirement).
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      return {prop, trueVal};
    }
    if (isa<ltl::SequenceType>(prop.getType())) {
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      // Extract clock from ltl.clock op if present, for warmup computation
      Value warmupClock = clock;
      if (auto clockOp = prop.getDefiningOp<ltl::ClockOp>()) {
        warmupClock = normalizeClock(clockOp.getClock(), clockOp.getEdge());
      }
      auto match = lowerSequence(prop, clock, edge);
      if (!match)
        return {Value(), {}};
      // Skip warmup for assumes - constraints should apply from cycle 0.
      // Only assertions need warmup to avoid false failures during sequence
      // startup.
      if (!skipWarmup && warmupClock) {
        if (auto minLen = getSequenceMinLength(prop); minLen && *minLen > 0) {
          uint64_t shift = *minLen - 1;
          if (shift > 0) {
            auto warmup = shiftValue(trueVal, shift, warmupClock);
            auto notWarmup = comb::XorOp::create(
                builder, loc, warmup,
                hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
            match = comb::OrOp::create(
                builder, loc, SmallVector<Value, 2>{notWarmup, match}, true);
          }
        }
      }
      return {match, trueVal};
    }

    if (auto clockOp = prop.getDefiningOp<ltl::ClockOp>()) {
      auto normalizedClock =
          normalizeClock(clockOp.getClock(), clockOp.getEdge());
      auto result =
          lowerProperty(clockOp.getInput(), normalizedClock, clockOp.getEdge());
      if (!result.safety || !result.finalCheck)
        return {Value(), {}};
      // Apply sampled-value semantics at the top-level clock boundary for
      // properties. Sequences already model their own cycle alignment.
      if (!clock && normalizedClock &&
          isa<ltl::PropertyType>(clockOp.getInput().getType())) {
        if (!getI1Constant(result.safety).value_or(false))
          result.safety = shiftValue(result.safety, 1, normalizedClock);
        if (!getI1Constant(result.finalCheck).value_or(false))
          result.finalCheck = shiftValue(result.finalCheck, 1, normalizedClock);
      }
      return result;
    }

    if (auto constOp = prop.getDefiningOp<ltl::BooleanConstantOp>()) {
      auto value =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                 constOp.getValueAttr().getValue() ? 1 : 0);
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      return {value, trueVal};
    }
    if (auto notOp = prop.getDefiningOp<ltl::NotOp>()) {
      auto inner = lowerProperty(notOp.getInput(), clock, edge);
      if (!inner.safety || !inner.finalCheck) {
        notOp.emitError("invalid property lowering");
        return {Value(), {}};
      }
      Value neg = comb::XorOp::create(
          builder, loc, inner.safety,
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      // Skip warmup for assumes - constraints should apply from cycle 0.
      if (!skipWarmup) {
        if (clock) {
          if (auto minLen = getSequenceMinLength(notOp.getInput());
              minLen && *minLen > 0) {
            uint64_t shift = *minLen - 1;
            if (shift > 0) {
              auto trueVal =
                  hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
              auto warmup = shiftValue(trueVal, shift, clock);
              auto notWarmup = comb::XorOp::create(
                  builder, loc, warmup,
                  hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
              neg = comb::OrOp::create(
                  builder, loc, SmallVector<Value, 2>{notWarmup, neg}, true);
            }
          }
        }
      }
      // For sequences (i1 or !ltl.sequence), the negation is a safety property:
      // the sequence should never complete. The finalCheck should remain true
      // (no liveness requirement). For properties, we also negate finalCheck.
      Value finalCheck;
      auto inputType = notOp.getInput().getType();
      if (isa<ltl::SequenceType>(inputType) || inputType.isInteger(1)) {
        // Negating a sequence is purely a safety property
        finalCheck =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      } else {
        finalCheck = comb::XorOp::create(
            builder, loc, inner.finalCheck,
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      }
      return {neg, finalCheck};
    }
    if (auto andOp = prop.getDefiningOp<ltl::AndOp>()) {
      SmallVector<Value, 4> safeties;
      SmallVector<PropertyResult, 4> results;
      for (auto input : andOp.getInputs()) {
        auto res = lowerProperty(input, clock, edge);
        if (!res.safety || !res.finalCheck) {
          andOp.emitError("invalid property lowering");
          return {Value(), {}};
        }
        safeties.push_back(res.safety);
        results.push_back(res);
      }
      auto safety = comb::AndOp::create(builder, loc, safeties, true);
      Value finalCheck = nullptr;
      for (auto &res : results) {
        if (!finalCheck)
          finalCheck = res.finalCheck;
        else
          finalCheck =
              comb::AndOp::create(builder, loc, finalCheck, res.finalCheck);
      }
      return {safety, finalCheck};
    }
    if (auto orOp = prop.getDefiningOp<ltl::OrOp>()) {
      if (orOp->hasAttr(kDisableIffAttr))
        return lowerDisableIff(orOp, clock, edge);
      SmallVector<Value, 4> safeties;
      SmallVector<PropertyResult, 4> results;
      Value finalCheck = nullptr;
      for (auto input : orOp.getInputs()) {
        auto res = lowerProperty(input, clock, edge);
        if (!res.safety || !res.finalCheck) {
          orOp.emitError("invalid property lowering");
          return {Value(), {}};
        }
        safeties.push_back(res.safety);
        results.push_back(res);
        if (!finalCheck)
          finalCheck = res.finalCheck;
        else
          finalCheck =
              comb::OrOp::create(builder, loc, finalCheck, res.finalCheck);
      }
      auto safety = comb::OrOp::create(builder, loc, safeties, true);
      return {safety, finalCheck};
    }
    if (auto implOp = prop.getDefiningOp<ltl::ImplicationOp>()) {
      Value antecedentSeq = implOp.getAntecedent();
      Value consequentValue = implOp.getConsequent();
      uint64_t shiftDelay = 0;
      if (auto delayOp = consequentValue.getDefiningOp<ltl::DelayOp>()) {
        if (auto length = delayOp.getLength()) {
          if (*length == 0 && delayOp.getDelay() > 0) {
            auto input = delayOp.getInput();
            uint64_t sequenceLen = 1;
            bool canShift = false;
            if (!isa<ltl::SequenceType>(input.getType())) {
              canShift = true;
            } else if (auto bounds = getSequenceLengthBounds(input)) {
              if (bounds->first == bounds->second && bounds->first > 0) {
                sequenceLen = bounds->first;
                canShift = true;
              }
            }
            if (canShift) {
              shiftDelay = delayOp.getDelay() + sequenceLen - 1;
              consequentValue = input;
            }
          }
        }
      }
      auto antecedent = lowerSequence(antecedentSeq, clock, edge);
      if (shiftDelay > 0) {
        if (!clock) {
          implOp.emitError("implication requires a clocked property");
          return {Value(), {}};
        }
        antecedent = shiftValue(antecedent, shiftDelay, clock);
      }
      auto consequent = lowerProperty(consequentValue, clock, edge);
      if (!consequent.finalCheck) {
        implOp.emitError("invalid property lowering");
        return {Value(), {}};
      }
      auto notAntecedent = comb::XorOp::create(
          builder, loc, antecedent,
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      auto safety = builder.createOrFold<comb::OrOp>(
          loc, SmallVector<Value, 2>{notAntecedent, consequent.safety}, true);
      if (!clock) {
        implOp.emitError("implication requires a clocked property");
        return {Value(), {}};
      }
      auto antecedentSeen = createStateRegister(
          antecedent, clock, "ltl_implication_seen");
      auto notSeen = comb::XorOp::create(
          builder, loc, antecedentSeen,
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      auto finalCheck = builder.createOrFold<comb::OrOp>(
          loc, SmallVector<Value, 2>{notSeen, consequent.finalCheck}, true);
      return {safety, finalCheck};
    }
    if (auto untilOp = prop.getDefiningOp<ltl::UntilOp>()) {
      auto input = lowerProperty(untilOp.getInput(), clock, edge);
      auto condition = lowerProperty(untilOp.getCondition(), clock, edge);
      if (!input.finalCheck || !condition.finalCheck) {
        untilOp.emitError("invalid property lowering");
        return {Value(), {}};
      }
      if (!clock) {
        untilOp.emitError("until requires a clocked property");
        return {Value(), {}};
      }
      auto seen = createStateRegister(condition.safety, clock,
                                      "ltl_until_seen");
      auto safety = comb::OrOp::create(
          builder, loc, SmallVector<Value, 2>{seen, input.safety}, true);
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      return {safety, trueVal};
    }
    if (auto eventuallyOp = prop.getDefiningOp<ltl::EventuallyOp>()) {
      auto input = lowerProperty(eventuallyOp.getInput(), clock, edge);
      if (!input.finalCheck) {
        eventuallyOp.emitError("invalid property lowering");
        return {Value(), {}};
      }
      if (!clock) {
        eventuallyOp.emitError("eventually requires a clocked property");
        return {Value(), {}};
      }
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      if (eventuallyOp->hasAttr(kWeakEventuallyAttr))
        return {trueVal, trueVal};
      auto seen =
          createStateRegister(input.safety, clock, "ltl_eventually_seen");
      return {trueVal, seen};
    }

    prop.getDefiningOp()->emitError("unsupported property lowering");
    return {Value(), {}};
  }

  Value lowerSequence(Value seq, Value clock, ltl::ClockEdge edge) {
    if (!seq)
      return {};
    if (!isa<ltl::SequenceType>(seq.getType())) {
      if (!clock)
        return seq;
    }

    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>()) {
      auto normalizedClock =
          normalizeClock(clockOp.getClock(), clockOp.getEdge());
      return lowerSequence(clockOp.getInput(), normalizedClock,
                           clockOp.getEdge());
    }
    if (auto pastOp = seq.getDefiningOp<ltl::PastOp>()) {
      if (!clock) {
        pastOp.emitError("ltl.past requires a clocked sequence");
        return {};
      }
      return shiftValue(pastOp.getInput(), pastOp.getDelay(), clock);
    }
    if (auto andOp = seq.getDefiningOp<ltl::AndOp>()) {
      SmallVector<Value, 4> inputs;
      for (auto input : andOp.getInputs())
        inputs.push_back(lowerSequence(input, clock, edge));
      return comb::AndOp::create(builder, loc, inputs, true);
    }
    if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
      SmallVector<Value, 4> inputs;
      for (auto input : orOp.getInputs())
        inputs.push_back(lowerSequence(input, clock, edge));
      return comb::OrOp::create(builder, loc, inputs, true);
    }
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    auto clockPredicate = [this, clock](Value clockSignal,
                                        ltl::ClockEdge edge) -> Value {
      return getClockTick(clockSignal, edge, clock);
    };
    NFABuilder nfa(trueVal, clockPredicate);
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>()) {
      auto maxLen = getSequenceMaxLength(firstMatch.getInput());
      if (maxLen)
        return lowerFirstMatchSequence(firstMatch.getInput(), clock, edge,
                                       *maxLen);
      return lowerFirstMatchSequenceUnbounded(firstMatch.getInput(), clock,
                                              edge);
    }
    auto fragment = nfa.build(seq, loc, builder);
    nfa.eliminateEpsilon();

    size_t numStates = nfa.states.size();
    if (numStates == 0)
      return trueVal;

    SmallVector<Value, 8> stateRegs;
    SmallVector<Backedge, 8> nextStates;
    BackedgeBuilder bb(builder, loc);
    auto falseVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);

    for (size_t i = 0; i < numStates; ++i) {
      auto next = bb.get(builder.getI1Type());
      nextStates.push_back(next);
      Value initVal = (static_cast<int>(i) == fragment.start) ? trueVal
                                                              : falseVal;
      auto powerOn = seq::createConstantInitialValue(
          builder, initVal.getDefiningOp());
      auto [reset, resetVal] = getResetPair(falseVal);
      auto reg = seq::CompRegOp::create(builder, loc, next, clock, reset,
                                        resetVal,
                                        builder.getStringAttr("ltl_state"),
                                        powerOn);
      stateRegs.push_back(reg);
    }

    SmallVector<SmallVector<SmallVector<Value, 4>, 4>, 8> incoming;
    incoming.resize(numStates);
    for (size_t from = 0; from < numStates; ++from) {
      for (auto &tr : nfa.states[from].transitions) {
        if (tr.isEpsilon)
          continue;
        incoming[tr.to].push_back(
            SmallVector<Value, 4>{stateRegs[from],
                                  nfa.conditions[tr.condIndex]});
      }
    }

    SmallVector<Value, 8> nextVals;
    nextVals.resize(numStates, falseVal);
    for (size_t i = 0; i < numStates; ++i) {
      SmallVector<Value, 8> orInputs;
      if (static_cast<int>(i) == fragment.start)
        orInputs.push_back(trueVal);
      for (auto &edgeVals : incoming[i]) {
        auto andVal = comb::AndOp::create(builder, loc, edgeVals, true);
        orInputs.push_back(andVal);
      }
      if (orInputs.empty())
        nextVals[i] = falseVal;
      else
        nextVals[i] = comb::OrOp::create(builder, loc, orInputs, true);
      nextStates[i].setValue(nextVals[i]);
    }

    SmallVector<Value, 8> accepting;
    for (size_t i = 0; i < numStates; ++i) {
      if (nfa.states[i].accepting)
        accepting.push_back(nextVals[i]);
    }
    if (accepting.empty())
      return falseVal;
    Value match = comb::OrOp::create(builder, loc, accepting, true);
    return match;
  }

  Value lowerSequenceMatched(Value seq, Value clock, ltl::ClockEdge edge) {
    return lowerSequence(seq, clock, edge);
  }

  Value lowerSequenceTriggered(Value seq, Value clock, ltl::ClockEdge edge) {
    if (!seq)
      return {};
    if (!isa<ltl::SequenceType>(seq.getType()))
      return seq;

    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>()) {
      auto normalizedClock =
          normalizeClock(clockOp.getClock(), clockOp.getEdge());
      return lowerSequenceTriggered(clockOp.getInput(), normalizedClock,
                                    clockOp.getEdge());
    }
    if (auto pastOp = seq.getDefiningOp<ltl::PastOp>()) {
      if (!clock) {
        pastOp.emitError("ltl.past requires a clocked sequence");
        return {};
      }
      auto triggered =
          lowerSequenceTriggered(pastOp.getInput(), clock, edge);
      if (!triggered)
        return {};
      return shiftValue(triggered, pastOp.getDelay(), clock);
    }
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    auto falseVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    auto clockPredicate = [this, clock](Value clockSignal,
                                        ltl::ClockEdge edge) -> Value {
      return getClockTick(clockSignal, edge, clock);
    };
    NFABuilder nfa(trueVal, clockPredicate);
    auto fragment = nfa.build(seq, loc, builder);
    nfa.eliminateEpsilon();
    if (fragment.start < 0 ||
        fragment.start >= static_cast<int>(nfa.states.size()))
      return falseVal;

    SmallVector<Value, 4> startConds;
    for (auto &tr : nfa.states[fragment.start].transitions) {
      if (tr.isEpsilon)
        continue;
      startConds.push_back(nfa.conditions[tr.condIndex]);
    }

    Value start;
    if (startConds.empty()) {
      start = nfa.states[fragment.start].accepting ? trueVal : falseVal;
    } else {
      start = comb::OrOp::create(builder, loc, startConds, true);
    }

    return shiftValue(start, 1, clock);
  }

  Value normalizeClock(Value clock, ltl::ClockEdge edge) {
    if (!clock)
      return {};
    if (isa<seq::ClockType>(clock.getType()))
      return clock;
    Value clockSignal = clock;
    if (edge == ltl::ClockEdge::Neg) {
      auto one =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      clockSignal = comb::XorOp::create(builder, loc, clockSignal, one);
    }
    return seq::ToClockOp::create(builder, loc, clockSignal);
  }

  Value createStateRegister(Value input, Value clock, StringRef name) {
    auto next = input;
    auto initVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    auto powerOn = seq::createConstantInitialValue(
        builder, initVal.getOperation());
    auto [reset, resetVal] = getResetPair(initVal);
    return seq::CompRegOp::create(builder, loc, next, clock, reset, resetVal,
                                  builder.getStringAttr(name), powerOn);
  }

  Value shiftValue(Value input, uint64_t delay, Value clock) {
    Value current = input;
    for (uint64_t i = 0; i < delay; ++i) {
      auto initVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
      auto powerOn = seq::createConstantInitialValue(
          builder, initVal.getOperation());
      auto [reset, resetVal] = getResetPair(initVal);
      current = seq::CompRegOp::create(builder, loc, current, clock, reset,
                                       resetVal,
                                       builder.getStringAttr("ltl_past"),
                                       powerOn);
    }
    return current;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {
  auto hwModule = getOperation();
  Value defaultClock;
  std::function<bool(Value, BlockArgument &)> traceClockRoot =
      [&](Value value, BlockArgument &root) -> bool {
    if (!value)
      return false;
    if (auto fromClock = value.getDefiningOp<seq::FromClockOp>())
      value = fromClock.getInput();
    if (auto toClock = value.getDefiningOp<seq::ToClockOp>())
      value = toClock.getInput();
    if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1 && cast->getNumResults() == 1)
        return traceClockRoot(cast->getOperand(0), root);
    }
    if (auto bitcast = value.getDefiningOp<hw::BitcastOp>())
      return traceClockRoot(bitcast.getInput(), root);
    if (auto extract = value.getDefiningOp<hw::StructExtractOp>())
      return traceClockRoot(extract.getInput(), root);
    if (auto extractOp = value.getDefiningOp<comb::ExtractOp>())
      return traceClockRoot(extractOp.getInput(), root);
    if (auto cst = value.getDefiningOp<hw::ConstantOp>())
      return true;
    if (auto cst = value.getDefiningOp<arith::ConstantOp>())
      return true;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (!root)
        root = arg;
      return arg == root;
    }
    if (auto andOp = value.getDefiningOp<comb::AndOp>()) {
      for (auto operand : andOp.getOperands())
        if (!traceClockRoot(operand, root))
          return false;
      return true;
    }
    if (auto orOp = value.getDefiningOp<comb::OrOp>()) {
      for (auto operand : orOp.getOperands())
        if (!traceClockRoot(operand, root))
          return false;
      return true;
    }
    if (auto xorOp = value.getDefiningOp<comb::XorOp>()) {
      for (auto operand : xorOp.getOperands())
        if (!traceClockRoot(operand, root))
          return false;
      return true;
    }
    if (auto concatOp = value.getDefiningOp<comb::ConcatOp>()) {
      for (auto operand : concatOp.getOperands())
        if (!traceClockRoot(operand, root))
          return false;
      return true;
    }
    return false;
  };
  auto resolveClockInputName = [&](Value clock) -> StringAttr {
    if (!clock || !hwModule)
      return {};
    BlockArgument root;
    if (!traceClockRoot(clock, root))
      return {};
    if (!root)
      return {};
    if (root.getOwner() != hwModule.getBodyBlock())
      return {};
    auto inputNames = hwModule.getInputNames();
    if (root.getArgNumber() >= inputNames.size())
      return {};
    auto nameAttr = dyn_cast<StringAttr>(inputNames[root.getArgNumber()]);
    if (!nameAttr || nameAttr.getValue().empty())
      return {};
    return nameAttr;
  };
  auto getDefaultClock = [&]() -> Value {
    if (defaultClock || !hwModule)
      return defaultClock;

    hwModule.walk([&](seq::ToClockOp op) {
      if (!defaultClock)
        defaultClock = op.getResult();
    });
    if (defaultClock)
      return defaultClock;

    auto &entryBlock = hwModule.getBody().front();
    auto inputTypes = hwModule.getInputTypes();
    for (auto it : llvm::enumerate(inputTypes)) {
      if (isa<seq::ClockType>(it.value())) {
        defaultClock = entryBlock.getArgument(it.index());
        return defaultClock;
      }
      if (auto hwStruct = dyn_cast<hw::StructType>(it.value())) {
        for (auto field : hwStruct.getElements()) {
          if (isa<seq::ClockType>(field.type)) {
            OpBuilder builder(&entryBlock, entryBlock.begin());
            auto extract = hw::StructExtractOp::create(
                builder, hwModule.getLoc(), entryBlock.getArgument(it.index()),
                field.name);
            defaultClock = extract;
            return defaultClock;
          }
        }
      }
    }
    return defaultClock;
  };

  // Set target dialects: We don't want to see any ltl or verif that might
  // come from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<HasBeenResetOpConversion>(converter, patterns.getContext());

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Lower LTL sequence match helpers to plain i1 signals.
  SmallVector<ltl::MatchedOp, 8> matchedOps;
  SmallVector<ltl::TriggeredOp, 8> triggeredOps;
  getOperation().walk([&](ltl::MatchedOp op) { matchedOps.push_back(op); });
  getOperation().walk([&](ltl::TriggeredOp op) { triggeredOps.push_back(op); });

  for (auto op : matchedOps) {
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto match =
        lowerer.lowerSequenceMatched(op.getInput(), getDefaultClock(),
                                     ltl::ClockEdge::Pos);
    if (!match) {
      op.emitError("failed to lower ltl.matched");
      return signalPassFailure();
    }
    op.replaceAllUsesWith(match);
    op.erase();
  }

  for (auto op : triggeredOps) {
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto triggered =
        lowerer.lowerSequenceTriggered(op.getInput(), getDefaultClock(),
                                       ltl::ClockEdge::Pos);
    if (!triggered) {
      op.emitError("failed to lower ltl.triggered");
      return signalPassFailure();
    }
    op.replaceAllUsesWith(triggered);
    op.erase();
  }

  // Lower any remaining LTL properties in verif assertions to i1 signals.
  SmallVector<verif::AssertOp, 8> asserts;
  SmallVector<verif::AssumeOp, 8> assumes;
  SmallVector<verif::CoverOp, 8> covers;
  getOperation().walk([&](verif::AssertOp op) { asserts.push_back(op); });
  getOperation().walk([&](verif::AssumeOp op) { assumes.push_back(op); });
  getOperation().walk([&](verif::CoverOp op) { covers.push_back(op); });

  for (auto op : asserts) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto result = lowerer.lowerProperty(op.getProperty(), getDefaultClock(),
                                        ltl::ClockEdge::Pos);
    if (!result.safety || !result.finalCheck)
      return signalPassFailure();
    op.getPropertyMutable().assign(result.safety);
    auto finalAssert = verif::AssertOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(),
        StringAttr{});
    finalAssert->setAttr("bmc.final", builder.getUnitAttr());
  }

  for (auto op : assumes) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    // skipWarmup=true: Assumes should constrain from cycle 0, not wait for
    // sequence warmup. This matches Yosys behavior with `-early -assume`.
    LTLPropertyLowerer lowerer{builder, op.getLoc(), /*skipWarmup=*/true};
    auto result = lowerer.lowerProperty(op.getProperty(), getDefaultClock(),
                                        ltl::ClockEdge::Pos);
    if (!result.safety || !result.finalCheck)
      return signalPassFailure();
    op.getPropertyMutable().assign(result.safety);
    auto finalAssume = verif::AssumeOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(),
        StringAttr{});
    finalAssume->setAttr("bmc.final", builder.getUnitAttr());
  }

  for (auto op : covers) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto result = lowerer.lowerProperty(op.getProperty(), getDefaultClock(),
                                        ltl::ClockEdge::Pos);
    if (!result.safety || !result.finalCheck)
      return signalPassFailure();
    op.getPropertyMutable().assign(result.safety);
    auto finalCover = verif::CoverOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(), StringAttr{});
    finalCover->setAttr("bmc.final", builder.getUnitAttr());
  }

  // Handle clocked assertions
  SmallVector<verif::ClockedAssertOp, 8> clockedAsserts;
  SmallVector<verif::ClockedAssumeOp, 8> clockedAssumes;
  SmallVector<verif::ClockedCoverOp, 8> clockedCovers;
  getOperation().walk(
      [&](verif::ClockedAssertOp op) { clockedAsserts.push_back(op); });
  getOperation().walk(
      [&](verif::ClockedAssumeOp op) { clockedAssumes.push_back(op); });
  getOperation().walk(
      [&](verif::ClockedCoverOp op) { clockedCovers.push_back(op); });

  for (auto op : clockedAsserts) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    auto clockName = resolveClockInputName(op.getClock());
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto ltlEdge = static_cast<ltl::ClockEdge>(
        static_cast<uint32_t>(op.getEdge()));
    auto normalizedClock = lowerer.normalizeClock(op.getClock(), ltlEdge);
    auto result = lowerer.lowerProperty(op.getProperty(), normalizedClock,
                                        ltlEdge);
    if (!result.safety || !result.finalCheck) {
      op.emitError("failed to lower clocked assertion");
      return signalPassFailure();
    }
    // Replace clocked assert with a regular assert
    auto enable = op.getEnable();
    Value property = result.safety;
    if (enable)
      property = comb::OrOp::create(builder, op.getLoc(),
                                    SmallVector<Value, 2>{
                                        comb::XorOp::create(
                                            builder, op.getLoc(), enable,
                                            hw::ConstantOp::create(
                                                builder, op.getLoc(),
                                                builder.getI1Type(), 1)),
                                        result.safety},
                                    true);
    auto assertOp = verif::AssertOp::create(builder, op.getLoc(), property,
                                            Value(), op.getLabelAttr());
    auto edgeAttr = ltl::ClockEdgeAttr::get(builder.getContext(), ltlEdge);
    if (clockName)
      assertOp->setAttr("bmc.clock", clockName);
    assertOp->setAttr("bmc.clock_edge", edgeAttr);
    auto finalAssert = verif::AssertOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(),
        StringAttr{});
    if (clockName)
      finalAssert->setAttr("bmc.clock", clockName);
    finalAssert->setAttr("bmc.clock_edge", edgeAttr);
    finalAssert->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }

  for (auto op : clockedAssumes) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    auto clockName = resolveClockInputName(op.getClock());
    // skipWarmup=true: Assumes should constrain from cycle 0, not wait for
    // sequence warmup. This matches Yosys behavior with `-early -assume`.
    LTLPropertyLowerer lowerer{builder, op.getLoc(), /*skipWarmup=*/true};
    auto ltlEdge = static_cast<ltl::ClockEdge>(
        static_cast<uint32_t>(op.getEdge()));
    auto normalizedClock = lowerer.normalizeClock(op.getClock(), ltlEdge);
    auto result = lowerer.lowerProperty(op.getProperty(), normalizedClock,
                                        ltlEdge);
    if (!result.safety || !result.finalCheck) {
      op.emitError("failed to lower clocked assumption");
      return signalPassFailure();
    }
    auto enable = op.getEnable();
    Value property = result.safety;
    if (enable)
      property = comb::OrOp::create(builder, op.getLoc(),
                                    SmallVector<Value, 2>{
                                        comb::XorOp::create(
                                            builder, op.getLoc(), enable,
                                            hw::ConstantOp::create(
                                                builder, op.getLoc(),
                                                builder.getI1Type(), 1)),
                                        result.safety},
                                    true);
    auto assumeOp = verif::AssumeOp::create(builder, op.getLoc(), property,
                                            Value(), op.getLabelAttr());
    auto edgeAttr = ltl::ClockEdgeAttr::get(builder.getContext(), ltlEdge);
    if (clockName)
      assumeOp->setAttr("bmc.clock", clockName);
    assumeOp->setAttr("bmc.clock_edge", edgeAttr);
    auto finalAssume = verif::AssumeOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(),
        StringAttr{});
    if (clockName)
      finalAssume->setAttr("bmc.clock", clockName);
    finalAssume->setAttr("bmc.clock_edge", edgeAttr);
    finalAssume->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }

  for (auto op : clockedCovers) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    auto clockName = resolveClockInputName(op.getClock());
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto ltlEdge = static_cast<ltl::ClockEdge>(
        static_cast<uint32_t>(op.getEdge()));
    auto normalizedClock = lowerer.normalizeClock(op.getClock(), ltlEdge);
    auto result = lowerer.lowerProperty(op.getProperty(), normalizedClock,
                                        ltlEdge);
    if (!result.safety || !result.finalCheck) {
      op.emitError("failed to lower clocked cover");
      return signalPassFailure();
    }
    auto enable = op.getEnable();
    Value property = result.safety;
    if (enable)
      property = comb::AndOp::create(builder, op.getLoc(),
                                     SmallVector<Value, 2>{enable,
                                                           result.safety},
                                     true);
    auto coverOp = verif::CoverOp::create(builder, op.getLoc(), property,
                                          Value(), op.getLabelAttr());
    auto edgeAttr = ltl::ClockEdgeAttr::get(builder.getContext(), ltlEdge);
    if (clockName)
      coverOp->setAttr("bmc.clock", clockName);
    coverOp->setAttr("bmc.clock_edge", edgeAttr);
    auto finalCover = verif::CoverOp::create(
        builder, op.getLoc(), result.finalCheck, op.getEnable(), StringAttr{});
    if (clockName)
      finalCover->setAttr("bmc.clock", clockName);
    finalCover->setAttr("bmc.clock_edge", edgeAttr);
    finalCover->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
