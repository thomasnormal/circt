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

struct NFABuilder {
  struct Transition {
    int condIndex;
    int to;
    bool isEpsilon;
  };
  struct State {
    bool accepting;
    SmallVector<Transition, 4> transitions;
  };
  struct Fragment {
    int start;
    SmallVector<int, 4> accepts;
  };

  NFABuilder(Value anyCondition) : anyCondition(anyCondition) {}

  int addState(bool accepting = false) {
    int id = states.size();
    states.push_back(State{accepting, {}});
    return id;
  }

  int getCondIndex(Value cond) {
    auto it = condToIndex.find(cond);
    if (it != condToIndex.end())
      return it->second;
    int index = conditions.size();
    conditions.push_back(cond);
    condToIndex[cond] = index;
    return index;
  }

  Fragment makeEmpty() {
    int start = addState(true);
    return Fragment{start, {start}};
  }

  Fragment makeSymbol(Value cond) {
    int start = addState(false);
    int accept = addState(true);
    int condIndex = getCondIndex(cond);
    states[start].transitions.push_back(Transition{condIndex, accept, false});
    return Fragment{start, {accept}};
  }

  Fragment makeAny() { return makeSymbol(anyCondition); }

  void addEpsilon(int from, int to) {
    states[from].transitions.push_back(Transition{-1, to, true});
  }

  Fragment concat(Fragment lhs, Fragment rhs) {
    for (int accept : lhs.accepts) {
      addEpsilon(accept, rhs.start);
      states[accept].accepting = false;
    }
    return Fragment{lhs.start, rhs.accepts};
  }

  Fragment makeOr(Fragment lhs, Fragment rhs) {
    int start = addState(false);
    int accept = addState(true);
    addEpsilon(start, lhs.start);
    addEpsilon(start, rhs.start);
    for (int s : lhs.accepts) {
      addEpsilon(s, accept);
      states[s].accepting = false;
    }
    for (int s : rhs.accepts) {
      addEpsilon(s, accept);
      states[s].accepting = false;
    }
    return Fragment{start, {accept}};
  }

  Fragment repeat(Fragment input, uint64_t minCount,
                  std::optional<uint64_t> maxCount) {
    if (minCount == 0 && maxCount.has_value() && *maxCount == 0)
      return makeEmpty();

    Fragment result;
    bool initialized = false;
    for (uint64_t i = 0; i < minCount; ++i) {
      Fragment copy = cloneFragment(input);
      if (!initialized) {
        result = copy;
        initialized = true;
      } else {
        result = concat(result, copy);
      }
    }

    if (!maxCount.has_value()) {
      Fragment loop = cloneFragment(input);
      if (!initialized) {
        result = makeEmpty();
        initialized = true;
      }
      for (int accept : result.accepts)
        addEpsilon(accept, loop.start);
      for (int accept : loop.accepts)
        addEpsilon(accept, loop.start);
      result.accepts.append(loop.accepts.begin(), loop.accepts.end());
      return result;
    }

    uint64_t maxVal = *maxCount;
    if (!initialized)
      result = makeEmpty();

    for (uint64_t count = minCount; count < maxVal; ++count) {
      Fragment copy = cloneFragment(input);
      for (int accept : result.accepts)
        addEpsilon(accept, copy.start);
      result.accepts.append(copy.accepts.begin(), copy.accepts.end());
    }
    return result;
  }

  Fragment delay(Fragment input, uint64_t minDelay,
                 std::optional<uint64_t> maxDelay) {
    Fragment anyFrag = makeAny();
    Fragment padding = repeat(anyFrag, minDelay, maxDelay);
    return concat(padding, input);
  }

  Fragment makeAnyStar() {
    Fragment anyFrag = makeAny();
    return repeat(anyFrag, 0, std::nullopt);
  }

  Fragment gotoRepeat(Fragment input, uint64_t minCount, uint64_t maxCount) {
    Fragment anyStar = makeAnyStar();
    Fragment result;
    bool initialized = false;
    for (uint64_t i = 0; i < minCount; ++i) {
      Fragment copy = cloneFragment(input);
      if (!initialized) {
        result = copy;
        initialized = true;
      } else {
        result = concat(result, anyStar);
        result = concat(result, copy);
      }
    }
    for (uint64_t i = minCount; i < maxCount; ++i) {
      Fragment copy = cloneFragment(input);
      Fragment option = concat(anyStar, copy);
      for (int accept : result.accepts)
        addEpsilon(accept, option.start);
      result.accepts.append(option.accepts.begin(), option.accepts.end());
    }
    return result;
  }

  Fragment nonConsecutiveRepeat(Fragment input, uint64_t minCount,
                                uint64_t maxCount) {
    Fragment result = gotoRepeat(input, minCount, maxCount);
    Fragment anyStar = makeAnyStar();
    return concat(result, anyStar);
  }

  Fragment build(Value seq, Location loc) {
    if (!seq)
      return makeEmpty();
    if (!isa<ltl::SequenceType>(seq.getType()))
      return makeSymbol(seq);

    if (auto delayOp = seq.getDefiningOp<ltl::DelayOp>()) {
      auto input = build(delayOp.getInput(), loc);
      uint64_t minDelay = delayOp.getDelay();
      std::optional<uint64_t> maxDelay;
      if (auto length = delayOp.getLength())
        maxDelay = minDelay + *length;
      return delay(input, minDelay, maxDelay);
    }
    if (auto concatOp = seq.getDefiningOp<ltl::ConcatOp>()) {
      auto inputs = concatOp.getInputs();
      if (inputs.empty())
        return makeEmpty();
      Fragment result = build(inputs.front(), loc);
      for (auto input : inputs.drop_front())
        result = concat(result, build(input, loc));
      return result;
    }
    if (auto repeatOp = seq.getDefiningOp<ltl::RepeatOp>()) {
      auto input = build(repeatOp.getInput(), loc);
      std::optional<uint64_t> maxCount;
      if (auto more = repeatOp.getMore())
        maxCount = repeatOp.getBase() + *more;
      return repeat(input, repeatOp.getBase(), maxCount);
    }
    if (auto gotoOp = seq.getDefiningOp<ltl::GoToRepeatOp>()) {
      auto input = build(gotoOp.getInput(), loc);
      return gotoRepeat(input, gotoOp.getBase(),
                        gotoOp.getBase() + gotoOp.getMore());
    }
    if (auto nonconOp = seq.getDefiningOp<ltl::NonConsecutiveRepeatOp>()) {
      auto input = build(nonconOp.getInput(), loc);
      return nonConsecutiveRepeat(input, nonconOp.getBase(),
                                  nonconOp.getBase() + nonconOp.getMore());
    }
    if (auto orOp = seq.getDefiningOp<ltl::OrOp>()) {
      auto inputs = orOp.getInputs();
      if (inputs.empty())
        return makeEmpty();
      Fragment result = build(inputs.front(), loc);
      for (auto input : inputs.drop_front())
        result = makeOr(result, build(input, loc));
      return result;
    }
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>()) {
      Fragment inner = build(firstMatch.getInput(), loc);
      return inner;
    }
    seq.getDefiningOp()->emitError("unsupported sequence lowering");
    return makeEmpty();
  }

  void eliminateEpsilon() {
    SmallVector<SmallVector<int, 4>, 8> closure(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
      SmallVector<int, 4> worklist;
      SmallVector<int, 4> visited;
      worklist.push_back(i);
      while (!worklist.empty()) {
        int s = worklist.pop_back_val();
        if (llvm::is_contained(visited, s))
          continue;
        visited.push_back(s);
        for (auto &tr : states[s].transitions) {
          if (tr.isEpsilon)
            worklist.push_back(tr.to);
        }
      }
      closure[i] = visited;
    }

    SmallVector<State, 8> newStates;
    newStates.resize(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
      bool accepting = false;
      SmallVector<Transition, 4> newTrans;
      for (int s : closure[i]) {
        if (states[s].accepting)
          accepting = true;
        for (auto &tr : states[s].transitions) {
          if (tr.isEpsilon)
            continue;
          for (int dst : closure[tr.to]) {
            newTrans.push_back(Transition{tr.condIndex, dst, false});
          }
        }
      }
      newStates[i] = State{accepting, newTrans};
    }
    states.swap(newStates);
  }

  Fragment cloneFragment(Fragment input) {
    DenseMap<int, int> map;
    SmallVector<int, 8> worklist;
    worklist.push_back(input.start);
    while (!worklist.empty()) {
      int s = worklist.pop_back_val();
      if (map.count(s))
        continue;
      int newId = addState(states[s].accepting);
      map[s] = newId;
      for (auto &tr : states[s].transitions)
        worklist.push_back(tr.to);
    }
    for (auto [oldId, newId] : map) {
      for (auto &tr : states[oldId].transitions) {
        states[newId].transitions.push_back(
            Transition{tr.condIndex, map[tr.to], tr.isEpsilon});
      }
    }
    Fragment result;
    result.start = map[input.start];
    for (int accept : input.accepts)
      result.accepts.push_back(map[accept]);
    return result;
  }

  SmallVector<State, 8> states;
  SmallVector<Value, 8> conditions;
  DenseMap<Value, int> condToIndex;
  Value anyCondition;
};

struct PropertyResult {
  Value safety;
  Value finalCheck;
};

struct LTLPropertyLowerer {
  OpBuilder &builder;
  Location loc;

  std::optional<uint64_t> getSequenceMaxLength(Value seq) {
    if (!seq)
      return 0;
    if (!isa<ltl::SequenceType>(seq.getType()))
      return 1;

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
      uint64_t maxLen = 0;
      for (auto input : intersectOp.getInputs()) {
        auto maxInput = getSequenceMaxLength(input);
        if (!maxInput)
          return std::nullopt;
        maxLen = std::max(maxLen, *maxInput);
      }
      return maxLen;
    }
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>())
      return getSequenceMaxLength(firstMatch.getInput());

    return std::nullopt;
  }

  Value shiftAges(Value input, Value zeroBit, unsigned width, Value zeroBits) {
    if (width == 1)
      return zeroBits;
    auto lowBits = comb::ExtractOp::create(builder, loc, input, 0, width - 1);
    return comb::ConcatOp::create(builder, loc,
                                  SmallVector<Value, 2>{lowBits, zeroBit});
  }

  Value lowerFirstMatchSequence(Value seq, Value clock, ltl::ClockEdge edge,
                                uint64_t maxLen) {
    static_cast<void>(edge);
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }
    if (maxLen == 0) {
      seq.getDefiningOp()->emitError(
          "first_match with empty sequences is not supported");
      return {};
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    NFABuilder nfa(trueVal);
    auto fragment = nfa.build(seq, loc);
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
      auto reg = seq::CompRegOp::create(builder, loc, next, clock, Value(),
                                        Value(),
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
      for (auto &tr : nfa.states[from].transitions) {
        if (tr.isEpsilon)
          continue;
        auto shifted =
            shiftAges(currentStates[from], zeroBit, ageWidth, zeroBits);
        auto mask =
            comb::ReplicateOp::create(builder, loc,
                                      nfa.conditions[tr.condIndex], ageWidth);
        auto masked = comb::AndOp::create(builder, loc, shifted, mask);
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
      return {lowerSequence(prop, clock, edge), trueVal};
    }

    if (auto clockOp = prop.getDefiningOp<ltl::ClockOp>()) {
      return lowerProperty(clockOp.getInput(),
                           normalizeClock(clockOp.getClock(), clockOp.getEdge()),
                           clockOp.getEdge());
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
      auto neg = comb::XorOp::create(builder, loc, inner.safety,
                                     hw::ConstantOp::create(
                                         builder, loc, builder.getI1Type(), 1));
      auto finalNeg = comb::XorOp::create(
          builder, loc, inner.finalCheck,
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      return {neg, finalNeg};
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
      SmallVector<Value, 4> safeties;
      SmallVector<PropertyResult, 4> results;
      Value finalCheck = nullptr;
      Value disableInput;
      PropertyResult disableRes;
      bool haveDisableRes = false;
      uint64_t disableShift = 0;
      auto inputs = orOp.getInputs();
      if (inputs.size() == 2) {
        auto implOp = inputs[0].getDefiningOp<ltl::ImplicationOp>();
        if (!implOp)
          implOp = inputs[1].getDefiningOp<ltl::ImplicationOp>();
        if (implOp) {
          Value otherInput = (implOp == inputs[0].getDefiningOp<ltl::ImplicationOp>())
                                 ? inputs[1]
                                 : inputs[0];
          if (auto delayOp =
                  implOp.getConsequent().getDefiningOp<ltl::DelayOp>()) {
            if (auto length = delayOp.getLength()) {
              if (*length == 0 && delayOp.getDelay() > 0) {
                disableShift = delayOp.getDelay();
                disableInput = otherInput;
              }
            }
          }
        }
      }
      for (auto input : inputs) {
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
        if (input == disableInput) {
          disableRes = res;
          haveDisableRes = true;
        }
      }
      if (disableShift > 0) {
        if (!clock || !haveDisableRes) {
          orOp.emitError("disable iff requires a clocked property");
          return {Value(), {}};
        }
        Value shiftedDisable =
            shiftValue(disableRes.safety, disableShift, clock);
        safeties.push_back(shiftedDisable);
        if (!finalCheck)
          finalCheck = shiftedDisable;
        else
          finalCheck = comb::OrOp::create(builder, loc, finalCheck,
                                          shiftedDisable);
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
            shiftDelay = delayOp.getDelay();
            consequentValue = delayOp.getInput();
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
      auto safety =
          comb::OrOp::create(builder, loc,
                             SmallVector<Value, 2>{notAntecedent,
                                                   consequent.safety},
                             true);
      if (!clock) {
        implOp.emitError("implication requires a clocked property");
        return {Value(), {}};
      }
      auto antecedentSeen = createStateRegister(
          antecedent, clock, "ltl_implication_seen");
      auto notSeen = comb::XorOp::create(
          builder, loc, antecedentSeen,
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1));
      auto finalCheck =
          comb::OrOp::create(builder, loc,
                             SmallVector<Value, 2>{notSeen,
                                                   consequent.finalCheck},
                             true);
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
      auto seen = createStateRegister(input.safety, clock,
                                      "ltl_eventually_seen");
      auto trueVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
      PropertyResult result{trueVal, seen};
      return result;
    }

    prop.getDefiningOp()->emitError("unsupported property lowering");
    return {Value(), {}};
  }

  Value lowerSequence(Value seq, Value clock, ltl::ClockEdge edge) {
    if (!seq)
      return {};
    if (!isa<ltl::SequenceType>(seq.getType()))
      return seq;

    if (auto clockOp = seq.getDefiningOp<ltl::ClockOp>()) {
      return lowerSequence(clockOp.getInput(),
                           normalizeClock(clockOp.getClock(), clockOp.getEdge()),
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
    if (auto intersectOp = seq.getDefiningOp<ltl::IntersectOp>()) {
      SmallVector<Value, 4> inputs;
      for (auto input : intersectOp.getInputs())
        inputs.push_back(lowerSequence(input, clock, edge));
      return comb::AndOp::create(builder, loc, inputs, true);
    }
    if (!clock) {
      seq.getDefiningOp()->emitError("sequence lowering requires a clock");
      return {};
    }

    auto trueVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 1);
    NFABuilder nfa(trueVal);
    if (auto firstMatch = seq.getDefiningOp<ltl::FirstMatchOp>()) {
      auto maxLen = getSequenceMaxLength(firstMatch.getInput());
      if (!maxLen) {
        firstMatch.emitError(
            "first_match lowering requires a bounded sequence");
        return {};
      }
      return lowerFirstMatchSequence(firstMatch.getInput(), clock, edge,
                                     *maxLen);
    }
    auto fragment = nfa.build(seq, loc);
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
      auto reg = seq::CompRegOp::create(builder, loc, next, clock,
                                        Value(), Value(),
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
    if (edge == ltl::ClockEdge::Both) {
      mlir::emitError(loc,
                      "both-edge clocks are not supported in LTL lowering");
      return {};
    }
    return seq::ToClockOp::create(builder, loc, clockSignal);
  }

  Value createStateRegister(Value input, Value clock, StringRef name) {
    auto next = input;
    auto initVal =
        hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
    auto powerOn = seq::createConstantInitialValue(
        builder, initVal.getOperation());
    return seq::CompRegOp::create(builder, loc, next, clock, Value(), Value(),
                                  builder.getStringAttr(name), powerOn);
  }

  Value shiftValue(Value input, uint64_t delay, Value clock) {
    Value current = input;
    for (uint64_t i = 0; i < delay; ++i) {
      auto initVal =
          hw::ConstantOp::create(builder, loc, builder.getI1Type(), 0);
      auto powerOn = seq::createConstantInitialValue(
          builder, initVal.getOperation());
      current = seq::CompRegOp::create(builder, loc, current, clock, Value(),
                                       Value(),
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
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
    finalAssert->setAttr("bmc.final", builder.getUnitAttr());
  }

  for (auto op : assumes) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
    auto result = lowerer.lowerProperty(op.getProperty(), getDefaultClock(),
                                        ltl::ClockEdge::Pos);
    if (!result.safety || !result.finalCheck)
      return signalPassFailure();
    op.getPropertyMutable().assign(result.safety);
    auto finalAssume = verif::AssumeOp::create(
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
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
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
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
    (void)verif::AssertOp::create(builder, op.getLoc(), property,
                                   Value(), op.getLabelAttr());
    auto finalAssert = verif::AssertOp::create(
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
    finalAssert->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }

  for (auto op : clockedAssumes) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
    LTLPropertyLowerer lowerer{builder, op.getLoc()};
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
    (void)verif::AssumeOp::create(builder, op.getLoc(), property,
                                   Value(), op.getLabelAttr());
    auto finalAssume = verif::AssumeOp::create(
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
    finalAssume->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }

  for (auto op : clockedCovers) {
    if (!isa<ltl::PropertyType, ltl::SequenceType>(op.getProperty().getType()))
      continue;
    OpBuilder builder(op);
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
    (void)verif::CoverOp::create(builder, op.getLoc(), property,
                                  Value(), op.getLabelAttr());
    auto finalCover = verif::CoverOp::create(
        builder, op.getLoc(), result.finalCheck, Value(), StringAttr{});
    finalCover->setAttr("bmc.final", builder.getUnitAttr());
    op.erase();
  }
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
