//===- LTLSequenceNFA.h - LTL sequence NFA utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_LTLSEQUENCENFA_H
#define CIRCT_SUPPORT_LTLSEQUENCENFA_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include <functional>
#include <utility>

namespace circt {
namespace ltl {

struct NFABuilder {
  using ClockEdgePredicate = std::function<Value(Value, ltl::ClockEdge)>;
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

  NFABuilder(Value anyCondition,
             ClockEdgePredicate clockEdgePredicate = {})
      : clockEdgePredicate(std::move(clockEdgePredicate)),
        anyCondition(anyCondition) {}

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

  Fragment gotoRepeat(Fragment input, uint64_t minCount,
                      std::optional<uint64_t> maxCount) {
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
    if (!maxCount.has_value()) {
      if (!initialized) {
        result = makeEmpty();
        initialized = true;
      }
      Fragment copy = cloneFragment(input);
      Fragment option = concat(anyStar, copy);
      for (int accept : result.accepts)
        addEpsilon(accept, option.start);
      for (int accept : option.accepts)
        addEpsilon(accept, option.start);
      result.accepts.append(option.accepts.begin(), option.accepts.end());
      return result;
    }
    for (uint64_t i = minCount; i < *maxCount; ++i) {
      Fragment copy = cloneFragment(input);
      Fragment option = concat(anyStar, copy);
      for (int accept : result.accepts)
        addEpsilon(accept, option.start);
      result.accepts.append(option.accepts.begin(), option.accepts.end());
    }
    return result;
  }

  Fragment nonConsecutiveRepeat(Fragment input, uint64_t minCount,
                                std::optional<uint64_t> maxCount) {
    Fragment result = gotoRepeat(input, minCount, maxCount);
    Fragment anyStar = makeAnyStar();
    return concat(result, anyStar);
  }

  static std::pair<NFABuilder, Fragment>
  buildEpsilonFreeNFA(Value seq, Location loc, Value anyCondition,
                      OpBuilder &builder,
                      ClockEdgePredicate clockEdgePredicate = {}) {
    NFABuilder nfa(anyCondition, std::move(clockEdgePredicate));
    auto fragment = nfa.build(seq, loc, builder);
    nfa.eliminateEpsilon();
    return {std::move(nfa), fragment};
  }

  static std::pair<NFABuilder, Fragment>
  intersectNFAs(const NFABuilder &lhs, const Fragment &lhsFrag,
                const NFABuilder &rhs, const Fragment &rhsFrag,
                OpBuilder &builder, Location loc, Value anyCondition) {
    NFABuilder result(anyCondition);
    DenseMap<uint64_t, int> stateMap;
    SmallVector<std::pair<int, int>, 8> worklist;

    auto packKey = [](int a, int b) -> uint64_t {
      return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
             static_cast<uint32_t>(b);
    };

    auto getState = [&](int a, int b) -> int {
      uint64_t key = packKey(a, b);
      auto it = stateMap.find(key);
      if (it != stateMap.end())
        return it->second;
      bool accepting = lhs.states[a].accepting && rhs.states[b].accepting;
      int id = result.addState(accepting);
      stateMap[key] = id;
      worklist.push_back({a, b});
      return id;
    };

    int start = getState(lhsFrag.start, rhsFrag.start);

    while (!worklist.empty()) {
      auto [a, b] = worklist.pop_back_val();
      int from = stateMap[packKey(a, b)];
      auto &lhsState = lhs.states[a];
      auto &rhsState = rhs.states[b];
      for (auto &lhsTr : lhsState.transitions) {
        if (lhsTr.isEpsilon)
          continue;
        Value lhsCond = lhs.conditions[lhsTr.condIndex];
        for (auto &rhsTr : rhsState.transitions) {
          if (rhsTr.isEpsilon)
            continue;
          Value rhsCond = rhs.conditions[rhsTr.condIndex];
          Value cond;
          if (lhsCond == rhsCond) {
            cond = lhsCond;
          } else {
            cond = comb::AndOp::create(builder, loc,
                                       SmallVector<Value, 2>{lhsCond, rhsCond},
                                       true).getResult();
          }
          int condIndex = result.getCondIndex(cond);
          int to = getState(lhsTr.to, rhsTr.to);
          result.states[from].transitions.push_back(
              Transition{condIndex, to, false});
        }
      }
    }

    SmallVector<int, 4> accepts;
    for (size_t i = 0; i < result.states.size(); ++i) {
      if (result.states[i].accepting)
        accepts.push_back(static_cast<int>(i));
    }
    return {std::move(result), Fragment{start, accepts}};
  }

  Fragment build(Value seq, Location loc, OpBuilder &builder) {
    if (!seq)
      return makeEmpty();
    if (!isa<ltl::SequenceType>(seq.getType()))
      return makeSymbol(seq);

    Operation *defOp = seq.getDefiningOp();
    if (!defOp) {
      emitError(loc, "unsupported sequence lowering for block argument");
      return makeEmpty();
    }

    if (auto clockOp = dyn_cast<ltl::ClockOp>(defOp)) {
      auto inner = build(clockOp.getInput(), loc, builder);
      if (clockEdgePredicate) {
        auto tick = clockEdgePredicate(clockOp.getClock(), clockOp.getEdge());
        if (tick)
          gateFragmentOnTick(inner, tick, loc, builder);
      }
      return inner;
    }
    if (auto delayOp = dyn_cast<ltl::DelayOp>(defOp)) {
      auto input = build(delayOp.getInput(), loc, builder);
      uint64_t minDelay = delayOp.getDelay();
      std::optional<uint64_t> maxDelay;
      if (auto length = delayOp.getLength())
        maxDelay = minDelay + *length;
      return delay(input, minDelay, maxDelay);
    }
    if (auto concatOp = dyn_cast<ltl::ConcatOp>(defOp)) {
      auto inputs = concatOp.getInputs();
      if (inputs.empty())
        return makeEmpty();
      Fragment result = build(inputs.front(), loc, builder);
      for (auto input : inputs.drop_front())
        result = concat(result, build(input, loc, builder));
      return result;
    }
    if (auto repeatOp = dyn_cast<ltl::RepeatOp>(defOp)) {
      auto input = build(repeatOp.getInput(), loc, builder);
      std::optional<uint64_t> maxCount;
      if (auto more = repeatOp.getMore())
        maxCount = repeatOp.getBase() + *more;
      return repeat(input, repeatOp.getBase(), maxCount);
    }
    if (auto gotoOp = dyn_cast<ltl::GoToRepeatOp>(defOp)) {
      auto input = build(gotoOp.getInput(), loc, builder);
      std::optional<uint64_t> maxCount;
      if (auto more = gotoOp.getMore())
        maxCount = gotoOp.getBase() + *more;
      return gotoRepeat(input, gotoOp.getBase(), maxCount);
    }
    if (auto nonconOp = dyn_cast<ltl::NonConsecutiveRepeatOp>(defOp)) {
      auto input = build(nonconOp.getInput(), loc, builder);
      std::optional<uint64_t> maxCount;
      if (auto more = nonconOp.getMore())
        maxCount = nonconOp.getBase() + *more;
      return nonConsecutiveRepeat(input, nonconOp.getBase(), maxCount);
    }
    if (auto orOp = dyn_cast<ltl::OrOp>(defOp)) {
      auto inputs = orOp.getInputs();
      if (inputs.empty())
        return makeEmpty();
      Fragment result = build(inputs.front(), loc, builder);
      for (auto input : inputs.drop_front())
        result = makeOr(result, build(input, loc, builder));
      return result;
    }
    if (auto andOp = dyn_cast<ltl::AndOp>(defOp)) {
      auto inputs = andOp.getInputs();
      if (inputs.empty())
        return makeEmpty();

      auto [lhsBuilder, lhsFrag] =
          buildEpsilonFreeNFA(inputs.front(), loc, anyCondition, builder,
                              clockEdgePredicate);
      for (auto input : inputs.drop_front()) {
        auto [rhsBuilder, rhsFrag] =
            buildEpsilonFreeNFA(input, loc, anyCondition, builder,
                                clockEdgePredicate);
        auto intersected = intersectNFAs(lhsBuilder, lhsFrag, rhsBuilder,
                                         rhsFrag, builder, loc, anyCondition);
        lhsBuilder = std::move(intersected.first);
        lhsFrag = intersected.second;
      }

      *this = std::move(lhsBuilder);
      return lhsFrag;
    }
    if (auto intersectOp = dyn_cast<ltl::IntersectOp>(defOp)) {
      auto inputs = intersectOp.getInputs();
      if (inputs.empty())
        return makeEmpty();

      auto [lhsBuilder, lhsFrag] =
          buildEpsilonFreeNFA(inputs.front(), loc, anyCondition, builder,
                              clockEdgePredicate);
      for (auto input : inputs.drop_front()) {
        auto [rhsBuilder, rhsFrag] =
            buildEpsilonFreeNFA(input, loc, anyCondition, builder,
                                clockEdgePredicate);
        auto intersected = intersectNFAs(lhsBuilder, lhsFrag, rhsBuilder,
                                         rhsFrag, builder, loc, anyCondition);
        lhsBuilder = std::move(intersected.first);
        lhsFrag = intersected.second;
      }

      *this = std::move(lhsBuilder);
      return lhsFrag;
    }
    if (auto firstMatch = dyn_cast<ltl::FirstMatchOp>(defOp)) {
      Fragment inner = build(firstMatch.getInput(), loc, builder);
      return inner;
    }
    defOp->emitError("unsupported sequence lowering");
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

  void gateFragmentOnTick(const Fragment &fragment, Value tick, Location loc,
                          OpBuilder &builder) {
    if (!tick)
      return;
    if (auto cst = tick.getDefiningOp<hw::ConstantOp>())
      if (cst.getValue().isOne())
        return;

    auto i1Type = builder.getI1Type();
    Value one = hw::ConstantOp::create(builder, loc, i1Type, 1).getResult();
    Value notTick = comb::XorOp::create(builder, loc, tick, one).getResult();
    int notTickIndex = getCondIndex(notTick);

    SmallVector<int, 16> worklist;
    SmallVector<char, 16> visited(states.size(), false);
    worklist.push_back(fragment.start);
    while (!worklist.empty()) {
      int state = worklist.pop_back_val();
      if (visited[state])
        continue;
      visited[state] = true;

      for (auto &tr : states[state].transitions)
        worklist.push_back(tr.to);

      bool hasNotTickLoop = false;
      for (auto &tr : states[state].transitions) {
        if (!tr.isEpsilon && tr.to == state &&
            tr.condIndex == notTickIndex) {
          hasNotTickLoop = true;
          break;
        }
      }
      if (!hasNotTickLoop)
        states[state].transitions.push_back(
            Transition{notTickIndex, state, false});

      for (auto &tr : states[state].transitions) {
        if (tr.isEpsilon)
          continue;
        if (tr.to == state && tr.condIndex == notTickIndex)
          continue;
        Value cond = conditions[tr.condIndex];
        Value gated = comb::AndOp::create(
            builder, loc, SmallVector<Value, 2>{cond, tick}, true).getResult();
        tr.condIndex = getCondIndex(gated);
      }
    }
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
        if (!map.count(tr.to))
          continue;
        states[newId].transitions.push_back(
            Transition{tr.condIndex, map[tr.to], tr.isEpsilon});
      }
    }

    SmallVector<int, 4> accepts;
    for (int s : input.accepts)
      accepts.push_back(map[s]);
    return Fragment{map[input.start], accepts};
  }

  ClockEdgePredicate clockEdgePredicate;
  Value anyCondition;
  SmallVector<State, 8> states;
  SmallVector<Value, 8> conditions;
  DenseMap<Value, int> condToIndex;
};

} // namespace ltl
} // namespace circt

#endif // CIRCT_SUPPORT_LTLSEQUENCENFA_H
