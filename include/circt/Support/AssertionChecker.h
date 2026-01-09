//===- AssertionChecker.h - SVA Assertion Runtime Checker -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the runtime assertion checker infrastructure for
// SystemVerilog Assertions (SVA). The checker provides:
//
// - Automata-based sequence matching for temporal sequences
// - Property evaluation with proper handling of implications
// - Failure reporting with trace information
// - Vacuity detection for identifying vacuously passing assertions
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_ASSERTIONCHECKER_H
#define CIRCT_SUPPORT_ASSERTIONCHECKER_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace circt {
namespace sva {

//===----------------------------------------------------------------------===//
// Assertion Status
//===----------------------------------------------------------------------===//

/// Status of an assertion evaluation.
enum class AssertionStatus {
  /// Assertion is still being evaluated (pending match).
  Pending,
  /// Assertion has passed (property holds).
  Passed,
  /// Assertion has failed (property violated).
  Failed,
  /// Assertion passed vacuously (antecedent never matched).
  VacuousPass
};

//===----------------------------------------------------------------------===//
// Assertion Failure Information
//===----------------------------------------------------------------------===//

/// Information about an assertion failure.
struct AssertionFailure {
  /// Name/label of the assertion.
  std::string name;

  /// Custom failure message (if any).
  std::string message;

  /// Cycle number when the failure was detected.
  uint64_t failureCycle;

  /// Cycle number when the assertion started evaluating.
  uint64_t startCycle;

  /// Source file location (if available).
  std::string sourceFile;

  /// Source line number (if available).
  uint32_t sourceLine;

  /// Trace of signal values during evaluation.
  std::vector<std::pair<std::string, bool>> signalTrace;
};

//===----------------------------------------------------------------------===//
// Sequence Automaton
//===----------------------------------------------------------------------===//

/// A state in the sequence automaton.
struct AutomatonState {
  /// Unique state ID.
  uint32_t id;

  /// Whether this is an accepting state.
  bool isAccepting;

  /// Transitions: maps condition index to next state ID.
  /// Condition index -1 means epsilon transition (unconditional).
  llvm::SmallVector<std::pair<int32_t, uint32_t>> transitions;

  /// Delay count (number of cycles to wait before taking transitions).
  uint32_t delay;

  /// Whether this state has been visited in current evaluation.
  bool visited;
};

/// Automaton-based sequence matcher.
class SequenceAutomaton {
public:
  SequenceAutomaton();

  /// Add a new state to the automaton.
  uint32_t addState(bool isAccepting = false, uint32_t delay = 0);

  /// Add a transition between states.
  void addTransition(uint32_t fromState, uint32_t toState,
                     int32_t conditionIndex = -1);

  /// Set the initial state.
  void setInitialState(uint32_t stateId);

  /// Reset the automaton to initial state.
  void reset();

  /// Advance the automaton by one cycle with given condition values.
  /// Returns true if any accepting state is reached.
  bool step(const std::vector<bool> &conditions);

  /// Check if the automaton is in an accepting state.
  bool isAccepting() const;

  /// Check if the automaton has any active states.
  bool hasActiveStates() const;

  /// Get the number of cycles since the last reset.
  uint64_t getCycleCount() const { return cycleCount; }

private:
  std::vector<AutomatonState> states;
  llvm::SmallVector<uint32_t> activeStates;
  llvm::SmallVector<uint32_t> nextActiveStates;
  uint32_t initialState;
  uint64_t cycleCount;
};

//===----------------------------------------------------------------------===//
// Property Checker
//===----------------------------------------------------------------------===//

/// Types of property checks.
enum class PropertyType {
  Assert,
  Assume,
  Cover,
  Expect
};

/// A single property being checked.
struct PropertyCheck {
  /// Unique ID of this property.
  uint32_t id;

  /// Name/label of the property.
  std::string name;

  /// Type of check (assert, assume, cover, expect).
  PropertyType type;

  /// Sequence automaton for the antecedent (if implication).
  std::unique_ptr<SequenceAutomaton> antecedent;

  /// Sequence automaton for the consequent.
  std::unique_ptr<SequenceAutomaton> consequent;

  /// Whether this is an overlapping implication.
  bool overlapping;

  /// Custom failure message.
  std::string message;

  /// Source file location.
  std::string sourceFile;

  /// Source line number.
  uint32_t sourceLine;

  /// Whether the antecedent has ever matched (for vacuity detection).
  bool antecedentMatched;

  /// Current status.
  AssertionStatus status;

  /// Cycle when this property started evaluating.
  uint64_t startCycle;
};

/// Callback type for reporting assertion failures.
using FailureCallback = std::function<void(const AssertionFailure &)>;

/// Callback type for reporting assertion passes (including vacuous).
using PassCallback = std::function<void(const std::string &name,
                                         AssertionStatus status)>;

/// Runtime assertion checker.
class AssertionChecker {
public:
  AssertionChecker();
  ~AssertionChecker();

  /// Register a failure callback.
  void setFailureCallback(FailureCallback callback);

  /// Register a pass callback.
  void setPassCallback(PassCallback callback);

  /// Add a new property to check.
  uint32_t addProperty(PropertyType type, StringRef name, StringRef message = "",
                       StringRef sourceFile = "", uint32_t sourceLine = 0);

  /// Set the antecedent automaton for a property.
  void setAntecedent(uint32_t propertyId, std::unique_ptr<SequenceAutomaton> automaton);

  /// Set the consequent automaton for a property.
  void setConsequent(uint32_t propertyId, std::unique_ptr<SequenceAutomaton> automaton);

  /// Set whether the implication is overlapping.
  void setOverlapping(uint32_t propertyId, bool overlapping);

  /// Advance all properties by one cycle.
  /// conditions: map from condition index to boolean value.
  void step(const std::vector<bool> &conditions);

  /// Reset all properties to initial state.
  void reset();

  /// Get the current cycle count.
  uint64_t getCycleCount() const { return cycleCount; }

  /// Get the number of failures.
  uint64_t getFailureCount() const { return failureCount; }

  /// Get the number of passes (including vacuous).
  uint64_t getPassCount() const { return passCount; }

  /// Get the number of vacuous passes.
  uint64_t getVacuousCount() const { return vacuousCount; }

  /// Get the number of cover hits.
  uint64_t getCoverCount() const { return coverCount; }

  /// Check if all assertions have passed (no failures).
  bool allPassed() const { return failureCount == 0; }

  /// Enable/disable vacuity warnings.
  void setVacuityWarning(bool enable) { vacuityWarning = enable; }

  /// Get property by ID.
  const PropertyCheck *getProperty(uint32_t id) const;

private:
  void evaluateProperty(PropertyCheck &prop, const std::vector<bool> &conditions);
  void reportFailure(const PropertyCheck &prop);
  void reportPass(const PropertyCheck &prop);

  std::vector<std::unique_ptr<PropertyCheck>> properties;
  FailureCallback failureCallback;
  PassCallback passCallback;
  uint64_t cycleCount;
  uint64_t failureCount;
  uint64_t passCount;
  uint64_t vacuousCount;
  uint64_t coverCount;
  bool vacuityWarning;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Create a simple sequence automaton for a single boolean condition.
std::unique_ptr<SequenceAutomaton> createBooleanSequence(uint32_t conditionIndex);

/// Create a delay sequence automaton (##n).
std::unique_ptr<SequenceAutomaton> createDelaySequence(
    std::unique_ptr<SequenceAutomaton> input, uint32_t delay);

/// Create a range delay sequence automaton (##[m:n]).
std::unique_ptr<SequenceAutomaton> createRangeDelaySequence(
    std::unique_ptr<SequenceAutomaton> input, uint32_t minDelay,
    uint32_t maxDelay);

/// Create a consecutive repetition sequence automaton ([*n]).
std::unique_ptr<SequenceAutomaton> createRepeatSequence(
    std::unique_ptr<SequenceAutomaton> input, uint32_t count);

/// Create a range repetition sequence automaton ([*m:n]).
std::unique_ptr<SequenceAutomaton> createRangeRepeatSequence(
    std::unique_ptr<SequenceAutomaton> input, uint32_t minCount,
    uint32_t maxCount);

/// Create a concatenation of two sequence automatons.
std::unique_ptr<SequenceAutomaton> createConcatSequence(
    std::unique_ptr<SequenceAutomaton> first,
    std::unique_ptr<SequenceAutomaton> second);

/// Create a disjunction of two sequence automatons.
std::unique_ptr<SequenceAutomaton> createOrSequence(
    std::unique_ptr<SequenceAutomaton> first,
    std::unique_ptr<SequenceAutomaton> second);

} // namespace sva
} // namespace circt

#endif // CIRCT_SUPPORT_ASSERTIONCHECKER_H
