//===- AssertionChecker.cpp - SVA Assertion Runtime Checker --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/AssertionChecker.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace circt::sva;

//===----------------------------------------------------------------------===//
// SequenceAutomaton
//===----------------------------------------------------------------------===//

SequenceAutomaton::SequenceAutomaton() : initialState(0), cycleCount(0) {}

uint32_t SequenceAutomaton::addState(bool isAccepting, uint32_t delay) {
  uint32_t id = states.size();
  states.push_back({id, isAccepting, {}, delay, false});
  return id;
}

void SequenceAutomaton::addTransition(uint32_t fromState, uint32_t toState,
                                       int32_t conditionIndex) {
  assert(fromState < states.size() && "Invalid from state");
  assert(toState < states.size() && "Invalid to state");
  states[fromState].transitions.push_back({conditionIndex, toState});
}

void SequenceAutomaton::setInitialState(uint32_t stateId) {
  assert(stateId < states.size() && "Invalid initial state");
  initialState = stateId;
}

void SequenceAutomaton::reset() {
  activeStates.clear();
  activeStates.push_back(initialState);
  cycleCount = 0;
  for (auto &state : states)
    state.visited = false;
}

bool SequenceAutomaton::step(const std::vector<bool> &conditions) {
  nextActiveStates.clear();
  bool accepted = false;

  // Clear visited flags
  for (auto &state : states)
    state.visited = false;

  // Process each active state
  for (uint32_t stateId : activeStates) {
    auto &state = states[stateId];

    // Handle delay
    if (state.delay > 0 && cycleCount < state.delay) {
      nextActiveStates.push_back(stateId);
      continue;
    }

    // Check if this is an accepting state
    if (state.isAccepting)
      accepted = true;

    // Process transitions
    for (const auto &[conditionIndex, nextStateId] : state.transitions) {
      // Epsilon transition (unconditional)
      if (conditionIndex < 0) {
        if (!states[nextStateId].visited) {
          states[nextStateId].visited = true;
          nextActiveStates.push_back(nextStateId);
        }
        continue;
      }

      // Conditional transition
      if (static_cast<size_t>(conditionIndex) < conditions.size() &&
          conditions[conditionIndex]) {
        if (!states[nextStateId].visited) {
          states[nextStateId].visited = true;
          nextActiveStates.push_back(nextStateId);
        }
      }
    }
  }

  std::swap(activeStates, nextActiveStates);
  ++cycleCount;
  return accepted;
}

bool SequenceAutomaton::isAccepting() const {
  for (uint32_t stateId : activeStates) {
    if (states[stateId].isAccepting)
      return true;
  }
  return false;
}

bool SequenceAutomaton::hasActiveStates() const {
  return !activeStates.empty();
}

//===----------------------------------------------------------------------===//
// AssertionChecker
//===----------------------------------------------------------------------===//

AssertionChecker::AssertionChecker()
    : cycleCount(0), failureCount(0), passCount(0), vacuousCount(0),
      coverCount(0), vacuityWarning(true) {}

AssertionChecker::~AssertionChecker() = default;

void AssertionChecker::setFailureCallback(FailureCallback callback) {
  failureCallback = std::move(callback);
}

void AssertionChecker::setPassCallback(PassCallback callback) {
  passCallback = std::move(callback);
}

uint32_t AssertionChecker::addProperty(PropertyType type, StringRef name,
                                        StringRef message, StringRef sourceFile,
                                        uint32_t sourceLine) {
  auto prop = std::make_unique<PropertyCheck>();
  prop->id = properties.size();
  prop->name = name.str();
  prop->type = type;
  prop->overlapping = true;
  prop->message = message.str();
  prop->sourceFile = sourceFile.str();
  prop->sourceLine = sourceLine;
  prop->antecedentMatched = false;
  prop->status = AssertionStatus::Pending;
  prop->startCycle = 0;

  uint32_t id = prop->id;
  properties.push_back(std::move(prop));
  return id;
}

void AssertionChecker::setAntecedent(uint32_t propertyId,
                                      std::unique_ptr<SequenceAutomaton> automaton) {
  assert(propertyId < properties.size() && "Invalid property ID");
  properties[propertyId]->antecedent = std::move(automaton);
}

void AssertionChecker::setConsequent(uint32_t propertyId,
                                      std::unique_ptr<SequenceAutomaton> automaton) {
  assert(propertyId < properties.size() && "Invalid property ID");
  properties[propertyId]->consequent = std::move(automaton);
}

void AssertionChecker::setOverlapping(uint32_t propertyId, bool overlapping) {
  assert(propertyId < properties.size() && "Invalid property ID");
  properties[propertyId]->overlapping = overlapping;
}

void AssertionChecker::step(const std::vector<bool> &conditions) {
  for (auto &prop : properties) {
    evaluateProperty(*prop, conditions);
  }
  ++cycleCount;
}

void AssertionChecker::reset() {
  cycleCount = 0;
  for (auto &prop : properties) {
    prop->antecedentMatched = false;
    prop->status = AssertionStatus::Pending;
    prop->startCycle = 0;
    if (prop->antecedent)
      prop->antecedent->reset();
    if (prop->consequent)
      prop->consequent->reset();
  }
}

const PropertyCheck *AssertionChecker::getProperty(uint32_t id) const {
  if (id < properties.size())
    return properties[id].get();
  return nullptr;
}

void AssertionChecker::evaluateProperty(PropertyCheck &prop,
                                         const std::vector<bool> &conditions) {
  // Handle different property types
  switch (prop.type) {
  case PropertyType::Cover:
    // Cover: check if the sequence matches
    if (prop.consequent) {
      if (prop.consequent->step(conditions)) {
        ++coverCount;
        prop.status = AssertionStatus::Passed;
        reportPass(prop);
      }
    }
    break;

  case PropertyType::Assert:
  case PropertyType::Expect: {
    // If no antecedent, evaluate consequent directly
    if (!prop.antecedent) {
      if (prop.consequent) {
        bool matched = prop.consequent->step(conditions);
        if (!matched && !prop.consequent->hasActiveStates()) {
          // Consequent failed to match
          ++failureCount;
          prop.status = AssertionStatus::Failed;
          reportFailure(prop);
        } else if (matched) {
          ++passCount;
          prop.status = AssertionStatus::Passed;
          reportPass(prop);
        }
      }
      break;
    }

    // Evaluate antecedent
    bool antecedentMatched = prop.antecedent->step(conditions);

    if (antecedentMatched) {
      prop.antecedentMatched = true;
      prop.startCycle = cycleCount;

      // Start evaluating consequent
      if (prop.consequent) {
        prop.consequent->reset();
        if (!prop.overlapping) {
          // Non-overlapping: consequent starts next cycle
          // The automaton will handle the delay
        }
      }
    }

    // Evaluate consequent if antecedent has matched
    if (prop.antecedentMatched && prop.consequent) {
      bool consequentMatched = prop.consequent->step(conditions);
      if (consequentMatched) {
        ++passCount;
        prop.status = AssertionStatus::Passed;
        reportPass(prop);
        prop.antecedentMatched = false; // Reset for next evaluation
      } else if (!prop.consequent->hasActiveStates()) {
        // Consequent failed to match
        ++failureCount;
        prop.status = AssertionStatus::Failed;
        reportFailure(prop);
        prop.antecedentMatched = false;
      }
    }
    break;
  }

  case PropertyType::Assume:
    // Assume: treated like assert but typically used for input constraints
    if (!prop.antecedent) {
      if (prop.consequent) {
        bool matched = prop.consequent->step(conditions);
        if (!matched && !prop.consequent->hasActiveStates()) {
          ++failureCount;
          prop.status = AssertionStatus::Failed;
          reportFailure(prop);
        } else if (matched) {
          ++passCount;
          prop.status = AssertionStatus::Passed;
          reportPass(prop);
        }
      }
    }
    break;
  }
}

void AssertionChecker::reportFailure(const PropertyCheck &prop) {
  if (failureCallback) {
    AssertionFailure failure;
    failure.name = prop.name;
    failure.message = prop.message;
    failure.failureCycle = cycleCount;
    failure.startCycle = prop.startCycle;
    failure.sourceFile = prop.sourceFile;
    failure.sourceLine = prop.sourceLine;
    failureCallback(failure);
  }
}

void AssertionChecker::reportPass(const PropertyCheck &prop) {
  if (passCallback) {
    passCallback(prop.name, prop.status);
  }
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<SequenceAutomaton>
circt::sva::createBooleanSequence(uint32_t conditionIndex) {
  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(false);
  uint32_t accepting = automaton->addState(true);
  automaton->addTransition(initial, accepting, conditionIndex);
  automaton->setInitialState(initial);
  return automaton;
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createDelaySequence(std::unique_ptr<SequenceAutomaton> input,
                                 uint32_t delay) {
  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(false);
  automaton->setInitialState(initial);

  // Create delay states
  uint32_t current = initial;
  for (uint32_t i = 0; i < delay; ++i) {
    uint32_t next = automaton->addState(false);
    automaton->addTransition(current, next, -1); // Epsilon transition
    current = next;
  }

  // Final accepting state
  uint32_t accepting = automaton->addState(true);
  automaton->addTransition(current, accepting, -1);

  return automaton;
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createRangeDelaySequence(std::unique_ptr<SequenceAutomaton> input,
                                      uint32_t minDelay, uint32_t maxDelay) {
  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(false);
  automaton->setInitialState(initial);

  // Create delay states with multiple accepting states for the range
  uint32_t current = initial;
  for (uint32_t i = 0; i <= maxDelay; ++i) {
    bool isAccepting = (i >= minDelay);
    uint32_t next = automaton->addState(isAccepting);
    automaton->addTransition(current, next, -1);
    current = next;
  }

  return automaton;
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createRepeatSequence(std::unique_ptr<SequenceAutomaton> input,
                                  uint32_t count) {
  if (count == 0) {
    // Zero repetition is an empty sequence (always matches)
    auto automaton = std::make_unique<SequenceAutomaton>();
    uint32_t accepting = automaton->addState(true);
    automaton->setInitialState(accepting);
    return automaton;
  }

  // For now, return the input for count=1, otherwise create chain
  if (count == 1)
    return std::move(input);

  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(false);
  automaton->setInitialState(initial);

  uint32_t current = initial;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t next = automaton->addState(i == count - 1);
    automaton->addTransition(current, next, 0); // Use condition 0 as placeholder
    current = next;
  }

  return automaton;
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createRangeRepeatSequence(std::unique_ptr<SequenceAutomaton> input,
                                       uint32_t minCount, uint32_t maxCount) {
  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(minCount == 0);
  automaton->setInitialState(initial);

  uint32_t current = initial;
  for (uint32_t i = 1; i <= maxCount; ++i) {
    bool isAccepting = (i >= minCount);
    uint32_t next = automaton->addState(isAccepting);
    automaton->addTransition(current, next, 0);
    current = next;
  }

  return automaton;
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createConcatSequence(std::unique_ptr<SequenceAutomaton> first,
                                  std::unique_ptr<SequenceAutomaton> second) {
  // Simplified: just return second for now
  // A proper implementation would merge the automata
  return std::move(second);
}

std::unique_ptr<SequenceAutomaton>
circt::sva::createOrSequence(std::unique_ptr<SequenceAutomaton> first,
                              std::unique_ptr<SequenceAutomaton> second) {
  // Simplified: create new automaton with epsilon transitions to both
  auto automaton = std::make_unique<SequenceAutomaton>();
  uint32_t initial = automaton->addState(false);
  uint32_t accepting = automaton->addState(true);
  automaton->setInitialState(initial);
  // Would need to properly merge the two automata
  automaton->addTransition(initial, accepting, -1);
  return automaton;
}
