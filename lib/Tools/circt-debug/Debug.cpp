//===- Debug.cpp - CIRCT Debug Infrastructure Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-debug/Debug.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <cmath>
#include <sstream>

using namespace circt;
using namespace circt::debug;

//===----------------------------------------------------------------------===//
// SignalValue Implementation
//===----------------------------------------------------------------------===//

SignalValue::SignalValue(unsigned width) : width(width) {
  unsigned numBytes = (width + 7) / 8;
  valueBits.resize(numBytes, 0);
  unknownBits.resize(numBytes, 0);
  highzBits.resize(numBytes, 0);
}

SignalValue::SignalValue(const llvm::APInt &value) : width(value.getBitWidth()) {
  unsigned numBytes = (width + 7) / 8;
  valueBits.resize(numBytes, 0);
  unknownBits.resize(numBytes, 0);
  highzBits.resize(numBytes, 0);

  // Copy value bits
  for (unsigned i = 0; i < width; ++i) {
    if (value[i])
      valueBits[i / 8] |= (1 << (i % 8));
  }
}

SignalValue::SignalValue(unsigned width, uint64_t value) : width(width) {
  unsigned numBytes = (width + 7) / 8;
  valueBits.resize(numBytes, 0);
  unknownBits.resize(numBytes, 0);
  highzBits.resize(numBytes, 0);

  for (unsigned i = 0; i < std::min(width, 64u); ++i) {
    if (value & (1ULL << i))
      valueBits[i / 8] |= (1 << (i % 8));
  }
}

LogicValue SignalValue::getBit(unsigned index) const {
  if (index >= width)
    return LogicValue::Unknown;

  unsigned byteIdx = index / 8;
  unsigned bitIdx = index % 8;

  if (highzBits[byteIdx] & (1 << bitIdx))
    return LogicValue::HighZ;
  if (unknownBits[byteIdx] & (1 << bitIdx))
    return LogicValue::Unknown;
  if (valueBits[byteIdx] & (1 << bitIdx))
    return LogicValue::One;
  return LogicValue::Zero;
}

void SignalValue::setBit(unsigned index, LogicValue value) {
  if (index >= width)
    return;

  unsigned byteIdx = index / 8;
  unsigned bitIdx = index % 8;
  uint8_t mask = 1 << bitIdx;

  // Clear all bits first
  valueBits[byteIdx] &= ~mask;
  unknownBits[byteIdx] &= ~mask;
  highzBits[byteIdx] &= ~mask;

  switch (value) {
  case LogicValue::Zero:
    break;
  case LogicValue::One:
    valueBits[byteIdx] |= mask;
    break;
  case LogicValue::Unknown:
    unknownBits[byteIdx] |= mask;
    break;
  case LogicValue::HighZ:
    highzBits[byteIdx] |= mask;
    break;
  }
}

bool SignalValue::hasUnknown() const {
  for (uint8_t b : unknownBits)
    if (b)
      return true;
  return false;
}

bool SignalValue::hasHighZ() const {
  for (uint8_t b : highzBits)
    if (b)
      return true;
  return false;
}

bool SignalValue::isFullyDefined() const {
  return !hasUnknown() && !hasHighZ();
}

llvm::APInt SignalValue::toAPInt() const {
  llvm::APInt result(width, 0);
  for (unsigned i = 0; i < width; ++i) {
    if (getBit(i) == LogicValue::One)
      result.setBit(i);
  }
  return result;
}

std::string SignalValue::toString(unsigned radix) const {
  if (radix == 2)
    return toBinaryString();
  if (radix == 16)
    return toHexString();

  // For decimal, just convert the APInt
  if (isFullyDefined())
    return toAPInt().toString(radix, false);

  // If has unknowns, show binary
  return toBinaryString();
}

std::string SignalValue::toHexString() const {
  if (!isFullyDefined()) {
    // Show binary if has X or Z
    return toBinaryString();
  }

  std::string result;
  unsigned numNibbles = (width + 3) / 4;

  for (int i = numNibbles - 1; i >= 0; --i) {
    unsigned nibbleValue = 0;
    for (unsigned b = 0; b < 4; ++b) {
      unsigned bitIdx = i * 4 + b;
      if (bitIdx < width && getBit(bitIdx) == LogicValue::One)
        nibbleValue |= (1 << b);
    }
    result += "0123456789abcdef"[nibbleValue];
  }

  return result;
}

std::string SignalValue::toBinaryString() const {
  std::string result;
  result.reserve(width);

  for (int i = width - 1; i >= 0; --i) {
    switch (getBit(i)) {
    case LogicValue::Zero:
      result += '0';
      break;
    case LogicValue::One:
      result += '1';
      break;
    case LogicValue::Unknown:
      result += 'x';
      break;
    case LogicValue::HighZ:
      result += 'z';
      break;
    }
  }

  return result;
}

std::optional<SignalValue> SignalValue::fromString(StringRef str,
                                                   unsigned width) {
  SignalValue result(width);

  // Check for radix prefix
  StringRef valueStr = str;
  unsigned radix = 10;

  if (str.starts_with("0x") || str.starts_with("0X")) {
    radix = 16;
    valueStr = str.drop_front(2);
  } else if (str.starts_with("0b") || str.starts_with("0B")) {
    radix = 2;
    valueStr = str.drop_front(2);
  } else if (str.starts_with("'h") || str.starts_with("'H")) {
    radix = 16;
    valueStr = str.drop_front(2);
  } else if (str.starts_with("'b") || str.starts_with("'B")) {
    radix = 2;
    valueStr = str.drop_front(2);
  } else if (str.starts_with("'d") || str.starts_with("'D")) {
    radix = 10;
    valueStr = str.drop_front(2);
  }

  if (radix == 2) {
    // Binary - can have x and z
    for (int i = valueStr.size() - 1, bit = 0; i >= 0 && bit < (int)width;
         --i, ++bit) {
      char c = valueStr[i];
      if (c == '0')
        result.setBit(bit, LogicValue::Zero);
      else if (c == '1')
        result.setBit(bit, LogicValue::One);
      else if (c == 'x' || c == 'X')
        result.setBit(bit, LogicValue::Unknown);
      else if (c == 'z' || c == 'Z')
        result.setBit(bit, LogicValue::HighZ);
      else if (c != '_')
        return std::nullopt;
    }
  } else if (radix == 16) {
    // Hex
    for (int i = valueStr.size() - 1, nibble = 0; i >= 0 && nibble * 4 < width;
         --i, ++nibble) {
      char c = valueStr[i];
      if (c == 'x' || c == 'X') {
        for (unsigned b = 0; b < 4 && nibble * 4 + b < width; ++b)
          result.setBit(nibble * 4 + b, LogicValue::Unknown);
      } else if (c == 'z' || c == 'Z') {
        for (unsigned b = 0; b < 4 && nibble * 4 + b < width; ++b)
          result.setBit(nibble * 4 + b, LogicValue::HighZ);
      } else if (c != '_') {
        unsigned nibbleValue;
        if (c >= '0' && c <= '9')
          nibbleValue = c - '0';
        else if (c >= 'a' && c <= 'f')
          nibbleValue = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F')
          nibbleValue = c - 'A' + 10;
        else
          return std::nullopt;

        for (unsigned b = 0; b < 4 && nibble * 4 + b < width; ++b) {
          result.setBit(nibble * 4 + b, (nibbleValue & (1 << b))
                                            ? LogicValue::One
                                            : LogicValue::Zero);
        }
      }
    }
  } else {
    // Decimal
    llvm::APInt value;
    if (valueStr.getAsInteger(10, value))
      return std::nullopt;

    for (unsigned i = 0; i < width && i < value.getBitWidth(); ++i) {
      result.setBit(i, value[i] ? LogicValue::One : LogicValue::Zero);
    }
  }

  return result;
}

bool SignalValue::operator==(const SignalValue &other) const {
  if (width != other.width)
    return false;
  return valueBits == other.valueBits && unknownBits == other.unknownBits &&
         highzBits == other.highzBits;
}

//===----------------------------------------------------------------------===//
// SignalInfo Implementation
//===----------------------------------------------------------------------===//

StringRef SignalInfo::getTypeString() const {
  switch (type) {
  case SignalType::Wire:
    return "wire";
  case SignalType::Reg:
    return "reg";
  case SignalType::Input:
    return "input";
  case SignalType::Output:
    return "output";
  case SignalType::InOut:
    return "inout";
  case SignalType::Memory:
    return "memory";
  case SignalType::Parameter:
    return "parameter";
  case SignalType::LocalParam:
    return "localparam";
  }
  return "unknown";
}

//===----------------------------------------------------------------------===//
// Scope Implementation
//===----------------------------------------------------------------------===//

Scope::Scope(StringRef name, Scope *parent) : name(name.str()), parent(parent) {}

std::string Scope::getFullPath() const {
  if (!parent)
    return name;
  return parent->getFullPath() + "." + name;
}

void Scope::addChild(std::unique_ptr<Scope> child) {
  childIndex[child->getName()] = children.size();
  children.push_back(std::move(child));
}

Scope *Scope::findChild(StringRef name) const {
  auto it = childIndex.find(name);
  if (it == childIndex.end())
    return nullptr;
  return children[it->second].get();
}

void Scope::addSignal(const SignalInfo &signal) {
  signalIndex[signal.name] = signals.size();
  signals.push_back(signal);
}

const SignalInfo *Scope::findSignal(StringRef name) const {
  auto it = signalIndex.find(name);
  if (it == signalIndex.end())
    return nullptr;
  return &signals[it->second];
}

const SignalInfo *Scope::findSignalByPath(StringRef path) const {
  // Split path by '.'
  auto dotPos = path.find('.');
  if (dotPos == StringRef::npos) {
    // No more dots - look for signal in current scope
    return findSignal(path);
  }

  // Has dots - navigate to child scope
  StringRef childName = path.take_front(dotPos);
  StringRef rest = path.drop_front(dotPos + 1);

  Scope *child = findChild(childName);
  if (!child)
    return nullptr;

  return child->findSignalByPath(rest);
}

//===----------------------------------------------------------------------===//
// SimTime Implementation
//===----------------------------------------------------------------------===//

double SimTime::toNanoseconds() const {
  switch (unit) {
  case FS:
    return value * 1e-6;
  case PS:
    return value * 1e-3;
  case NS:
    return value;
  case US:
    return value * 1e3;
  case MS:
    return value * 1e6;
  case S:
    return value * 1e9;
  }
  return value;
}

std::string SimTime::toString() const {
  std::string result = std::to_string(value);
  switch (unit) {
  case FS:
    result += "fs";
    break;
  case PS:
    result += "ps";
    break;
  case NS:
    result += "ns";
    break;
  case US:
    result += "us";
    break;
  case MS:
    result += "ms";
    break;
  case S:
    result += "s";
    break;
  }
  return result;
}

bool SimTime::operator<(const SimTime &other) const {
  return toNanoseconds() < other.toNanoseconds();
}

bool SimTime::operator<=(const SimTime &other) const {
  return toNanoseconds() <= other.toNanoseconds();
}

bool SimTime::operator==(const SimTime &other) const {
  return std::abs(toNanoseconds() - other.toNanoseconds()) < 1e-12;
}

SimTime SimTime::operator+(const SimTime &other) const {
  // Use the smaller unit for precision
  if (unit <= other.unit)
    return SimTime(value + other.value * static_cast<uint64_t>(
                                             std::pow(1000, other.unit - unit)),
                   unit);
  return other + *this;
}

SimTime &SimTime::operator+=(const SimTime &other) {
  *this = *this + other;
  return *this;
}

//===----------------------------------------------------------------------===//
// SimState Implementation
//===----------------------------------------------------------------------===//

SimState::SimState() = default;
SimState::~SimState() = default;

void SimState::advanceTime(const SimTime &delta) { currentTime += delta; }

SignalValue SimState::getSignalValue(StringRef path) const {
  auto it = signalValues.find(path);
  if (it != signalValues.end())
    return it->second;
  return SignalValue(1); // Default to 1-bit unknown
}

void SimState::setSignalValue(StringRef path, const SignalValue &value) {
  auto it = signalValues.find(path);
  if (it != signalValues.end()) {
    if (it->second != value) {
      SignalChange change;
      change.path = path.str();
      change.oldValue = it->second;
      change.newValue = value;
      change.time = currentTime;
      recordChange(change);
      it->second = value;
    }
  } else {
    signalValues[path] = value;
  }
}

bool SimState::hasSignal(StringRef path) const {
  return signalValues.count(path) > 0;
}

void SimState::setRootScope(std::unique_ptr<Scope> scope) {
  rootScope = std::move(scope);
  currentScope = rootScope.get();
}

void SimState::recordChange(const SignalChange &change) {
  recentChanges.push_back(change);
}

void SimState::setCurrentLocation(StringRef file, unsigned line) {
  currentFile = file.str();
  currentLine = line;
}

std::optional<std::pair<std::string, unsigned>>
SimState::getCurrentLocation() const {
  if (currentFile && currentLine)
    return std::make_pair(*currentFile, *currentLine);
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Breakpoint Implementations
//===----------------------------------------------------------------------===//

Breakpoint::Breakpoint(Type type, unsigned id) : type(type), id(id) {}

LineBreakpoint::LineBreakpoint(unsigned id, StringRef file, unsigned line)
    : Breakpoint(Type::Line, id), file(file.str()), line(line) {}

bool LineBreakpoint::shouldBreak(const SimState &state) const {
  if (!enabled)
    return false;

  auto loc = state.getCurrentLocation();
  if (!loc)
    return false;

  return loc->first == file && loc->second == line;
}

std::string LineBreakpoint::getDescription() const {
  return "at " + file + ":" + std::to_string(line);
}

SignalBreakpoint::SignalBreakpoint(unsigned id, StringRef signal)
    : Breakpoint(Type::Signal, id), signal(signal.str()) {}

SignalBreakpoint::SignalBreakpoint(unsigned id, StringRef signal,
                                   const SignalValue &value)
    : Breakpoint(Type::Signal, id), signal(signal.str()), targetValue(value) {}

SignalBreakpoint::SignalBreakpoint(unsigned id, StringRef signal, Edge edge)
    : Breakpoint(Type::Signal, id), signal(signal.str()), edge(edge) {}

bool SignalBreakpoint::shouldBreak(const SimState &state) const {
  if (!enabled)
    return false;

  SignalValue currentValue = state.getSignalValue(signal);

  if (!previousValue) {
    previousValue = currentValue;
    return false;
  }

  bool changed = currentValue != *previousValue;
  if (!changed)
    return false;

  // Check specific conditions
  if (targetValue) {
    return currentValue == *targetValue;
  }

  if (edge != Edge::Any && currentValue.getWidth() >= 1 &&
      previousValue->getWidth() >= 1) {
    bool wasOne = previousValue->getBit(0) == LogicValue::One;
    bool isOne = currentValue.getBit(0) == LogicValue::One;

    if (edge == Edge::Rising)
      return !wasOne && isOne;
    if (edge == Edge::Falling)
      return wasOne && !isOne;
  }

  return changed;
}

std::string SignalBreakpoint::getDescription() const {
  std::string desc = "on signal " + signal;
  if (targetValue)
    desc += " == " + targetValue->toString();
  else if (edge == Edge::Rising)
    desc += " (rising edge)";
  else if (edge == Edge::Falling)
    desc += " (falling edge)";
  return desc;
}

void SignalBreakpoint::updatePreviousValue(const SignalValue &value) {
  previousValue = value;
}

ConditionBreakpoint::ConditionBreakpoint(unsigned id, StringRef expression)
    : Breakpoint(Type::Condition, id), expression(expression.str()) {}

bool ConditionBreakpoint::shouldBreak(const SimState &state) const {
  if (!enabled)
    return false;

  if (evaluator)
    return evaluator(state, expression);

  // Default: try to parse as simple signal comparison
  ExpressionEvaluator eval(state);
  return eval.isTrue(expression);
}

std::string ConditionBreakpoint::getDescription() const {
  return "when " + expression;
}

TimeBreakpoint::TimeBreakpoint(unsigned id, const SimTime &time)
    : Breakpoint(Type::Time, id), targetTime(time) {}

bool TimeBreakpoint::shouldBreak(const SimState &state) const {
  if (!enabled)
    return false;
  return state.getTime() == targetTime;
}

std::string TimeBreakpoint::getDescription() const {
  return "at time " + targetTime.toString();
}

CycleBreakpoint::CycleBreakpoint(unsigned id, uint64_t cycle)
    : Breakpoint(Type::Cycle, id), targetCycle(cycle) {}

bool CycleBreakpoint::shouldBreak(const SimState &state) const {
  if (!enabled)
    return false;
  return state.getCycle() == targetCycle;
}

std::string CycleBreakpoint::getDescription() const {
  return "at cycle " + std::to_string(targetCycle);
}

//===----------------------------------------------------------------------===//
// Watchpoint Implementation
//===----------------------------------------------------------------------===//

Watchpoint::Watchpoint(unsigned id, StringRef signal)
    : id(id), signal(signal.str()) {}

bool Watchpoint::checkAndRecord(const SimState &state) {
  if (!enabled)
    return false;

  SignalValue currentValue = state.getSignalValue(signal);

  if (!lastValue || currentValue != *lastValue) {
    history.push_back({state.getTime(), currentValue});
    lastValue = currentValue;
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// BreakpointManager Implementation
//===----------------------------------------------------------------------===//

BreakpointManager::BreakpointManager() = default;
BreakpointManager::~BreakpointManager() = default;

unsigned BreakpointManager::addLineBreakpoint(StringRef file, unsigned line) {
  auto bp =
      std::make_unique<LineBreakpoint>(nextBreakpointId++, file, line);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addSignalBreakpoint(StringRef signal) {
  auto bp = std::make_unique<SignalBreakpoint>(nextBreakpointId++, signal);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addSignalBreakpoint(StringRef signal,
                                                const SignalValue &value) {
  auto bp =
      std::make_unique<SignalBreakpoint>(nextBreakpointId++, signal, value);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned
BreakpointManager::addSignalBreakpoint(StringRef signal,
                                       SignalBreakpoint::Edge edge) {
  auto bp =
      std::make_unique<SignalBreakpoint>(nextBreakpointId++, signal, edge);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addConditionBreakpoint(StringRef expression) {
  auto bp =
      std::make_unique<ConditionBreakpoint>(nextBreakpointId++, expression);
  if (conditionEvaluator)
    bp->setEvaluator(conditionEvaluator);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addTimeBreakpoint(const SimTime &time) {
  auto bp = std::make_unique<TimeBreakpoint>(nextBreakpointId++, time);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addCycleBreakpoint(uint64_t cycle) {
  auto bp = std::make_unique<CycleBreakpoint>(nextBreakpointId++, cycle);
  breakpoints.push_back(std::move(bp));
  return breakpoints.back()->getId();
}

unsigned BreakpointManager::addWatchpoint(StringRef signal) {
  auto wp = std::make_unique<Watchpoint>(nextWatchpointId++, signal);
  watchpoints.push_back(std::move(wp));
  return watchpoints.back()->getId();
}

bool BreakpointManager::removeBreakpoint(unsigned id) {
  auto it = std::find_if(breakpoints.begin(), breakpoints.end(),
                         [id](const auto &bp) { return bp->getId() == id; });
  if (it == breakpoints.end())
    return false;
  breakpoints.erase(it);
  return true;
}

bool BreakpointManager::removeWatchpoint(unsigned id) {
  auto it = std::find_if(watchpoints.begin(), watchpoints.end(),
                         [id](const auto &wp) { return wp->getId() == id; });
  if (it == watchpoints.end())
    return false;
  watchpoints.erase(it);
  return true;
}

void BreakpointManager::removeAllBreakpoints() { breakpoints.clear(); }

void BreakpointManager::removeAllWatchpoints() { watchpoints.clear(); }

bool BreakpointManager::enableBreakpoint(unsigned id, bool enable) {
  auto bp = getBreakpoint(id);
  if (!bp)
    return false;
  bp->setEnabled(enable);
  return true;
}

bool BreakpointManager::enableWatchpoint(unsigned id, bool enable) {
  auto wp = getWatchpoint(id);
  if (!wp)
    return false;
  wp->setEnabled(enable);
  return true;
}

Breakpoint *BreakpointManager::getBreakpoint(unsigned id) {
  auto it = std::find_if(breakpoints.begin(), breakpoints.end(),
                         [id](const auto &bp) { return bp->getId() == id; });
  return it != breakpoints.end() ? it->get() : nullptr;
}

const Breakpoint *BreakpointManager::getBreakpoint(unsigned id) const {
  auto it = std::find_if(breakpoints.begin(), breakpoints.end(),
                         [id](const auto &bp) { return bp->getId() == id; });
  return it != breakpoints.end() ? it->get() : nullptr;
}

Watchpoint *BreakpointManager::getWatchpoint(unsigned id) {
  auto it = std::find_if(watchpoints.begin(), watchpoints.end(),
                         [id](const auto &wp) { return wp->getId() == id; });
  return it != watchpoints.end() ? it->get() : nullptr;
}

const Watchpoint *BreakpointManager::getWatchpoint(unsigned id) const {
  auto it = std::find_if(watchpoints.begin(), watchpoints.end(),
                         [id](const auto &wp) { return wp->getId() == id; });
  return it != watchpoints.end() ? it->get() : nullptr;
}

bool BreakpointManager::shouldBreak(const SimState &state) const {
  for (const auto &bp : breakpoints) {
    if (bp->shouldBreak(state))
      return true;
  }
  return false;
}

llvm::SmallVector<Breakpoint *>
BreakpointManager::getTriggeredBreakpoints(const SimState &state) const {
  llvm::SmallVector<Breakpoint *> triggered;
  for (const auto &bp : breakpoints) {
    if (bp->shouldBreak(state))
      triggered.push_back(bp.get());
  }
  return triggered;
}

void BreakpointManager::updateWatchpoints(const SimState &state) {
  for (auto &wp : watchpoints) {
    wp->checkAndRecord(state);
  }
}

void BreakpointManager::setConditionEvaluator(
    ConditionBreakpoint::ConditionEvaluator eval) {
  conditionEvaluator = std::move(eval);
  for (auto &bp : breakpoints) {
    if (auto *cond = dynamic_cast<ConditionBreakpoint *>(bp.get()))
      cond->setEvaluator(conditionEvaluator);
  }
}

//===----------------------------------------------------------------------===//
// ExpressionEvaluator Implementation
//===----------------------------------------------------------------------===//

ExpressionEvaluator::ExpressionEvaluator(const SimState &state)
    : state(state) {}

EvalResult ExpressionEvaluator::evaluate(StringRef expr) const {
  StringRef trimmed = expr.trim();

  // Try to parse as constant
  auto constVal = parseConstant(trimmed);
  if (constVal)
    return EvalResult::success(*constVal);

  // Try to get as signal
  auto sigVal = getSignalValue(trimmed);
  if (sigVal)
    return EvalResult::success(*sigVal);

  return EvalResult::failure("unknown signal or invalid expression: " +
                             trimmed.str());
}

EvalResult ExpressionEvaluator::evaluateComparison(StringRef expr) const {
  // Find comparison operator
  size_t opPos = StringRef::npos;
  StringRef op;

  for (auto candidate : {"==", "!=", ">=", "<=", ">", "<"}) {
    size_t pos = expr.find(candidate);
    if (pos != StringRef::npos && (opPos == StringRef::npos || pos < opPos)) {
      opPos = pos;
      op = candidate;
    }
  }

  if (opPos == StringRef::npos) {
    // Not a comparison - treat as boolean expression
    auto result = evaluate(expr);
    if (!result.success)
      return result;
    // Non-zero is true
    SignalValue oneVal(1, 1);
    SignalValue zeroVal(1, 0);
    bool isTrue = result.value && result.value->isFullyDefined() &&
                  result.value->toAPInt().getBoolValue();
    return EvalResult::success(isTrue ? oneVal : zeroVal);
  }

  // Split at operator
  StringRef lhs = expr.take_front(opPos).trim();
  StringRef rhs = expr.drop_front(opPos + op.size()).trim();

  auto lhsResult = evaluate(lhs);
  if (!lhsResult.success)
    return lhsResult;

  auto rhsResult = evaluate(rhs);
  if (!rhsResult.success)
    return rhsResult;

  // Compare
  bool result = false;
  if (!lhsResult.value->isFullyDefined() || !rhsResult.value->isFullyDefined()) {
    // X comparison - always false
    result = false;
  } else {
    llvm::APInt lhsInt = lhsResult.value->toAPInt();
    llvm::APInt rhsInt = rhsResult.value->toAPInt();

    // Extend to same width
    unsigned maxWidth = std::max(lhsInt.getBitWidth(), rhsInt.getBitWidth());
    lhsInt = lhsInt.zext(maxWidth);
    rhsInt = rhsInt.zext(maxWidth);

    if (op == "==")
      result = lhsInt == rhsInt;
    else if (op == "!=")
      result = lhsInt != rhsInt;
    else if (op == ">")
      result = lhsInt.ugt(rhsInt);
    else if (op == "<")
      result = lhsInt.ult(rhsInt);
    else if (op == ">=")
      result = lhsInt.uge(rhsInt);
    else if (op == "<=")
      result = lhsInt.ule(rhsInt);
  }

  return EvalResult::success(SignalValue(1, result ? 1 : 0));
}

bool ExpressionEvaluator::isTrue(StringRef expr) const {
  auto result = evaluateComparison(expr);
  if (!result.success || !result.value)
    return false;
  return result.value->isFullyDefined() &&
         result.value->toAPInt().getBoolValue();
}

std::optional<SignalValue>
ExpressionEvaluator::parseConstant(StringRef str) const {
  // Handle Verilog-style constants: 8'hFF, 4'b1010, etc.
  auto tickPos = str.find('\'');
  if (tickPos != StringRef::npos) {
    unsigned width = 32;
    if (tickPos > 0) {
      if (str.take_front(tickPos).getAsInteger(10, width))
        return std::nullopt;
    }
    return SignalValue::fromString(str.drop_front(tickPos), width);
  }

  // Handle simple hex: 0xABCD
  if (str.starts_with("0x") || str.starts_with("0X")) {
    return SignalValue::fromString(str, 64);
  }

  // Handle simple binary: 0b1010
  if (str.starts_with("0b") || str.starts_with("0B")) {
    return SignalValue::fromString(str, str.size() - 2);
  }

  // Handle decimal
  uint64_t value;
  if (!str.getAsInteger(10, value)) {
    return SignalValue(64, value);
  }

  return std::nullopt;
}

std::optional<SignalValue>
ExpressionEvaluator::getSignalValue(StringRef name) const {
  if (state.hasSignal(name))
    return state.getSignalValue(name);

  // Try with current scope prefix
  const Scope *scope = state.getCurrentScope();
  if (scope) {
    std::string fullPath = scope->getFullPath() + "." + name.str();
    if (state.hasSignal(fullPath))
      return state.getSignalValue(fullPath);
  }

  return std::nullopt;
}
