//===- Debug.h - CIRCT Debug Infrastructure ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the core debug infrastructure for the CIRCT
// interactive debugger, including simulation state management, breakpoint
// handling, and signal inspection.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_DEBUG_DEBUG_H
#define CIRCT_TOOLS_CIRCT_DEBUG_DEBUG_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace circt {
namespace debug {

//===----------------------------------------------------------------------===//
// Signal Value Representation
//===----------------------------------------------------------------------===//

/// Represents a 4-state logic value (0, 1, X, Z) for a single bit.
enum class LogicValue : uint8_t {
  Zero = 0,
  One = 1,
  Unknown = 2, // X
  HighZ = 3    // Z
};

/// Represents a multi-bit signal value with 4-state logic support.
class SignalValue {
public:
  SignalValue() = default;
  explicit SignalValue(unsigned width);
  SignalValue(const llvm::APInt &value);
  SignalValue(unsigned width, uint64_t value);

  /// Get the width of the signal in bits.
  unsigned getWidth() const { return width; }

  /// Get/set individual bit values.
  LogicValue getBit(unsigned index) const;
  void setBit(unsigned index, LogicValue value);

  /// Check if the value contains any unknown (X) or high-impedance (Z) bits.
  bool hasUnknown() const;
  bool hasHighZ() const;
  bool isFullyDefined() const;

  /// Convert to APInt (X/Z bits become 0).
  llvm::APInt toAPInt() const;

  /// Convert to string representation.
  std::string toString(unsigned radix = 2) const;
  std::string toHexString() const;
  std::string toBinaryString() const;

  /// Parse from string.
  static std::optional<SignalValue> fromString(StringRef str, unsigned width);

  /// Comparison operators.
  bool operator==(const SignalValue &other) const;
  bool operator!=(const SignalValue &other) const { return !(*this == other); }

private:
  unsigned width = 0;
  llvm::SmallVector<uint8_t> valueBits;   // 0 or 1 for each bit
  llvm::SmallVector<uint8_t> unknownBits; // 1 if X, 0 otherwise
  llvm::SmallVector<uint8_t> highzBits;   // 1 if Z, 0 otherwise
};

//===----------------------------------------------------------------------===//
// Signal Information
//===----------------------------------------------------------------------===//

/// Types of signals in the design.
enum class SignalType {
  Wire,
  Reg,
  Input,
  Output,
  InOut,
  Memory,
  Parameter,
  LocalParam
};

/// Information about a signal in the design.
struct SignalInfo {
  std::string name;
  std::string fullPath;
  SignalType type;
  unsigned width;
  bool isSigned;

  /// For arrays/memories.
  std::optional<unsigned> arraySize;
  std::optional<unsigned> arrayStart;

  /// Source location if available.
  std::optional<std::string> sourceFile;
  std::optional<unsigned> sourceLine;

  /// String representation of signal type.
  StringRef getTypeString() const;
};

//===----------------------------------------------------------------------===//
// Scope Navigation
//===----------------------------------------------------------------------===//

/// Represents a hierarchical scope in the design.
class Scope {
public:
  Scope(StringRef name, Scope *parent = nullptr);

  StringRef getName() const { return name; }
  std::string getFullPath() const;
  Scope *getParent() const { return parent; }

  /// Child scope management.
  void addChild(std::unique_ptr<Scope> child);
  Scope *findChild(StringRef name) const;
  const std::vector<std::unique_ptr<Scope>> &getChildren() const {
    return children;
  }

  /// Signal management within this scope.
  void addSignal(const SignalInfo &signal);
  const SignalInfo *findSignal(StringRef name) const;
  const std::vector<SignalInfo> &getSignals() const { return signals; }

  /// Find a signal by hierarchical path relative to this scope.
  const SignalInfo *findSignalByPath(StringRef path) const;

private:
  std::string name;
  Scope *parent;
  std::vector<std::unique_ptr<Scope>> children;
  std::vector<SignalInfo> signals;
  llvm::StringMap<size_t> signalIndex;
  llvm::StringMap<size_t> childIndex;
};

//===----------------------------------------------------------------------===//
// Simulation Time
//===----------------------------------------------------------------------===//

/// Represents simulation time with precision.
struct SimTime {
  uint64_t value = 0;
  enum Unit { FS, PS, NS, US, MS, S } unit = NS;

  SimTime() = default;
  SimTime(uint64_t v, Unit u = NS) : value(v), unit(u) {}

  /// Convert to nanoseconds.
  double toNanoseconds() const;

  /// Convert to string.
  std::string toString() const;

  /// Comparison operators.
  bool operator<(const SimTime &other) const;
  bool operator<=(const SimTime &other) const;
  bool operator>(const SimTime &other) const { return other < *this; }
  bool operator>=(const SimTime &other) const { return other <= *this; }
  bool operator==(const SimTime &other) const;
  bool operator!=(const SimTime &other) const { return !(*this == other); }

  /// Arithmetic operators.
  SimTime operator+(const SimTime &other) const;
  SimTime &operator+=(const SimTime &other);
};

//===----------------------------------------------------------------------===//
// Simulation State
//===----------------------------------------------------------------------===//

/// Represents the complete simulation state at a point in time.
class SimState {
public:
  SimState();
  ~SimState();

  /// Current simulation time.
  const SimTime &getTime() const { return currentTime; }
  void setTime(const SimTime &time) { currentTime = time; }
  void advanceTime(const SimTime &delta);

  /// Current simulation cycle (clock cycles).
  uint64_t getCycle() const { return currentCycle; }
  void setCycle(uint64_t cycle) { currentCycle = cycle; }
  void advanceCycle(uint64_t n = 1) { currentCycle += n; }

  /// Delta cycle within current time step.
  uint64_t getDeltaCycle() const { return deltaCycle; }
  void setDeltaCycle(uint64_t delta) { deltaCycle = delta; }
  void advanceDeltaCycle() { ++deltaCycle; }
  void resetDeltaCycle() { deltaCycle = 0; }

  /// Signal value access.
  SignalValue getSignalValue(StringRef path) const;
  void setSignalValue(StringRef path, const SignalValue &value);
  bool hasSignal(StringRef path) const;

  /// Scope hierarchy.
  Scope *getRootScope() { return rootScope.get(); }
  const Scope *getRootScope() const { return rootScope.get(); }
  void setRootScope(std::unique_ptr<Scope> scope);

  /// Current scope for navigation.
  Scope *getCurrentScope() { return currentScope; }
  const Scope *getCurrentScope() const { return currentScope; }
  void setCurrentScope(Scope *scope) { currentScope = scope; }

  /// Signal change tracking.
  struct SignalChange {
    std::string path;
    SignalValue oldValue;
    SignalValue newValue;
    SimTime time;
  };
  const std::vector<SignalChange> &getRecentChanges() const {
    return recentChanges;
  }
  void recordChange(const SignalChange &change);
  void clearChanges() { recentChanges.clear(); }

  /// Source location tracking.
  void setCurrentLocation(StringRef file, unsigned line);
  std::optional<std::pair<std::string, unsigned>> getCurrentLocation() const;

private:
  SimTime currentTime;
  uint64_t currentCycle = 0;
  uint64_t deltaCycle = 0;

  std::unique_ptr<Scope> rootScope;
  Scope *currentScope = nullptr;

  llvm::StringMap<SignalValue> signalValues;
  std::vector<SignalChange> recentChanges;

  std::optional<std::string> currentFile;
  std::optional<unsigned> currentLine;
};

//===----------------------------------------------------------------------===//
// Breakpoint Types
//===----------------------------------------------------------------------===//

/// Base class for all breakpoint types.
class Breakpoint {
public:
  enum class Type {
    Line,      // Break at source line
    Signal,    // Break on signal change
    Condition, // Break when condition is true
    Time,      // Break at specific time
    Cycle      // Break at specific cycle
  };

  Breakpoint(Type type, unsigned id);
  virtual ~Breakpoint() = default;

  Type getType() const { return type; }
  unsigned getId() const { return id; }

  bool isEnabled() const { return enabled; }
  void setEnabled(bool e) { enabled = e; }

  unsigned getHitCount() const { return hitCount; }
  void incrementHitCount() { ++hitCount; }
  void resetHitCount() { hitCount = 0; }

  /// Check if this breakpoint should trigger given the current state.
  virtual bool shouldBreak(const SimState &state) const = 0;

  /// Get a description of the breakpoint.
  virtual std::string getDescription() const = 0;

protected:
  Type type;
  unsigned id;
  bool enabled = true;
  unsigned hitCount = 0;
};

/// Breakpoint at a source line.
class LineBreakpoint : public Breakpoint {
public:
  LineBreakpoint(unsigned id, StringRef file, unsigned line);

  StringRef getFile() const { return file; }
  unsigned getLine() const { return line; }

  bool shouldBreak(const SimState &state) const override;
  std::string getDescription() const override;

private:
  std::string file;
  unsigned line;
};

/// Breakpoint on signal value change.
class SignalBreakpoint : public Breakpoint {
public:
  /// Break on any change to the signal.
  SignalBreakpoint(unsigned id, StringRef signal);

  /// Break when signal changes to a specific value.
  SignalBreakpoint(unsigned id, StringRef signal, const SignalValue &value);

  /// Break on rising/falling edge.
  enum class Edge { Any, Rising, Falling };
  SignalBreakpoint(unsigned id, StringRef signal, Edge edge);

  StringRef getSignal() const { return signal; }

  bool shouldBreak(const SimState &state) const override;
  std::string getDescription() const override;

  /// Update the previous value (call after checking).
  void updatePreviousValue(const SignalValue &value);

private:
  std::string signal;
  std::optional<SignalValue> targetValue;
  Edge edge = Edge::Any;
  mutable std::optional<SignalValue> previousValue;
};

/// Breakpoint with a condition expression.
class ConditionBreakpoint : public Breakpoint {
public:
  ConditionBreakpoint(unsigned id, StringRef expression);

  StringRef getExpression() const { return expression; }

  bool shouldBreak(const SimState &state) const override;
  std::string getDescription() const override;

  /// Set custom condition evaluator.
  using ConditionEvaluator = std::function<bool(const SimState &, StringRef)>;
  void setEvaluator(ConditionEvaluator eval) { evaluator = std::move(eval); }

private:
  std::string expression;
  ConditionEvaluator evaluator;
};

/// Breakpoint at a specific simulation time.
class TimeBreakpoint : public Breakpoint {
public:
  TimeBreakpoint(unsigned id, const SimTime &time);

  const SimTime &getTargetTime() const { return targetTime; }

  bool shouldBreak(const SimState &state) const override;
  std::string getDescription() const override;

private:
  SimTime targetTime;
};

/// Breakpoint at a specific cycle.
class CycleBreakpoint : public Breakpoint {
public:
  CycleBreakpoint(unsigned id, uint64_t cycle);

  uint64_t getTargetCycle() const { return targetCycle; }

  bool shouldBreak(const SimState &state) const override;
  std::string getDescription() const override;

private:
  uint64_t targetCycle;
};

//===----------------------------------------------------------------------===//
// Watchpoint
//===----------------------------------------------------------------------===//

/// Watches a signal and records its value history.
class Watchpoint {
public:
  Watchpoint(unsigned id, StringRef signal);

  unsigned getId() const { return id; }
  StringRef getSignal() const { return signal; }

  bool isEnabled() const { return enabled; }
  void setEnabled(bool e) { enabled = e; }

  /// Record current value if changed.
  bool checkAndRecord(const SimState &state);

  /// Get value history.
  struct HistoryEntry {
    SimTime time;
    SignalValue value;
  };
  const std::vector<HistoryEntry> &getHistory() const { return history; }
  void clearHistory() { history.clear(); }

private:
  unsigned id;
  std::string signal;
  bool enabled = true;
  std::vector<HistoryEntry> history;
  std::optional<SignalValue> lastValue;
};

//===----------------------------------------------------------------------===//
// Breakpoint Manager
//===----------------------------------------------------------------------===//

/// Manages all breakpoints and watchpoints for a debug session.
class BreakpointManager {
public:
  BreakpointManager();
  ~BreakpointManager();

  /// Add breakpoints.
  unsigned addLineBreakpoint(StringRef file, unsigned line);
  unsigned addSignalBreakpoint(StringRef signal);
  unsigned addSignalBreakpoint(StringRef signal, const SignalValue &value);
  unsigned addSignalBreakpoint(StringRef signal, SignalBreakpoint::Edge edge);
  unsigned addConditionBreakpoint(StringRef expression);
  unsigned addTimeBreakpoint(const SimTime &time);
  unsigned addCycleBreakpoint(uint64_t cycle);

  /// Add watchpoints.
  unsigned addWatchpoint(StringRef signal);

  /// Remove breakpoints/watchpoints.
  bool removeBreakpoint(unsigned id);
  bool removeWatchpoint(unsigned id);
  void removeAllBreakpoints();
  void removeAllWatchpoints();

  /// Enable/disable breakpoints.
  bool enableBreakpoint(unsigned id, bool enable);
  bool enableWatchpoint(unsigned id, bool enable);

  /// Get breakpoints/watchpoints.
  Breakpoint *getBreakpoint(unsigned id);
  const Breakpoint *getBreakpoint(unsigned id) const;
  Watchpoint *getWatchpoint(unsigned id);
  const Watchpoint *getWatchpoint(unsigned id) const;

  /// List all breakpoints/watchpoints.
  const std::vector<std::unique_ptr<Breakpoint>> &getBreakpoints() const {
    return breakpoints;
  }
  const std::vector<std::unique_ptr<Watchpoint>> &getWatchpoints() const {
    return watchpoints;
  }

  /// Check if any breakpoint should trigger.
  bool shouldBreak(const SimState &state) const;

  /// Get all triggered breakpoints.
  llvm::SmallVector<Breakpoint *> getTriggeredBreakpoints(
      const SimState &state) const;

  /// Update watchpoints with current state.
  void updateWatchpoints(const SimState &state);

  /// Set condition evaluator for condition breakpoints.
  void setConditionEvaluator(ConditionBreakpoint::ConditionEvaluator eval);

private:
  unsigned nextBreakpointId = 1;
  unsigned nextWatchpointId = 1;
  std::vector<std::unique_ptr<Breakpoint>> breakpoints;
  std::vector<std::unique_ptr<Watchpoint>> watchpoints;
  ConditionBreakpoint::ConditionEvaluator conditionEvaluator;
};

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

/// Result of evaluating an expression.
struct EvalResult {
  bool succeeded = false;
  std::optional<SignalValue> value;
  std::string error;

  static EvalResult success(const SignalValue &v) {
    return EvalResult{true, v, ""};
  }
  static EvalResult failure(StringRef err) {
    return EvalResult{false, std::nullopt, err.str()};
  }
};

/// Evaluates expressions in the context of the simulation state.
class ExpressionEvaluator {
public:
  ExpressionEvaluator(const SimState &state);

  /// Evaluate a simple expression (signal name or constant).
  EvalResult evaluate(StringRef expr) const;

  /// Evaluate a comparison expression.
  EvalResult evaluateComparison(StringRef expr) const;

  /// Check if an expression is true (for conditions).
  bool isTrue(StringRef expr) const;

private:
  const SimState &state;

  /// Parse a numeric constant.
  std::optional<SignalValue> parseConstant(StringRef str) const;

  /// Get signal value by name.
  std::optional<SignalValue> getSignalValue(StringRef name) const;
};

} // namespace debug
} // namespace circt

#endif // CIRCT_TOOLS_CIRCT_DEBUG_DEBUG_H
