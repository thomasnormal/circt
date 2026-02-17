//===- JITCompileManager.cpp - circt-sim JIT governance ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITCompileManager.h"

using namespace circt::sim;

JITCompileManager::JITCompileManager() : JITCompileManager(Config{}) {}

JITCompileManager::JITCompileManager(Config config)
    : config(std::move(config)) {}

void JITCompileManager::noteCompile() { ++stats.jitCompilesTotal; }

void JITCompileManager::noteCacheHit() { ++stats.jitCacheHitsTotal; }

void JITCompileManager::noteExecHit() { ++stats.jitExecHitsTotal; }

void JITCompileManager::addCompileWallMs(uint64_t wallMs) {
  stats.jitCompileWallMs += wallMs;
}

void JITCompileManager::addExecWallMs(uint64_t wallMs) {
  stats.jitExecWallMs += wallMs;
}

void JITCompileManager::noteDeopt(DeoptReason reason) {
  ++stats.jitDeoptsTotal;
  switch (reason) {
  case DeoptReason::Unknown:
    ++stats.jitDeoptReasonUnknown;
    break;
  case DeoptReason::InterpreterFallback:
    ++stats.jitDeoptReasonInterpreterFallback;
    break;
  case DeoptReason::GuardFailed:
    ++stats.jitDeoptReasonGuardFailed;
    break;
  case DeoptReason::UnsupportedOperation:
    ++stats.jitDeoptReasonUnsupportedOperation;
    break;
  case DeoptReason::MissingThunk:
    ++stats.jitDeoptReasonMissingThunk;
    break;
  }
}

void JITCompileManager::noteStrictViolation() {
  ++stats.jitStrictViolationsTotal;
}

JITCompileManager::CompileAttemptDecision
JITCompileManager::classifyProcessCompileAttempt(uint64_t processKey) {
  uint64_t activations = ++processActivationCounts[processKey];

  if (config.hotThreshold != 0 && activations < config.hotThreshold)
    return CompileAttemptDecision::BelowHotThreshold;

  if (config.compileBudget == 0)
    return CompileAttemptDecision::CompileBudgetZero;

  if (config.compileBudget > 0 &&
      stats.jitCompilesTotal >= static_cast<uint64_t>(config.compileBudget))
    return CompileAttemptDecision::CompileBudgetExhausted;

  return CompileAttemptDecision::Proceed;
}

bool JITCompileManager::shouldAttemptProcessCompile(uint64_t processKey) {
  return classifyProcessCompileAttempt(processKey) ==
         CompileAttemptDecision::Proceed;
}

bool JITCompileManager::installProcessThunk(uint64_t processKey,
                                            ProcessThunk thunk) {
  if (!thunk)
    return false;
  processThunkCache[processKey] = std::move(thunk);
  return true;
}

bool JITCompileManager::hasProcessThunk(uint64_t processKey) const {
  return processThunkCache.find(processKey) != processThunkCache.end();
}

bool JITCompileManager::executeProcessThunk(uint64_t processKey,
                                            ProcessThunkExecutionState &state) {
  auto it = processThunkCache.find(processKey);
  if (it == processThunkCache.end())
    return false;
  noteCacheHit();
  noteExecHit();
  it->second(state);
  return true;
}

void JITCompileManager::invalidateProcessThunk(uint64_t processKey) {
  processThunkCache.erase(processKey);
}

bool JITCompileManager::noteProcessDeoptOnce(uint64_t processKey,
                                             DeoptReason reason) {
  if (!processKeysWithDeopt.insert(processKey).second)
    return false;
  noteDeopt(reason);
  if (shouldFailOnDeopt())
    noteStrictViolation();
  return true;
}

llvm::StringRef JITCompileManager::getCompileAttemptDecisionName(
    CompileAttemptDecision decision) {
  switch (decision) {
  case CompileAttemptDecision::Proceed:
    return "proceed";
  case CompileAttemptDecision::BelowHotThreshold:
    return "below_hot_threshold";
  case CompileAttemptDecision::CompileBudgetZero:
    return "compile_budget_zero";
  case CompileAttemptDecision::CompileBudgetExhausted:
    return "compile_budget_exhausted";
  }
  return "unknown";
}

llvm::StringRef JITCompileManager::getDeoptReasonName(DeoptReason reason) {
  switch (reason) {
  case DeoptReason::Unknown:
    return "unknown";
  case DeoptReason::InterpreterFallback:
    return "interpreter_fallback";
  case DeoptReason::GuardFailed:
    return "guard_failed";
  case DeoptReason::UnsupportedOperation:
    return "unsupported_operation";
  case DeoptReason::MissingThunk:
    return "missing_thunk";
  }
  return "unknown";
}
