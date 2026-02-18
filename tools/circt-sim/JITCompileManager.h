//===- JITCompileManager.h - circt-sim JIT governance ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a lightweight compile-mode governance manager used by
// circt-sim's native JIT rollout scaffolding.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_JITCOMPILEMANAGER_H
#define CIRCT_TOOLS_CIRCT_SIM_JITCOMPILEMANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <cstdint>
#include <string>

namespace circt::sim {

struct ProcessThunkExecutionState {
  bool deoptRequested = false;
  // Some thunks may execute side effects before deciding to deopt. In that
  // case, interpreter fallback should continue from the mutated process state
  // instead of restoring the pre-thunk snapshot.
  bool restoreSnapshotOnDeopt = true;
  bool halted = false;
  bool waiting = false;
  uint64_t resumeToken = 0;
  std::string deoptDetail;
};

class JITCompileManager {
public:
  enum class CompileAttemptDecision : uint8_t {
    Proceed = 0,
    BelowHotThreshold = 1,
    CompileBudgetZero = 2,
    CompileBudgetExhausted = 3,
  };

  enum class DeoptReason : uint8_t {
    Unknown = 0,
    InterpreterFallback = 1,
    GuardFailed = 2,
    UnsupportedOperation = 3,
    MissingThunk = 4,
  };

  struct Config {
    uint64_t hotThreshold = 0;
    int64_t compileBudget = 0;
    std::string cachePolicy = "memory";
    bool failOnDeopt = false;
  };

  struct Statistics {
    uint64_t jitCompilesTotal = 0;
    uint64_t jitCacheHitsTotal = 0;
    uint64_t jitExecHitsTotal = 0;
    uint64_t jitDeoptsTotal = 0;
    uint64_t jitCompileWallMs = 0;
    uint64_t jitExecWallMs = 0;
    uint64_t jitStrictViolationsTotal = 0;

    uint64_t jitDeoptReasonUnknown = 0;
    uint64_t jitDeoptReasonInterpreterFallback = 0;
    uint64_t jitDeoptReasonGuardFailed = 0;
    uint64_t jitDeoptReasonUnsupportedOperation = 0;
    uint64_t jitDeoptReasonMissingThunk = 0;
  };

  using ProcessThunk = std::function<void(ProcessThunkExecutionState &)>;

  JITCompileManager();
  explicit JITCompileManager(Config config);

  const Config &getConfig() const { return config; }
  const Statistics &getStatistics() const { return stats; }

  bool shouldFailOnDeopt() const { return config.failOnDeopt; }
  bool hasAnyDeopt() const { return stats.jitDeoptsTotal != 0; }

  void noteCompile();
  void noteCacheHit();
  void noteExecHit();
  void addCompileWallMs(uint64_t wallMs);
  void addExecWallMs(uint64_t wallMs);
  void noteDeopt(DeoptReason reason);
  void noteStrictViolation();

  /// Record a process activation and decide whether a compile should be tried.
  CompileAttemptDecision classifyProcessCompileAttempt(uint64_t processKey);
  bool shouldAttemptProcessCompile(uint64_t processKey);

  bool installProcessThunk(uint64_t processKey, ProcessThunk thunk);
  bool hasProcessThunk(uint64_t processKey) const;
  bool executeProcessThunk(uint64_t processKey,
                           ProcessThunkExecutionState &state);
  void invalidateProcessThunk(uint64_t processKey);

  bool noteProcessDeoptOnce(uint64_t processKey, DeoptReason reason);

  static llvm::StringRef
  getCompileAttemptDecisionName(CompileAttemptDecision decision);
  static llvm::StringRef getDeoptReasonName(DeoptReason reason);

private:
  Config config;
  Statistics stats;
  llvm::DenseMap<uint64_t, uint64_t> processActivationCounts;
  llvm::DenseMap<uint64_t, ProcessThunk> processThunkCache;
  llvm::DenseSet<uint64_t> processKeysWithDeopt;
};

} // namespace circt::sim

#endif // CIRCT_TOOLS_CIRCT_SIM_JITCOMPILEMANAGER_H
