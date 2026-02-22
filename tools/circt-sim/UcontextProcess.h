//===- UcontextProcess.h - ucontext-based compiled process exec -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coroutine-style execution of JIT-compiled LLHD processes using ucontext for
// initial stack setup and _setjmp/_longjmp for fast context switching.
// Each compiled process gets its own stack. When it hits llhd.wait (lowered to
// __circt_sim_yield()), it saves its context with _setjmp and does _longjmp
// back to the scheduler. The scheduler resumes it later with _longjmp.
//
// ucontext (getcontext/makecontext/setcontext) is used ONLY for the first
// entry into a process to set up its stack. All subsequent suspend/resume
// cycles use _setjmp/_longjmp which skip sigprocmask (~10-60x faster).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_UCONTEXTPROCESS_H
#define CIRCT_TOOLS_CIRCT_SIM_UCONTEXTPROCESS_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <memory>
#include <setjmp.h>
#include <ucontext.h>
#include <vector>

namespace circt {
namespace sim {

/// Default stack size for compiled process coroutines (32 KB).
/// Override at runtime with CIRCT_SIM_STACK_SIZE environment variable.
constexpr size_t kProcessStackSize = 1 << 15; // 32 KB

/// Get the effective default stack size, respecting CIRCT_SIM_STACK_SIZE.
size_t getDefaultProcessStackSize();

/// Wait kinds passed from compiled code to __circt_sim_yield.
enum class YieldKind : int32_t {
  /// Wait for signal changes (sensitivity list based).
  WaitSignal = 0,
  /// Wait for a specific time delay.
  WaitDelay = 1,
  /// Process has halted (terminated).
  Halt = 2,
};

/// Per-process coroutine state for ucontext-based compiled execution.
struct UcontextProcessState {
  /// The coroutine context (registers + stack pointer).
  /// Used only for initial stack setup (getcontext/makecontext/setcontext).
  ucontext_t processContext;

  /// Fast context-switch buffer for _setjmp/_longjmp (used after first entry).
  jmp_buf processJmpBuf;

  /// Whether processJmpBuf contains a valid saved context.
  bool hasValidJmpBuf = false;

  /// The process's stack memory (owned).
  std::unique_ptr<uint8_t[]> stack;

  /// Stack size in bytes.
  size_t stackSize = kProcessStackSize;

  /// The process ID in the scheduler.
  ProcessId processId = 0;

  /// The JIT-compiled entry function pointer.
  /// No-arg function: signal IDs are baked as constants in the compiled code.
  using EntryFuncTy = void (*)();
  EntryFuncTy entryFunc = nullptr;

  /// Whether the process has been started (entry function called at least once).
  bool started = false;

  /// Whether the process has completed (entry function returned).
  bool completed = false;

  /// Yield state: set by __circt_sim_yield before switching to scheduler.
  YieldKind lastYieldKind = YieldKind::Halt;

  /// Yield data: delay in encoded form (for WaitDelay) or signal count.
  int64_t yieldData = 0;

  /// Wait signal IDs: filled by compiled code before yielding.
  /// For WaitSignal: array of SignalId values to wait on.
  std::vector<SignalId> waitSignals;

  /// Wait edges: edge types for each wait signal.
  std::vector<EdgeType> waitEdges;
};

/// Manager for all ucontext-based compiled processes.
class UcontextProcessManager {
public:
  UcontextProcessManager() = default;
  ~UcontextProcessManager() = default;

  /// Allocate a coroutine state for a process. The process must not already
  /// have a coroutine state. Pass stackSize=0 to use the default (env var).
  UcontextProcessState &createProcess(ProcessId id, size_t stackSize = 0);

  /// Get the coroutine state for a process, or nullptr if not compiled.
  UcontextProcessState *getProcess(ProcessId id);

  /// Check if a process has a coroutine state.
  bool hasProcess(ProcessId id) const;

  /// Resume a compiled process. On first call, enters via setcontext.
  /// On subsequent calls, resumes via _longjmp.
  /// Returns when the process yields (calls __circt_sim_yield) or completes.
  void resumeProcess(ProcessId id);

  /// Get the scheduler context (ucontext, kept for compatibility).
  ucontext_t &getSchedulerContext() { return schedulerContext; }

  /// Get the scheduler's jmp_buf (for _longjmp back from process).
  jmp_buf &getSchedulerJmpBuf() { return schedulerJmpBuf; }

  /// Set the active process (for __circt_sim_yield to find the right state).
  void setActiveProcess(ProcessId id) { activeProcessId = id; }

  /// Get the active process state (called by __circt_sim_yield).
  UcontextProcessState *getActiveProcess();

  /// Get the active process ID.
  ProcessId getActiveProcessId() const { return activeProcessId; }

private:
  /// Scheduler's ucontext (kept for compatibility, not actively used).
  ucontext_t schedulerContext;

  /// Scheduler's jmp_buf for fast context switching via _setjmp/_longjmp.
  jmp_buf schedulerJmpBuf;

  /// Per-process coroutine states.
  llvm::DenseMap<ProcessId, std::unique_ptr<UcontextProcessState>> processes;

  /// Currently executing process ID (for __circt_sim_yield).
  ProcessId activeProcessId = 0;
};

/// Set the thread-local UcontextProcessManager pointer.
/// Must be called before resuming any compiled process.
void setUcontextManager(UcontextProcessManager *mgr);

/// Get the thread-local UcontextProcessManager pointer.
UcontextProcessManager *getUcontextManager();

} // namespace sim
} // namespace circt

//===----------------------------------------------------------------------===//
// Extern "C" runtime functions called by JIT-compiled processes.
//===----------------------------------------------------------------------===//

extern "C" {

/// Yield from a compiled process back to the scheduler.
/// Called when the process hits llhd.wait or llhd.halt.
///
/// @param yieldKind  Type of yield (WaitSignal=0, WaitDelay=1, Halt=2).
/// @param data       Delay value (encoded) for WaitDelay, or 0.
/// @param signalIds  Array of SignalId values for WaitSignal, or nullptr.
/// @param edgeTypes  Array of EdgeType values (parallel to signalIds), or nullptr.
/// @param numSignals Number of entries in signalIds/edgeTypes arrays.
void __circt_sim_yield(int32_t yieldKind, int64_t data,
                       const uint32_t *signalIds, const int32_t *edgeTypes,
                       int32_t numSignals);

} // extern "C"

#endif // CIRCT_TOOLS_CIRCT_SIM_UCONTEXTPROCESS_H
