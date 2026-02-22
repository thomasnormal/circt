//===- UcontextProcess.cpp - ucontext-based compiled process exec ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of coroutine-style execution for JIT-compiled LLHD processes.
// Uses POSIX ucontext (getcontext/makecontext/setcontext) for initial stack
// setup, then _setjmp/_longjmp for all subsequent context switches. This avoids
// the sigprocmask syscall that swapcontext performs, giving ~10-60x faster
// context switching (~9-20ns vs ~547ns).
//
//===----------------------------------------------------------------------===//

#include "UcontextProcess.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <cstring>

#define DEBUG_TYPE "ucontext-process"

//===----------------------------------------------------------------------===//
// Address Sanitizer fiber annotations
//===----------------------------------------------------------------------===//

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define CIRCT_ASAN_ENABLED 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(CIRCT_ASAN_ENABLED)
#define CIRCT_ASAN_ENABLED 1
#endif

#ifdef CIRCT_ASAN_ENABLED
extern "C" {
void __sanitizer_start_switch_fiber(void **fake_stack_save,
                                    const void *bottom, size_t size);
void __sanitizer_finish_switch_fiber(void *fake_stack_save,
                                     const void **bottom_old,
                                     size_t *size_old);
}
#define ASAN_START_SWITCH(save, bottom, size)                                   \
  __sanitizer_start_switch_fiber((save), (bottom), (size))
#define ASAN_FINISH_SWITCH(save, bottom_old, size_old)                         \
  __sanitizer_finish_switch_fiber((save), (bottom_old), (size_old))
#else
#define ASAN_START_SWITCH(save, bottom, size) ((void)0)
#define ASAN_FINISH_SWITCH(save, bottom_old, size_old) ((void)0)
#endif

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Thread-local manager pointer
//===----------------------------------------------------------------------===//

static thread_local UcontextProcessManager *g_ucontextMgr = nullptr;

void circt::sim::setUcontextManager(UcontextProcessManager *mgr) {
  g_ucontextMgr = mgr;
}

UcontextProcessManager *circt::sim::getUcontextManager() {
  return g_ucontextMgr;
}

//===----------------------------------------------------------------------===//
// Default stack size with env var override
//===----------------------------------------------------------------------===//

size_t circt::sim::getDefaultProcessStackSize() {
  static size_t size = []() -> size_t {
    if (const char *env = std::getenv("CIRCT_SIM_STACK_SIZE")) {
      char *end = nullptr;
      unsigned long val = std::strtoul(env, &end, 0);
      if (end != env && val >= 4096)
        return static_cast<size_t>(val);
    }
    return kProcessStackSize;
  }();
  return size;
}

//===----------------------------------------------------------------------===//
// UcontextProcessManager
//===----------------------------------------------------------------------===//

UcontextProcessState &
UcontextProcessManager::createProcess(ProcessId id, size_t stackSize) {
  assert(!processes.count(id) && "Process already has a coroutine state");

  if (stackSize == 0)
    stackSize = getDefaultProcessStackSize();

  auto state = std::make_unique<UcontextProcessState>();
  state->processId = id;
  state->stackSize = stackSize;

  // Allocate the coroutine stack.
  state->stack = std::make_unique<uint8_t[]>(stackSize);

  auto *ptr = state.get();
  processes[id] = std::move(state);
  return *ptr;
}

UcontextProcessState *UcontextProcessManager::getProcess(ProcessId id) {
  auto it = processes.find(id);
  return it != processes.end() ? it->second.get() : nullptr;
}

bool UcontextProcessManager::hasProcess(ProcessId id) const {
  return processes.count(id);
}

/// Trampoline function for makecontext. ucontext's makecontext only supports
/// int arguments, so we pass the process state pointer as two ints (for
/// portability on 64-bit systems where sizeof(void*) > sizeof(int)).
static void processTrampoline(unsigned int lo, unsigned int hi) {
  uintptr_t addr = static_cast<uintptr_t>(lo) |
                   (static_cast<uintptr_t>(hi) << 32);
  auto *state = reinterpret_cast<UcontextProcessState *>(addr);

  // We just arrived on the process stack via setcontext.
  ASAN_FINISH_SWITCH(nullptr, nullptr, nullptr);

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Starting compiled process "
                          << state->processId << "\n");

  // Call the JIT-compiled entry function.
  // This function will call __circt_sim_yield() at each llhd.wait point,
  // which does _setjmp/_longjmp back to the scheduler. When the scheduler
  // resumes us, we continue from after the __circt_sim_yield() call.
  state->entryFunc();

  // If we reach here, the entry function returned normally (process completed).
  state->completed = true;
  state->lastYieldKind = YieldKind::Halt;

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Process " << state->processId
                          << " completed (entry func returned)\n");

  // Return to scheduler via _longjmp. We must not return from this function
  // because makecontext doesn't set up a valid return address.
  auto *mgr = getUcontextManager();
  assert(mgr && "No ucontext manager set");
  ASAN_START_SWITCH(nullptr, nullptr, 0);
  _longjmp(mgr->getSchedulerJmpBuf(), 1);
  // Does not return.
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclobbered"
void UcontextProcessManager::resumeProcess(ProcessId id) {
  UcontextProcessState *state = getProcess(id);
  assert(state && "No coroutine state for process");
  assert(!state->completed && "Cannot resume completed process");

  activeProcessId = id;

  if (!state->started) {
    // First activation: set up the ucontext for initial stack entry.
    getcontext(&state->processContext);
    state->processContext.uc_stack.ss_sp = state->stack.get();
    state->processContext.uc_stack.ss_size = state->stackSize;
    state->processContext.uc_link = nullptr; // We use _longjmp, not uc_link

    // Pass state pointer as two unsigned ints to makecontext.
    uintptr_t addr = reinterpret_cast<uintptr_t>(state);
    unsigned int lo = static_cast<unsigned int>(addr & 0xFFFFFFFF);
    unsigned int hi = static_cast<unsigned int>(addr >> 32);
    makecontext(&state->processContext, (void (*)())processTrampoline, 2, lo,
                hi);

    state->started = true;

    LLVM_DEBUG(llvm::dbgs() << "[ucontext] First resume of process " << id
                            << ", stack=" << (void *)state->stack.get()
                            << " size=" << state->stackSize << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Resuming process " << id << "\n");

  // Save scheduler context and switch to process.
  if (_setjmp(schedulerJmpBuf) == 0) {
    ASAN_START_SWITCH(nullptr, state->stack.get(), state->stackSize);
    if (state->hasValidJmpBuf) {
      // Subsequent entry: resume from where process last yielded.
      _longjmp(state->processJmpBuf, 1);
    } else {
      // First entry: jump into the process via ucontext.
      setcontext(&state->processContext);
    }
    // Neither call returns.
  }
  // Process yielded or completed — we're back on the scheduler stack.
  ASAN_FINISH_SWITCH(nullptr, nullptr, nullptr);

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Process " << id
                          << " yielded with kind="
                          << static_cast<int>(state->lastYieldKind) << "\n");

  activeProcessId = 0;
}
#pragma GCC diagnostic pop

UcontextProcessState *UcontextProcessManager::getActiveProcess() {
  if (activeProcessId == 0)
    return nullptr;
  return getProcess(activeProcessId);
}

//===----------------------------------------------------------------------===//
// __circt_sim_yield — called by JIT-compiled processes to yield to scheduler
//===----------------------------------------------------------------------===//

extern "C" void __circt_sim_yield(int32_t yieldKind, int64_t data,
                                  const uint32_t *signalIds,
                                  const int32_t *edgeTypes,
                                  int32_t numSignals) {
  auto *mgr = getUcontextManager();
  assert(mgr && "No ucontext manager set when __circt_sim_yield called");

  auto *state = mgr->getActiveProcess();
  assert(state && "No active process when __circt_sim_yield called");

  // Record yield information.
  state->lastYieldKind = static_cast<YieldKind>(yieldKind);
  state->yieldData = data;

  // Copy signal wait list if provided.
  if (signalIds && numSignals > 0) {
    state->waitSignals.resize(numSignals);
    state->waitEdges.resize(numSignals);
    for (int32_t i = 0; i < numSignals; i++) {
      state->waitSignals[i] = static_cast<SignalId>(signalIds[i]);
      state->waitEdges[i] = static_cast<EdgeType>(edgeTypes[i]);
    }
  } else {
    state->waitSignals.clear();
    state->waitEdges.clear();
  }

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Process " << state->processId
                          << " yielding: kind=" << yieldKind
                          << " data=" << data << " signals=" << numSignals
                          << "\n");

  // Save process context and return to scheduler.
  if (_setjmp(state->processJmpBuf) == 0) {
    state->hasValidJmpBuf = true;
    ASAN_START_SWITCH(nullptr, nullptr, 0);
    _longjmp(mgr->getSchedulerJmpBuf(), 1);
    // Does not return.
  }
  // Scheduler resumed us via _longjmp(processJmpBuf, 1).
  ASAN_FINISH_SWITCH(nullptr, nullptr, nullptr);

  LLVM_DEBUG(llvm::dbgs() << "[ucontext] Process " << state->processId
                          << " resumed from yield\n");
}
