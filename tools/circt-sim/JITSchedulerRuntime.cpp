//===- JITSchedulerRuntime.cpp - JIT ↔ Scheduler bridge -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "JITSchedulerRuntime.h"
#include "UcontextProcess.h"
#include "circt/Dialect/Sim/EventQueue.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define DEBUG_TYPE "jit-scheduler-runtime"

using namespace circt::sim;

//===----------------------------------------------------------------------===//
// Thread-local JIT context
//===----------------------------------------------------------------------===//

static thread_local JITRuntimeContext *g_jitCtx = nullptr;

void circt::sim::setJITRuntimeContext(JITRuntimeContext *ctx) {
  g_jitCtx = ctx;
}

void circt::sim::clearJITRuntimeContext() { g_jitCtx = nullptr; }

//===----------------------------------------------------------------------===//
// Runtime bridge functions
//===----------------------------------------------------------------------===//

extern "C" {

int64_t __arc_sched_current_time() {
  assert(g_jitCtx && "JIT runtime context not set");
  return static_cast<int64_t>(
      g_jitCtx->scheduler->getCurrentTime().realTime);
}

void *__arc_sched_read_signal(void *signalHandle) {
  assert(g_jitCtx && "JIT runtime context not set");
  SignalId sigId = static_cast<SignalId>(reinterpret_cast<uintptr_t>(signalHandle));
  const SignalValue &val = g_jitCtx->scheduler->getSignalValue(sigId);

  // Return pointer to the raw APInt storage. On little-endian x86,
  // LLVM load instructions can read the correct number of bytes directly.
  // The pointer is valid until the next updateSignal() call on this signal.
  return const_cast<void *>(
      static_cast<const void *>(val.getAPInt().getRawData()));
}

void __arc_sched_drive_signal(void *signalHandle, void *valuePtr,
                              int64_t delayEncoded, int8_t enable) {
  if (!enable)
    return;
  assert(g_jitCtx && "JIT runtime context not set");

  SignalId sigId = static_cast<SignalId>(reinterpret_cast<uintptr_t>(signalHandle));

  // Determine signal width from the scheduler's signal state.
  const SignalValue &currentVal = g_jitCtx->scheduler->getSignalValue(sigId);
  uint32_t width = currentVal.getWidth();
  unsigned byteWidth = (width + 7) / 8;

  // Construct the new value from the raw bytes written by JIT code.
  llvm::APInt newBits(width, 0);
  std::memcpy(const_cast<uint64_t *>(newBits.getRawData()), valuePtr, byteWidth);
  // Raw byte copies can leave APInt unused high bits dirty. Mask the last
  // storage word so APInt single-word accessors (e.g. getZExtValue) stay valid.
  unsigned usedBitsInLastWord = width % 64;
  if (usedBitsInLastWord != 0) {
    uint64_t mask = (uint64_t{1} << usedBitsInLastWord) - 1;
    uint64_t *rawWords = const_cast<uint64_t *>(newBits.getRawData());
    rawWords[newBits.getNumWords() - 1] &= mask;
  }
  SignalValue newVal(newBits);

  // Decode the packed delay.
  uint64_t realTimeFs;
  uint32_t delta, epsilon;
  circt::sim::decodeJITDelay(delayEncoded, realTimeFs, delta, epsilon);

  LLVM_DEBUG(llvm::dbgs() << "[JIT-DRV] sig=" << sigId << " width=" << width
                          << " realFs=" << realTimeFs << " delta=" << delta
                          << " eps=" << epsilon << "\n");

  uint64_t combinedDelta = static_cast<uint64_t>(delta) + epsilon;
  if (realTimeFs == 0 && combinedDelta == 0) {
    // Immediate drive (delta-cycle, same time slot).
    g_jitCtx->scheduler->updateSignal(sigId, newVal);
  } else {
    // Scheduled drive via the event scheduler.
    SimTime currentTime = g_jitCtx->scheduler->getCurrentTime();
    SimTime targetTime = currentTime;
    if (realTimeFs > 0) {
      targetTime = currentTime.advanceTime(realTimeFs);
    }

    if (combinedDelta > 0) {
      if (realTimeFs == 0) {
        uint64_t scheduledDelta =
            static_cast<uint64_t>(currentTime.deltaStep) + combinedDelta;
        if (scheduledDelta > std::numeric_limits<uint32_t>::max())
          scheduledDelta = std::numeric_limits<uint32_t>::max();
        targetTime.deltaStep = static_cast<uint32_t>(scheduledDelta);
      } else {
        // Match interpreter behavior for mixed real-time+delta delays.
        targetTime.deltaStep = currentTime.deltaStep;
      }
    }
    // Keep all deferred drives in NBA to align with interpretDrive().
    targetTime.region = static_cast<uint8_t>(SchedulingRegion::NBA);

    auto *scheduler = g_jitCtx->scheduler;
    scheduler->getEventScheduler().schedule(
        targetTime, SchedulingRegion::NBA,
        Event([scheduler, sigId, newVal]() {
          scheduler->updateSignal(sigId, newVal);
        }));
  }
}

void *__arc_sched_create_signal(void * /*initPtr*/, int64_t /*sizeBytes*/) {
  // Signals are pre-registered by the interpreter. JIT-compiled hot blocks
  // never create new signals.
  return nullptr;
}

uint64_t *__circt_sim_signal_memory_base() {
  assert(g_jitCtx && "JIT runtime context not set");
  return g_jitCtx->scheduler->getSignalMemoryBase();
}

void __arc_sched_drive_signal_fast(uint32_t sigId, uint64_t value,
                                    int64_t delayEncoded, int8_t enable) {
  if (!enable)
    return;
  assert(g_jitCtx && "JIT runtime context not set");

  auto *scheduler = g_jitCtx->scheduler;
  uint32_t width = scheduler->getSignalValue(sigId).getWidth();

  // Decode the packed delay.
  uint64_t realTimeFs;
  uint32_t delta, epsilon;
  circt::sim::decodeJITDelay(delayEncoded, realTimeFs, delta, epsilon);

  LLVM_DEBUG(llvm::dbgs() << "[JIT-DRV-FAST] sig=" << sigId << " width="
                          << width << " value=" << value << " realFs="
                          << realTimeFs << " delta=" << delta << " eps="
                          << epsilon << "\n");

  uint64_t combinedDelta = static_cast<uint64_t>(delta) + epsilon;
  if (realTimeFs == 0 && combinedDelta == 0) {
    // Immediate drive (delta-cycle, same time slot).
    scheduler->updateSignalFast(sigId, value, width);
  } else {
    // Scheduled drive via the event scheduler.
    SimTime currentTime = scheduler->getCurrentTime();
    SimTime targetTime = currentTime;
    if (realTimeFs > 0) {
      targetTime = currentTime.advanceTime(realTimeFs);
    }

    if (combinedDelta > 0) {
      if (realTimeFs == 0) {
        uint64_t scheduledDelta =
            static_cast<uint64_t>(currentTime.deltaStep) + combinedDelta;
        if (scheduledDelta > std::numeric_limits<uint32_t>::max())
          scheduledDelta = std::numeric_limits<uint32_t>::max();
        targetTime.deltaStep = static_cast<uint32_t>(scheduledDelta);
      } else {
        targetTime.deltaStep = currentTime.deltaStep;
      }
    }
    // Keep all deferred drives in NBA to align with interpretDrive().
    targetTime.region = static_cast<uint8_t>(SchedulingRegion::NBA);

    SignalId sid = static_cast<SignalId>(sigId);
    scheduler->getEventScheduler().schedule(
        targetTime, SchedulingRegion::NBA,
        Event([scheduler, sid, value, width]() {
          scheduler->updateSignalFast(sid, value, width);
        }));
  }
}

/// Thread-local arena for format string temporaries. Strings are allocated
/// here by fmt helpers and freed in bulk after each sim.proc.print.
static thread_local std::vector<std::string> g_fmtArena;

static std::string *arenaAlloc() {
  g_fmtArena.emplace_back();
  return &g_fmtArena.back();
}

void __circt_sim_proc_print(void *fstringPtr) {
  if (fstringPtr) {
    auto *str = static_cast<std::string *>(fstringPtr);
    llvm::outs() << *str;
  }
  g_fmtArena.clear();
}

void *__circt_sim_fmt_literal(const char *data, int64_t len) {
  auto *s = arenaAlloc();
  if (data && len > 0)
    s->assign(data, static_cast<size_t>(len));
  return s;
}

void *__circt_sim_fmt_dec(int64_t val, int32_t width, int8_t isSigned) {
  auto *s = arenaAlloc();
  if (isSigned) {
    // Sign-extend from width bits to 64 bits, then print as signed.
    if (width < 64) {
      int64_t shift = 64 - width;
      val = (val << shift) >> shift;
    }
    *s = std::to_string(val);
  } else {
    // Mask to width bits and print as unsigned.
    uint64_t uval = static_cast<uint64_t>(val);
    if (width < 64)
      uval &= (1ULL << width) - 1;
    *s = std::to_string(uval);
  }
  return s;
}

void *__circt_sim_fmt_hex(int64_t val, int32_t width) {
  auto *s = arenaAlloc();
  uint64_t uval = static_cast<uint64_t>(val);
  if (width < 64)
    uval &= (1ULL << width) - 1;
  llvm::raw_string_ostream os(*s);
  os << llvm::format_hex_no_prefix(uval, 0);
  return s;
}

void *__circt_sim_fmt_bin(int64_t val, int32_t width) {
  auto *s = arenaAlloc();
  uint64_t uval = static_cast<uint64_t>(val);
  if (width < 64)
    uval &= (1ULL << width) - 1;
  if (width == 0) {
    *s = "0";
  } else {
    s->reserve(width);
    for (int32_t i = width - 1; i >= 0; --i)
      s->push_back((uval >> i) & 1 ? '1' : '0');
  }
  return s;
}

void *__circt_sim_fmt_char(int64_t val) {
  auto *s = arenaAlloc();
  s->push_back(static_cast<char>(val & 0xFF));
  return s;
}

void *__circt_sim_fmt_concat(void **parts, int32_t count) {
  auto *s = arenaAlloc();
  size_t totalLen = 0;
  for (int32_t i = 0; i < count; ++i) {
    if (parts[i])
      totalLen += static_cast<std::string *>(parts[i])->size();
  }
  s->reserve(totalLen);
  for (int32_t i = 0; i < count; ++i) {
    if (parts[i])
      s->append(*static_cast<std::string *>(parts[i]));
  }
  return s;
}

void __circt_sim_fmt_arena_reset() { g_fmtArena.clear(); }

} // extern "C"

//===----------------------------------------------------------------------===//
// Symbol registration for MLIR ExecutionEngine
//===----------------------------------------------------------------------===//

#ifdef CIRCT_SIM_JIT_ENABLED
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"

void circt::sim::registerJITRuntimeSymbols(void *enginePtr) {
  auto &engine = *static_cast<mlir::ExecutionEngine *>(enginePtr);
  engine.registerSymbols([](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbolMap;

    auto bind = [&](llvm::StringRef name, auto *funcPtr) {
      symbolMap[interner(name)] = {
          llvm::orc::ExecutorAddr::fromPtr(funcPtr),
          llvm::JITSymbolFlags::Exported};
    };

    bind("__arc_sched_current_time", &__arc_sched_current_time);
    bind("__arc_sched_read_signal", &__arc_sched_read_signal);
    bind("__arc_sched_drive_signal", &__arc_sched_drive_signal);
    bind("__arc_sched_create_signal", &__arc_sched_create_signal);
    bind("__circt_sim_signal_memory_base", &__circt_sim_signal_memory_base);
    bind("__arc_sched_drive_signal_fast", &__arc_sched_drive_signal_fast);
    bind("__circt_sim_yield", &__circt_sim_yield);
    bind("__circt_sim_proc_print", &__circt_sim_proc_print);
    bind("__circt_sim_fmt_literal", &__circt_sim_fmt_literal);
    bind("__circt_sim_fmt_dec", &__circt_sim_fmt_dec);
    bind("__circt_sim_fmt_hex", &__circt_sim_fmt_hex);
    bind("__circt_sim_fmt_bin", &__circt_sim_fmt_bin);
    bind("__circt_sim_fmt_char", &__circt_sim_fmt_char);
    bind("__circt_sim_fmt_concat", &__circt_sim_fmt_concat);
    bind("__circt_sim_fmt_arena_reset", &__circt_sim_fmt_arena_reset);
    return symbolMap;
  });
}
#else
void circt::sim::registerJITRuntimeSymbols(void * /*enginePtr*/) {
  llvm::report_fatal_error(
      "JIT not enabled — rebuild with CIRCT_SIM_JIT_ENABLED");
}
#endif
