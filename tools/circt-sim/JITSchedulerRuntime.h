//===- JITSchedulerRuntime.h - JIT ↔ Scheduler bridge -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime bridge between JIT-compiled native code and the ProcessScheduler.
// JIT-compiled hot blocks call extern "C" functions declared here to read/write
// signals through the scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_SIM_JITSCHEDULERRUNTIME_H
#define CIRCT_TOOLS_CIRCT_SIM_JITSCHEDULERRUNTIME_H

#include "circt/Dialect/Sim/ProcessScheduler.h"
#include <algorithm>
#include <cstdint>

namespace circt {
namespace sim {

/// Context passed to JIT-compiled block functions. Set up by the thunk
/// dispatcher before calling the native function and accessed by the
/// __arc_sched_* bridge functions via a thread-local pointer.
struct JITRuntimeContext {
  ProcessScheduler *scheduler = nullptr;
  ProcessId processId = 0;

  /// Per-signal width cache: signalWidths[i] = bit width of signal whose
  /// handle is stored at signalHandles[i]. Used by drive_signal to construct
  /// correctly-sized APInts from raw bytes.
  const uint32_t *signalWidths = nullptr;
  size_t numSignals = 0;
};

/// Install `ctx` as the active JIT runtime context for the current thread.
/// Must be called before invoking any JIT-compiled function.
void setJITRuntimeContext(JITRuntimeContext *ctx);

/// Clear the active JIT runtime context.
void clearJITRuntimeContext();

/// Register the __arc_sched_* symbols with an MLIR ExecutionEngine.
/// Call this after creating the engine and before looking up any JIT symbols.
void registerJITRuntimeSymbols(void *executionEngine);

//===----------------------------------------------------------------------===//
// Delay encoding helpers
//
// JIT-compiled code passes drive delays as a packed i64:
//   bits [63:32]  realTime in femtoseconds
//   bits [31:16]  delta steps
//   bits [15:0]   epsilon steps
// Values larger than field width are saturated to the field max.
//===----------------------------------------------------------------------===//

constexpr uint64_t kJITDelayRealTimeMask = 0xFFFFFFFFULL;
constexpr uint32_t kJITDelayDeltaMask = 0xFFFFU;
constexpr uint32_t kJITDelayEpsilonMask = 0xFFFFU;

inline int64_t encodeJITDelay(uint64_t realTimeFs, uint32_t delta,
                              uint32_t epsilon) {
  uint64_t packedRealTime =
      std::min<uint64_t>(realTimeFs, kJITDelayRealTimeMask);
  uint64_t packedDelta = std::min<uint32_t>(delta, kJITDelayDeltaMask);
  uint64_t packedEpsilon = std::min<uint32_t>(epsilon, kJITDelayEpsilonMask);
  uint64_t bits =
      (packedRealTime << 32) | (packedDelta << 16) | packedEpsilon;
  return static_cast<int64_t>(bits);
}

inline void decodeJITDelay(int64_t encoded, uint64_t &realTimeFs,
                           uint32_t &delta, uint32_t &epsilon) {
  uint64_t bits = static_cast<uint64_t>(encoded);
  realTimeFs = (bits >> 32) & kJITDelayRealTimeMask;
  delta = static_cast<uint32_t>((bits >> 16) & kJITDelayDeltaMask);
  epsilon = static_cast<uint32_t>(bits & kJITDelayEpsilonMask);
}

} // namespace sim
} // namespace circt

//===----------------------------------------------------------------------===//
// Extern "C" runtime functions called by JIT-compiled code.
// These are registered as symbols in the LLVM ORC JIT engine.
//===----------------------------------------------------------------------===//

extern "C" {

/// Read the current simulation time (in femtoseconds).
int64_t __arc_sched_current_time();

/// Read a signal's current value. Returns a pointer to the raw bytes of the
/// signal value (valid until the next signal update).
/// @param signalHandle  SignalId cast to void*.
void *__arc_sched_read_signal(void *signalHandle);

/// Schedule a signal drive.
/// @param signalHandle  SignalId cast to void*.
/// @param valuePtr      Pointer to raw bytes of the new value.
/// @param delayEncoded  Packed delay (see encodeJITDelay).
/// @param enable        Whether the drive is active.
void __arc_sched_drive_signal(void *signalHandle, void *valuePtr,
                              int64_t delayEncoded, int8_t enable);

/// Create a signal (stub — signals are pre-registered by the interpreter).
void *__arc_sched_create_signal(void *initPtr, int64_t sizeBytes);

/// Get the base pointer of the direct signal memory array.
/// JIT-compiled code uses this for zero-overhead narrow signal reads.
/// Returns a pointer to a uint64_t[] indexed by SignalId.
uint64_t *__circt_sim_signal_memory_base();

/// Drive a narrow signal (<=64 bits) using a uint64_t value directly.
/// Avoids APInt construction overhead of __arc_sched_drive_signal.
/// @param sigId  Signal ID (not cast to pointer).
/// @param value  Signal value zero-extended to uint64_t.
/// @param delayEncoded  Packed delay (see encodeJITDelay).
/// @param enable  Whether the drive is active.
void __arc_sched_drive_signal_fast(uint32_t sigId, uint64_t value,
                                    int64_t delayEncoded, int8_t enable);

/// Print the formatted string built by sim.fmt.* runtime helpers.
/// @param fstringPtr  Pointer to a std::string built by fmt helpers.
void __circt_sim_proc_print(void *fstringPtr);

/// Create a format string from a literal.
/// @param data  Pointer to the string data (not null-terminated).
/// @param len   Length of the string in bytes.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_literal(const char *data, int64_t len);

/// Format an integer as decimal.
/// @param val       Value (sign-extended to 64 bits if signed).
/// @param width     Bit width of the original value (for unsigned masking).
/// @param isSigned  Whether to format as signed decimal.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_dec(int64_t val, int32_t width, int8_t isSigned);

/// Format an integer as hexadecimal (lowercase).
/// @param val    Value (zero-extended to 64 bits).
/// @param width  Bit width of the original value.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_hex(int64_t val, int32_t width);

/// Format an integer as binary.
/// @param val    Value (zero-extended to 64 bits).
/// @param width  Bit width of the original value.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_bin(int64_t val, int32_t width);

/// Format a character (lowest 8 bits of val).
/// @param val  Character value.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_char(int64_t val);

/// Concatenate multiple format strings.
/// @param parts  Array of pointers to std::strings on the format arena.
/// @param count  Number of parts.
/// @return Opaque pointer to a std::string on the format arena.
void *__circt_sim_fmt_concat(void **parts, int32_t count);

/// Reset the format string arena (called after sim.proc.print).
void __circt_sim_fmt_arena_reset();

} // extern "C"

#endif // CIRCT_TOOLS_CIRCT_SIM_JITSCHEDULERRUNTIME_H
