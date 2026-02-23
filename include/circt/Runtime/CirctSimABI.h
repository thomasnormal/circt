//===- CirctSimABI.h - Stable C ABI for AOT-compiled simulation --*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stable C ABI contract between circt-sim (the runtime) and AOT-compiled
// simulation shared objects (.so). This header is used by both sides:
//
//   Compiler side (circt-sim-compile):
//     - Emits code that calls __circt_sim_* runtime functions declared here.
//     - Exports a CirctSimCompiledModule descriptor via
//       circt_sim_get_compiled_module().
//
//   Runtime side (circt-sim):
//     - Provides implementations of __circt_sim_* functions.
//     - Loads the .so via dlopen and reads the descriptor to register
//       compiled processes and functions.
//
// ABI stability rules:
//   - All functions use C linkage (no name mangling).
//   - All arguments are scalar or pointer (no aggregates at the boundary).
//   - New functions may be added; existing signatures must not change.
//   - CIRCT_SIM_ABI_VERSION is bumped on any breaking change.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_RUNTIME_CIRCT_SIM_ABI_H
#define CIRCT_RUNTIME_CIRCT_SIM_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ABI Version
//===----------------------------------------------------------------------===//

/// Bump this on any breaking change to the ABI (struct layout, function
/// signature changes, removed functions). Adding new functions is NOT a
/// breaking change.
#define CIRCT_SIM_ABI_VERSION 1

//===----------------------------------------------------------------------===//
// Opaque Types
//===----------------------------------------------------------------------===//

/// Opaque simulation context handle. Passed to all runtime callbacks so the
/// compiled code can access signals, scheduling, and memory allocation.
typedef struct CirctSimCtx CirctSimCtx;

//===----------------------------------------------------------------------===//
// Process Kinds
//===----------------------------------------------------------------------===//

/// Classification of compiled processes. Determines how the runtime dispatches
/// each process.
typedef enum {
  /// Run-to-completion callback (no yield). Covers CallbackStaticObserved,
  /// CallbackDynamicWait, and OneShotCallback from the ExecModel enum.
  /// The runtime calls the function; it returns when done.
  CIRCT_PROC_CALLBACK = 0,

  /// Coroutine that can yield mid-body (wait_event, fork/join). Requires a
  /// private stack (setjmp/longjmp or ucontext). Used for UVM sequences,
  /// fork/join blocks, and complex timed control.
  CIRCT_PROC_COROUTINE = 1,

  /// Lightweight time-only callback. Like CALLBACK but scheduled via the
  /// minnow fast-path (24-byte MinnowInfo) instead of the full TimeWheel.
  /// Typical use: clock generators (`always #5 clk = ~clk`).
  CIRCT_PROC_MINNOW = 2,
} CirctProcKind;

//===----------------------------------------------------------------------===//
// Compiled Module Descriptor
//===----------------------------------------------------------------------===//

/// Descriptor exported from a compiled .so file. Contains metadata about all
/// compiled processes and functions, allowing the runtime to register them
/// without parsing the original MLIR.
///
/// Invariants:
///   - abi_version must equal CIRCT_SIM_ABI_VERSION or the runtime rejects it.
///   - proc_names, proc_kind, proc_entry are parallel arrays of num_procs.
///   - func_names, func_entry are parallel arrays of num_funcs.
///   - All string pointers must remain valid for the lifetime of the .so.
typedef struct {
  /// Must equal CIRCT_SIM_ABI_VERSION. The runtime checks this on load and
  /// rejects mismatched .so files.
  uint32_t abi_version;

  /// Number of compiled processes.
  uint32_t num_procs;

  /// Process symbol names for matching against the MLIR module's process ops.
  /// Array of num_procs NUL-terminated strings.
  const char *const *proc_names;

  /// CirctProcKind for each process (cast from uint8_t).
  const uint8_t *proc_kind;

  /// Entry function pointers, one per process. Signature depends on proc_kind:
  ///   CALLBACK/MINNOW: void (*)(CirctSimCtx*, void* frame)
  ///   COROUTINE:       void (*)(CirctSimCtx*)
  const void *const *proc_entry;

  /// Number of compiled func.func bodies (non-process functions).
  uint32_t num_funcs;

  /// Function symbol names (e.g., "uvm_pkg::uvm_object::get_name").
  /// Array of num_funcs NUL-terminated strings.
  const char *const *func_names;

  /// Native function pointers, one per function. Signature matches the
  /// original func.func with LLVM-native ABI (scalars and pointers only).
  const void *const *func_entry;
} CirctSimCompiledModule;

//===----------------------------------------------------------------------===//
// .so Entrypoints
//===----------------------------------------------------------------------===//
//
// These functions are exported from the compiled .so with default visibility.
// The runtime loads them via dlsym after dlopen.

/// Returns the compiled module descriptor. The returned pointer must remain
/// valid for the lifetime of the loaded .so.
__attribute__((visibility("default")))
const CirctSimCompiledModule *circt_sim_get_compiled_module(void);

/// Returns a build ID string for cache invalidation. Encodes:
///   - CIRCT_SIM_ABI_VERSION
///   - Target triple + CPU features
///   - MLIR IR content hash
/// The runtime compares this against the current configuration to detect
/// stale cached .so files.
__attribute__((visibility("default")))
const char *circt_sim_get_build_id(void);

//===----------------------------------------------------------------------===//
// Runtime API — Signal Access
//===----------------------------------------------------------------------===//
//
// Functions provided by the runtime (circt-sim) for compiled code to read
// and drive simulation signals.

/// Read up to 64 bits of a signal's 2-state value.
/// For signals wider than 64 bits, use __circt_sim_signal_read_ptr instead.
uint64_t __circt_sim_signal_read_u64(CirctSimCtx *ctx, uint32_t sig_id);

/// Drive up to 64 bits onto a signal.
///   delay_kind: 0 = transport, 1 = inertial, 2 = NBA (non-blocking assign)
///   delay: delay in simulation time units (0 for combinational)
void __circt_sim_signal_drive_u64(CirctSimCtx *ctx, uint32_t sig_id,
                                  uint64_t val, uint8_t delay_kind,
                                  uint64_t delay);

/// Get a pointer to a signal's raw memory (for wide or 4-state signals).
/// The returned pointer is valid until the next delta step.
void *__circt_sim_signal_read_ptr(CirctSimCtx *ctx, uint32_t sig_id);

/// Drive raw bytes onto a signal (for wide or 4-state signals).
///   data: pointer to the value bytes
///   num_bytes: number of bytes to drive
void __circt_sim_signal_drive_ptr(CirctSimCtx *ctx, uint32_t sig_id,
                                  const void *data, uint32_t num_bytes,
                                  uint8_t delay_kind, uint64_t delay);

//===----------------------------------------------------------------------===//
// Runtime API — Coroutine Yields
//===----------------------------------------------------------------------===//
//
// These functions suspend the calling coroutine. They must only be called
// from processes with kind CIRCT_PROC_COROUTINE.

/// Suspend until the specified event fires.
void __circt_sim_wait_event(CirctSimCtx *ctx, uint32_t event_id);

/// Suspend for the specified number of simulation time units.
void __circt_sim_wait_time(CirctSimCtx *ctx, uint64_t delay);

//===----------------------------------------------------------------------===//
// Runtime API — Fork / Join
//===----------------------------------------------------------------------===//
//
// Coroutine-only. Implements SystemVerilog fork/join semantics.

/// Fork num_children child processes. Returns a fork ID for join operations.
uint32_t __circt_sim_fork(CirctSimCtx *ctx, uint32_t num_children);

/// Block until any one child of the fork completes.
void __circt_sim_join_any(CirctSimCtx *ctx, uint32_t fork_id);

/// Block until all children of the fork complete.
void __circt_sim_join_all(CirctSimCtx *ctx, uint32_t fork_id);

/// Kill all active children in the current fork scope.
void __circt_sim_disable_fork(CirctSimCtx *ctx);

//===----------------------------------------------------------------------===//
// Runtime API — Interpreter Fallback
//===----------------------------------------------------------------------===//

/// Call a function that was not compiled (falls back to the MLIR interpreter).
/// Arguments and return values are packed as arrays of 8-byte slots:
///   - Integer args: zero-extended to uint64_t
///   - Pointer args: stored as void* in a uint64_t slot
///
///   func_id: index into the runtime's function table
///   args: array of numArgs packed argument slots
///   num_args: number of arguments
///   rets: array of num_rets slots to receive return values
///   num_rets: number of return values (0 or 1)
void __circt_sim_call_interpreted(CirctSimCtx *ctx, uint32_t func_id,
                                  const uint64_t *args, uint32_t num_args,
                                  uint64_t *rets, uint32_t num_rets);

//===----------------------------------------------------------------------===//
// Runtime API — Memory Allocation
//===----------------------------------------------------------------------===//

/// Allocate memory managed by the simulation context. The returned memory
/// is valid for the lifetime of the simulation and is freed on shutdown.
/// Used by compiled code for dynamic allocations (new objects, strings, etc.).
void *__circt_sim_alloc(CirctSimCtx *ctx, uint64_t size);

//===----------------------------------------------------------------------===//
// Runtime API — Display / Control
//===----------------------------------------------------------------------===//

/// Formatted output ($display, $write, etc.). Uses printf-style format string.
void __circt_sim_display(CirctSimCtx *ctx, const char *fmt, ...);

/// Terminate simulation ($finish). The runtime handles graceful shutdown
/// including deferred fork cleanup and destructor avoidance via _exit(0).
void __circt_sim_finish(CirctSimCtx *ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRCT_RUNTIME_CIRCT_SIM_ABI_H
