// RUN: circt-sim-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: trampoline ABI packing for non-64-bit floats must not emit
// invalid bitcasts (e.g. `bitcast float -> i64`).
//
// COMPILE: [circt-sim-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-sim-compile] Generated 1 interpreter trampolines
// COMPILE: [circt-sim-compile] Wrote LLVM IR to

func.func @entry(%x: f32) -> f32 {
  %r = llvm.call @ext_f32(%x) : (f32) -> f32
  return %r : f32
}

llvm.func @ext_f32(f32) -> f32
