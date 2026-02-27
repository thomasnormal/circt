// RUN: circt-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: FileCheck %s --check-prefix=LLVM < %t.ll

// Regression: trampoline ABI packing for non-64-bit floats must not emit
// invalid bitcasts (e.g. `bitcast float -> i64`).
//
// COMPILE: [circt-compile] Functions: 1 total, 0 external, 0 rejected, 1 compilable
// COMPILE: [circt-compile] Wrote LLVM IR to
//
// LLVM: define internal float @entry(float
// LLVM: tail call float @ext_f32(float
// LLVM-NOT: bitcast float

func.func @entry(%x: f32) -> f32 {
  %r = llvm.call @ext_f32(%x) : (f32) -> f32
  return %r : f32
}

llvm.func @ext_f32(f32) -> f32
