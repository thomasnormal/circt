// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: arith.bitcast should lower to llvm.bitcast so compilable
// functions are not stripped as residual non-LLVM bodies.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE-NOT: Stripped
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen

func.func @bitcast_roundtrip(%x: i32) -> i32 {
  %f = arith.bitcast %x : i32 to f32
  %y = arith.bitcast %f : f32 to i32
  return %y : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}
