// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s

// Verify that circt-sim-compile correctly reports compilation statistics.
// Has one external declaration (counted as external) and one compilable body.

// CHECK: [circt-sim-compile] Functions: 2 total, 1 external, 0 rejected, 1 compilable
// CHECK: [circt-sim-compile] 1 functions + 0 processes ready for codegen

// External declaration — counted but not compiled.
func.func private @external_fn(i32) -> i32

// Compilable function: body uses only arith ops.
func.func @add_i32(%a: i32, %b: i32) -> i32 {
  %c = arith.addi %a, %b : i32
  return %c : i32
}
