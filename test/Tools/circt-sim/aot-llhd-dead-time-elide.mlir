// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: dead LLHD time ops should not force function rejection.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen

func.func @dead_time_const(%x: i32) -> i32 {
  %t = llhd.constant_time <0ns, 0d, 1e>
  %y = arith.addi %x, %x : i32
  return %y : i32
}

func.func @dead_time_current(%x: i32) -> i32 {
  %t = llhd.current_time
  %y = arith.addi %x, %x : i32
  return %y : i32
}
