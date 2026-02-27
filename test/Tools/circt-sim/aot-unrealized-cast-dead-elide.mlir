// RUN: circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: a dead unrealized_conversion_cast should not make an otherwise
// compilable function get rejected.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] 2 functions + 0 processes ready for codegen

func.func @dead_cast_only(%x: i32) -> i32 {
  %one = arith.constant 1 : i64
  %p = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
  %dead = builtin.unrealized_conversion_cast %p : !llvm.ptr to !llhd.ref<i32>
  %y = arith.addi %x, %x : i32
  return %y : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}
