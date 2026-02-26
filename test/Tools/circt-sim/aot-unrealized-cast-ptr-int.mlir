// RUN: circt-sim-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE

// Regression: ptr/int unrealized_conversion_cast should not force function
// rejection for AOT compilation.
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] 2 functions + 0 processes ready for codegen

func.func @ptr_roundtrip_i32(%x: i64) -> i32 {
  %p = builtin.unrealized_conversion_cast %x : i64 to !llvm.ptr
  %y = builtin.unrealized_conversion_cast %p : !llvm.ptr to i64
  %eq = arith.cmpi eq, %x, %y : i64
  %ret = arith.extui %eq : i1 to i32
  return %ret : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}
