// RUN: circt-compile --emit-llvm %s -o %t.ll 2>&1 | FileCheck %s --check-prefix=COMPILE
// XFAIL: *

// Regression: LowerTaggedIndirectCalls must not assign names to void call
// results when lowering indirect calls.
//
// COMPILE: [circt-compile] Functions: 3 total, 0 external, 0 rejected, 3 compilable
// COMPILE-DAG: [circt-compile] LowerTaggedIndirectCalls: lowered 1 indirect calls
// COMPILE-DAG: [circt-compile] 3 functions + 0 processes ready for codegen

func.func @callee_void(%x: i32) {
  return
}

func.func @dispatch_void(%fp: !llvm.ptr, %x: i32) {
  %f = builtin.unrealized_conversion_cast %fp : !llvm.ptr to (i32) -> ()
  func.call_indirect %f(%x) : (i32) -> ()
  return
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}
