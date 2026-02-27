// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: when an inner llvm.call reports an unhandled coverage runtime
// call, enclosing func.call frames must not absorb that failure and continue.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: failed internally (absorbing)

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  func.func @call_cov(%p: !llvm.ptr, %i: i32) -> i64 {
    %r = llvm.call @__moore_cross_get_bins_hit(%p, %i) : (!llvm.ptr, i32) -> i64
    return %r : i64
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32

    llhd.process {
      %0 = func.call @call_cov(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
