// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unhandled coverage runtime calls in interpreted mode must error
// explicitly instead of silently falling back as generic external calls.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llhd.process {
      %0 = llvm.call @__moore_cross_get_bins_hit(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
