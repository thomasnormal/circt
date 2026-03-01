// RUN: not circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unhandled coverage runtime calls via func.call in interpreted
// mode must error explicitly instead of silently returning X/default values.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit

module {
  func.func private @__moore_cross_get_bins_hit(%arg0: !llvm.ptr, %arg1: i32) -> i64

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    llhd.process {
      %0 = func.call @__moore_cross_get_bins_hit(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
