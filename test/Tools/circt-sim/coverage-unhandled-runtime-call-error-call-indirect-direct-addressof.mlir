// RUN: not circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: direct function-pointer call_indirect to a coverage runtime
// symbol must fail loudly instead of silently degrading to warning+zero.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32

    llhd.process {
      %fptr = llvm.mlir.addressof @__moore_cross_get_bins_hit : !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, i32) -> i64
      %0 = func.call_indirect %fn(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
