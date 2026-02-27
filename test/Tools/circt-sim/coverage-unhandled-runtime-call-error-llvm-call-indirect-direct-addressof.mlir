// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: indirect llvm.call through a direct function pointer to a
// coverage runtime symbol must fail loudly, not silently return X/defaults.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %fmt_after = sim.fmt.literal "AFTER"

    llhd.process {
      %fptr = llvm.mlir.addressof @__moore_cross_get_bins_hit : !llvm.ptr
      %r = llvm.call %fptr(%null, %c0_i32) : !llvm.ptr, (!llvm.ptr, i32) -> i64
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
