// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unresolved direct addressof in llvm.call must propagate through
// wrapper chains (wrapper1 -> wrapper2 -> coverage runtime).
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  llvm.func @wrapper2(!llvm.ptr, i32) -> i64 {
  ^bb0(%p: !llvm.ptr, %i: i32):
    %r = llvm.call @__moore_cross_get_bins_hit(%p, %i) : (!llvm.ptr, i32) -> i64
    llvm.return %r : i64
  }

  llvm.func @wrapper1(!llvm.ptr, i32) -> i64 {
  ^bb0(%p: !llvm.ptr, %i: i32):
    %r = llvm.call @wrapper2(%p, %i) : (!llvm.ptr, i32) -> i64
    llvm.return %r : i64
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %fmt_after = sim.fmt.literal "AFTER"

    llhd.process {
      %fptr = llvm.mlir.addressof @wrapper1 : !llvm.ptr
      %r = llvm.call %fptr(%null, %c0_i32) : !llvm.ptr, (!llvm.ptr, i32) -> i64
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
