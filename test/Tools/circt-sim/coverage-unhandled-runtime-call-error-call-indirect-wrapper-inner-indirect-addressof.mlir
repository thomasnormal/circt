// RUN: circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unresolved addressof in func.call_indirect must propagate
// through wrappers that call coverage runtime indirectly via addressof.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  llvm.func @wrapper_indirect(!llvm.ptr, i32) -> i64 {
  ^bb0(%p: !llvm.ptr, %i: i32):
    %fptr = llvm.mlir.addressof @__moore_cross_get_bins_hit : !llvm.ptr
    %r = llvm.call %fptr(%p, %i) : !llvm.ptr, (!llvm.ptr, i32) -> i64
    llvm.return %r : i64
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %fmt_after = sim.fmt.literal "AFTER"

    llhd.process {
      %fptr = llvm.mlir.addressof @wrapper_indirect : !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, i32) -> i64
      %r = func.call_indirect %fn(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
