// RUN: not circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: unresolved direct addressof wrapper in func.call_indirect must
// still fail loudly when the wrapper directly calls a coverage runtime symbol.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  llvm.func @wrapper(!llvm.ptr, i32) -> i64 {
  ^bb0(%p: !llvm.ptr, %i: i32):
    %r = llvm.call @__moore_cross_get_bins_hit(%p, %i) : (!llvm.ptr, i32) -> i64
    llvm.return %r : i64
  }

  hw.module @top() {
    %null = llvm.mlir.zero : !llvm.ptr
    %c0_i32 = llvm.mlir.constant(0 : i32) : i32
    %fmt_after = sim.fmt.literal "AFTER"

    llhd.process {
      %fptr = llvm.mlir.addressof @wrapper : !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr, i32) -> i64
      %r = func.call_indirect %fn(%null, %c0_i32) : (!llvm.ptr, i32) -> i64
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
