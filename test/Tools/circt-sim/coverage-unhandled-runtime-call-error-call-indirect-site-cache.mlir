// RUN: not circt-sim %s --top top 2>&1 | FileCheck %s
//
// Regression: if a cached func.call_indirect site falls back to interpreted
// execution and the callee reports an unhandled coverage runtime call, the
// failure must propagate instead of being silently absorbed as zero results.
//
// CHECK: error: unhandled coverage runtime call in interpreter: __moore_cross_get_bins_hit
// CHECK-NOT: AFTER

module {
  llvm.func @__moore_cross_get_bins_hit(!llvm.ptr, i32) -> i64

  func.func @maybe_cov_fail(%do_fail: i1, %x: i32) -> i32 {
    cf.cond_br %do_fail, ^fail, ^ok
  ^ok:
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    return %r : i32
  ^fail:
    %null = llvm.mlir.zero : !llvm.ptr
    %cov = llvm.call @__moore_cross_get_bins_hit(%null, %x) : (!llvm.ptr, i32) -> i64
    %tr = arith.trunci %cov : i64 to i32
    return %tr : i32
  }

  llvm.mlir.global internal @vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @maybe_cov_fail]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %fmt_after = sim.fmt.literal "AFTER"
    %c0_i32 = arith.constant 0 : i32

    llhd.process {
      cf.br ^loop(%c0_i32 : i32)

    ^loop(%iter: i32):
      %c2 = arith.constant 2 : i32
      %keep = arith.cmpi slt, %iter, %c2 : i32
      cf.cond_br %keep, ^body, ^done

    ^body:
      %vt = llvm.mlir.addressof @vtable : !llvm.ptr
      %c0_i64 = arith.constant 0 : i64
      %slot = llvm.getelementptr %vt[%c0_i64, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fp = llvm.load %slot : !llvm.ptr -> !llvm.ptr
      %fp_cast = builtin.unrealized_conversion_cast %fp : !llvm.ptr to (i1, i32) -> i32
      %c1 = arith.constant 1 : i32
      %is_fail = arith.cmpi eq, %iter, %c1 : i32
      %unused = func.call_indirect %fp_cast(%is_fail, %iter) : (i1, i32) -> i32
      %iter2 = arith.addi %iter, %c1 : i32
      cf.br ^loop(%iter2 : i32)

    ^done:
      sim.proc.print %fmt_after
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
