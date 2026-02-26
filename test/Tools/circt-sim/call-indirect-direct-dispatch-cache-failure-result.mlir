// RUN: circt-sim %s 2>&1 | FileCheck %s

// Regression: when a cached func.call_indirect site falls back to interpreted
// execution and the callee fails, the result must be overwritten with zero
// (same behavior as the non-cached path). Prior bug: cached failure left the
// previous iteration's value in the value map, causing stale result reuse.
//
// CHECK: sum=1
// CHECK-NOT: sum=2

module {
  func.func @maybe_fail(%do_fail: i1, %x: i32) -> i32 {
    cf.cond_br %do_fail, ^fail, ^ok
  ^ok:
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    return %r : i32
  ^fail:
    // Force an interpreted function-body failure via drive-to-unknown-ref.
    %zero64 = arith.constant 0 : i64
    %bogus = builtin.unrealized_conversion_cast %zero64 : i64 to !llhd.ref<i32>
    %eps = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %bogus, %x after %eps : i32
    %z = arith.constant 0 : i32
    return %z : i32
  }

  llvm.mlir.global internal @vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @maybe_fail]]
  } : !llvm.array<1 x ptr>

  hw.module @top() {
    %fmt_prefix = sim.fmt.literal "sum="
    %fmt_nl = sim.fmt.literal "\0A"
    %c0_i32 = arith.constant 0 : i32

    llhd.process {
      cf.br ^loop(%c0_i32, %c0_i32 : i32, i32)

    ^loop(%iter: i32, %sum: i32):
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
      %r = func.call_indirect %fp_cast(%is_fail, %iter) : (i1, i32) -> i32
      %sum2 = arith.addi %sum, %r : i32
      %iter2 = arith.addi %iter, %c1 : i32
      cf.br ^loop(%iter2, %sum2 : i32, i32)

    ^done:
      %val = sim.fmt.dec %sum : i32
      %line = sim.fmt.concat (%fmt_prefix, %val, %fmt_nl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
