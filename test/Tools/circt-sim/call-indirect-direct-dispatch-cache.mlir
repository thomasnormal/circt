// RUN: env CIRCT_SIM_PROFILE_SUMMARY_AT_EXIT=1 circt-sim %s 2>&1 | FileCheck %s
//
// Test call_indirect resolution cache. A vtable with two entries is called
// multiple times. The resolution cache should install entries on first
// miss and serve hits on subsequent calls.
//
// CHECK: sum=85
// CHECK: [circt-sim] call_indirect resolution cache: installs=
// CHECK-SAME: hits=
// CHECK-SAME: misses=
// CHECK-SAME: entries=

module {
  func.func @add_ten(%x: i32) -> i32 {
    %c10 = arith.constant 10 : i32
    %r = arith.addi %x, %c10 : i32
    return %r : i32
  }
  func.func @add_twenty(%x: i32) -> i32 {
    %c20 = arith.constant 20 : i32
    %r = arith.addi %x, %c20 : i32
    return %r : i32
  }

  llvm.mlir.global internal @vtable(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = [[0, @add_ten], [1, @add_twenty]]} : !llvm.array<2 x ptr>

  hw.module @top() {
    %fmt_prefix = sim.fmt.literal "sum="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      cf.br ^entry
    ^entry:
      %vt = llvm.mlir.addressof @vtable : !llvm.ptr
      %c0 = arith.constant 0 : i64

      // Load slot 0 (add_ten)
      %slot0 = llvm.getelementptr %vt[%c0, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fp0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fp0_cast = builtin.unrealized_conversion_cast %fp0 : !llvm.ptr to (i32) -> i32

      // Load slot 1 (add_twenty)
      %slot1 = llvm.getelementptr %vt[%c0, 1] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<2 x ptr>
      %fp1 = llvm.load %slot1 : !llvm.ptr -> !llvm.ptr
      %fp1_cast = builtin.unrealized_conversion_cast %fp1 : !llvm.ptr to (i32) -> i32

      // 5 -> 15 -> 25 -> 35 (add_ten x3)
      %c5 = arith.constant 5 : i32
      %r0 = func.call_indirect %fp0_cast(%c5) : (i32) -> i32
      %r1 = func.call_indirect %fp0_cast(%r0) : (i32) -> i32
      %r2 = func.call_indirect %fp0_cast(%r1) : (i32) -> i32

      // 35 -> 55 -> 75 (add_twenty x2)
      %r3 = func.call_indirect %fp1_cast(%r2) : (i32) -> i32
      %r4 = func.call_indirect %fp1_cast(%r3) : (i32) -> i32

      // 75 -> 85 (add_ten x1)
      %r5 = func.call_indirect %fp0_cast(%r4) : (i32) -> i32

      %fmt_val = sim.fmt.dec %r5 : i32
      %line = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
      sim.proc.print %line
      llhd.halt
    }
    hw.output
  }
}
