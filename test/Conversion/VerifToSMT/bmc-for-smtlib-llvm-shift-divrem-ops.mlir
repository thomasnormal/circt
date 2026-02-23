// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_allows_llvm_shift_divrem_ops() -> (i1) {
  %bmc = verif.bmc bound 2 num_regs 0 initial_values []
  init {
    %c0_i1 = hw.constant 0 : i1
    %clk = seq.to_clock %c0_i1
    verif.yield %clk : !seq.clock
  }
  loop {
    ^bb0(%clk: !seq.clock):
      verif.yield %clk : !seq.clock
  }
  circuit {
    ^bb0(%clk: !seq.clock, %x: i8):
      %one = llvm.mlir.constant(1 : i8) : i8
      %two = llvm.mlir.constant(2 : i8) : i8
      %shl = llvm.shl %x, %one : i8
      %lshr = llvm.lshr %shl, %one : i8
      %ashr = llvm.ashr %shl, %one : i8
      %udiv = llvm.udiv %lshr, %two : i8
      %sdiv = llvm.sdiv %ashr, %two : i8
      %urem = llvm.urem %udiv, %two : i8
      %srem = llvm.srem %sdiv, %two : i8
      %ok = llvm.icmp "uge" %urem, %srem : i8
      verif.assert %ok : i1
      verif.yield %x : i8
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_shift_divrem_ops
// CHECK: smt.solver
// CHECK-NOT: llvm.shl
// CHECK-NOT: llvm.lshr
// CHECK-NOT: llvm.ashr
// CHECK-NOT: llvm.udiv
// CHECK-NOT: llvm.sdiv
// CHECK-NOT: llvm.urem
// CHECK-NOT: llvm.srem
// CHECK: arith.shli
// CHECK: arith.shrui
// CHECK: arith.shrsi
// CHECK: arith.divui
// CHECK: arith.divsi
// CHECK: arith.remui
// CHECK: arith.remsi
