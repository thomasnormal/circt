// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_allows_llvm_scalar_int_ops() -> (i1) {
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
      %sum = llvm.add %x, %one : i8
      %cmp = llvm.icmp "ugt" %sum, %x : i8
      %sel = llvm.select %cmp, %sum, %x : i1, i8
      %eq = llvm.icmp "eq" %sel, %sum : i8
      verif.assert %eq : i1
      verif.yield %x : i8
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_scalar_int_ops
// CHECK: smt.solver
// CHECK-NOT: llvm.add
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.select
// CHECK: arith.addi
// CHECK: arith.cmpi ugt
// CHECK: arith.select
// CHECK: arith.cmpi eq
