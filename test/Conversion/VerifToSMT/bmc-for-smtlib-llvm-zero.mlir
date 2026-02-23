// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_allows_llvm_zero_scalar() -> (i1) {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values []
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
    ^bb0(%clk: !seq.clock):
      %z0 = llvm.mlir.zero : i8
      %z1 = llvm.mlir.zero : i8
      %ok = llvm.icmp "eq" %z0, %z1 : i8
      verif.assert %ok : i1
      verif.yield %ok : i1
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_zero_scalar
// CHECK: smt.solver
// CHECK: arith.constant true
// CHECK-NOT: llvm.mlir.zero
