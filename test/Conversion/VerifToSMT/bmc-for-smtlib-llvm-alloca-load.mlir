// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_legalizes_alloca_scalar_loads() -> (i1) {
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
    %one = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    %v0 = llvm.mlir.constant(true) : i1
    llvm.store %v0, %ptr : i1, !llvm.ptr
    %v = llvm.load %ptr : !llvm.ptr -> i1
    verif.assert %v : i1
    verif.yield %v : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @for_smtlib_legalizes_alloca_scalar_loads
// CHECK: smt.solver
// CHECK-NOT: llvm.alloca
// CHECK-NOT: llvm.store
// CHECK-NOT: llvm.load
