// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// Intentionally not marked `constant` to cover read-only, initialized globals.
llvm.mlir.global internal @__circt_proc_assertions_enabled(1 : i1) : i1

func.func @for_smtlib_allows_llvm_readonly_global_load() -> (i1) {
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
    ^bb0(%clk: !seq.clock):
      %addr = llvm.mlir.addressof @__circt_proc_assertions_enabled : !llvm.ptr
      %enabled = llvm.load %addr : !llvm.ptr -> i1
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_readonly_global_load
// CHECK: smt.solver
// CHECK: arith.constant true
// CHECK-NOT: llvm.mlir.addressof
// CHECK-NOT: llvm.load
