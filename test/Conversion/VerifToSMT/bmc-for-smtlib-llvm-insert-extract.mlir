// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_allows_llvm_insert_extract_projection() -> (i1) {
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
      %undef = llvm.mlir.undef : !llvm.struct<(i1, i8)>
      %one = llvm.mlir.constant(true) : i1
      %s = llvm.insertvalue %one, %undef[0] : !llvm.struct<(i1, i8)>
      %enabled = llvm.extractvalue %s[0] : !llvm.struct<(i1, i8)>
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_insert_extract_projection
// CHECK: smt.solver
// CHECK: arith.constant true
// CHECK-NOT: llvm.extractvalue
