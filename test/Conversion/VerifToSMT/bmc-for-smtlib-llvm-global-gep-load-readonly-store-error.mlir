// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --verify-diagnostics

// Not marked `constant`; store through GEP must prevent const-fold legalization.
llvm.mlir.global internal @__circt_assert_array(dense<[1, 0, 1]> : tensor<3xi1>) : !llvm.array<3 x i1>

func.func @for_smtlib_rejects_readonly_global_gep_load_with_store() -> (i1) {
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
      // expected-error @below {{for-smtlib-export does not support LLVM dialect operations inside verif.bmc regions; found 'llvm.mlir.addressof'}}
      %addr = llvm.mlir.addressof @__circt_assert_array : !llvm.ptr
      %elt = llvm.getelementptr %addr[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i1>
      %c0 = llvm.mlir.constant(false) : i1
      llvm.store %c0, %elt : i1, !llvm.ptr
      %enabled = llvm.load %elt : !llvm.ptr -> i1
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}
