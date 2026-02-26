// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

llvm.mlir.global internal constant @__circt_assert_array(dense<[1, 0, 1]> : tensor<3xi1>) : !llvm.array<3 x i1>

func.func @for_smtlib_allows_llvm_constant_global_gep_with_dynamic_constant_indices() -> (i1) {
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
      %addr = llvm.mlir.addressof @__circt_assert_array : !llvm.ptr
      %idx0 = llvm.mlir.constant(0 : i32) : i32
      %idx2 = llvm.mlir.constant(2 : i32) : i32
      %elt = llvm.getelementptr %addr[%idx0, %idx2] : (!llvm.ptr, i32, i32) -> !llvm.ptr, !llvm.array<3 x i1>
      %enabled = llvm.load %elt : !llvm.ptr -> i1
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_constant_global_gep_with_dynamic_constant_indices
// CHECK: smt.solver
// CHECK: smt.constant false
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
