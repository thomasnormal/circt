// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

llvm.mlir.global internal constant @__circt_struct_zero(#llvm.zero) : !llvm.struct<(i1, i8)>

func.func @for_smtlib_allows_llvm_constant_struct_global_gep_load() -> (i1) {
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
      %addr = llvm.mlir.addressof @__circt_struct_zero : !llvm.ptr
      %field0 = llvm.getelementptr %addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i8)>
      %enabled = llvm.load %field0 : !llvm.ptr -> i1
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}

// CHECK: func.func @for_smtlib_allows_llvm_constant_struct_global_gep_load
// CHECK: smt.solver
// CHECK: smt.constant false
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
