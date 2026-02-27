// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_allows_llvm_nested_insert_extract_projection() -> (i1) {
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
      %undef_arr = llvm.mlir.undef : !llvm.array<2 x struct<(i1, i1)>>
      %undef_s = llvm.mlir.undef : !llvm.struct<(i1, i1)>
      %one = llvm.mlir.constant(true) : i1
      %zero = llvm.mlir.constant(false) : i1
      %s0 = llvm.insertvalue %one, %undef_s[0] : !llvm.struct<(i1, i1)>
      %s1 = llvm.insertvalue %zero, %s0[1] : !llvm.struct<(i1, i1)>
      %arr0 = llvm.insertvalue %s1, %undef_arr[0] : !llvm.array<2 x struct<(i1, i1)>>
      %elem = llvm.extractvalue %arr0[0] : !llvm.array<2 x struct<(i1, i1)>>
      %enabled = llvm.extractvalue %elem[0] : !llvm.struct<(i1, i1)>
      verif.assert %enabled : i1
      verif.yield %enabled : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @for_smtlib_allows_llvm_nested_insert_extract_projection
// CHECK: smt.solver
// CHECK-NOT: llvm.extractvalue
// CHECK-NOT: llvm.insertvalue
// CHECK-NOT: llvm.mlir.undef
