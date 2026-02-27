// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

func.func @for_smtlib_legalizes_alloca_aggregate_extracts() -> (i1) {
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
    %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %v0 = llvm.mlir.constant(true) : i1
    %v1 = llvm.insertvalue %v0, %undef[0] : !llvm.struct<(i1, i1)>
    llvm.store %v1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
    %ld = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
    %v = llvm.extractvalue %ld[0] : !llvm.struct<(i1, i1)>
    verif.assert %v : i1
    verif.yield %v : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @for_smtlib_legalizes_alloca_aggregate_extracts
// CHECK: smt.solver
// CHECK-NOT: llvm.mlir.undef
// CHECK-NOT: llvm.insertvalue
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.extractvalue
