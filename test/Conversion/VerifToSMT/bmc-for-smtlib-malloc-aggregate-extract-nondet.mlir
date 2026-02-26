// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

llvm.func @malloc(i64) -> !llvm.ptr

func.func @for_smtlib_legalizes_malloc_aggregate_extracts() -> (i1) {
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
    %sz = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.call @malloc(%sz) : (i64) -> !llvm.ptr
    %obj = llvm.getelementptr %ptr[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i1, i1)>
    %agg = llvm.load %obj : !llvm.ptr -> !llvm.struct<(i1, i1)>
    %v = llvm.extractvalue %agg[0] : !llvm.struct<(i1, i1)>
    verif.assert %v : i1
    verif.yield %v : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @for_smtlib_legalizes_malloc_aggregate_extracts()
// CHECK: smt.solver
// CHECK: smt.declare_fun : !smt.bv<1>
// CHECK-NOT: llvm.call @malloc
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.extractvalue
