// RUN: circt-opt %s --convert-verif-to-smt='for-smtlib-export=true' --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

llvm.func @malloc(i64) -> !llvm.ptr

func.func @for_smtlib_legalizes_malloc_dynamic_gep_scalar_load() -> (i1) {
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
    %sz_arr = llvm.mlir.constant(2 : i64) : i64
    %ptr = llvm.call @malloc(%sz_arr) : (i64) -> !llvm.ptr
    %sz_idx = llvm.mlir.constant(4 : i64) : i64
    %idx_ptr = llvm.call @malloc(%sz_idx) : (i64) -> !llvm.ptr
    %idx = llvm.load %idx_ptr : !llvm.ptr -> i32
    %elt = llvm.getelementptr %ptr[0, %idx] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i1>
    %v = llvm.load %elt : !llvm.ptr -> i1
    verif.assert %v : i1
    verif.yield %v : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @for_smtlib_legalizes_malloc_dynamic_gep_scalar_load()
// CHECK: smt.solver
// CHECK: smt.declare_fun : !smt.bv<1>
// CHECK-NOT: llvm.call @malloc
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
