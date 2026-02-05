// RUN: circt-opt -strip-llhd-interface-signals %s | FileCheck %s

hw.module @comb_alloca(in %cond : i1, in %in0 : i1, in %in1 : i1, out out : i1) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %comb = llhd.combinational -> i1 {
    %ptr = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    llvm.store %in0, %ptr : i1, !llvm.ptr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    llvm.store %in1, %ptr : i1, !llvm.ptr
    cf.br ^bb2
  ^bb2:
    %load = llvm.load %ptr : !llvm.ptr -> i1
    llhd.yield %load : i1
  }
  hw.output %comb : i1
}

// CHECK-LABEL: hw.module @comb_alloca
// CHECK-NOT: llhd.combinational
// CHECK-NOT: llhd_comb
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
