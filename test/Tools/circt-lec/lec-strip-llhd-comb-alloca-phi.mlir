// RUN: circt-opt -strip-llhd-interface-signals %s | FileCheck %s

hw.module @comb_alloca_phi(in %cond : i1, in %in0 : i1, in %in1 : i1,
                           out out : i1) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %comb = llhd.combinational -> i1 {
    %ptr0 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    %ptr1 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    llvm.store %in0, %ptr0 : i1, !llvm.ptr
    llvm.store %in1, %ptr1 : i1, !llvm.ptr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    cf.br ^bb3(%ptr0 : !llvm.ptr)
  ^bb2:
    cf.br ^bb3(%ptr1 : !llvm.ptr)
  ^bb3(%arg : !llvm.ptr):
    %load = llvm.load %arg : !llvm.ptr -> i1
    llhd.yield %load : i1
  }
  hw.output %comb : i1
}

// CHECK-LABEL: hw.module @comb_alloca_phi
// CHECK-NOT: llhd.combinational
// CHECK-NOT: llhd_comb
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
// CHECK-NOT: llvm.alloca
