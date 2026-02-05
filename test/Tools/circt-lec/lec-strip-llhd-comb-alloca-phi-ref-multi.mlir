// RUN: circt-opt -strip-llhd-interface-signals %s | FileCheck %s
// RUN: circt-opt -strip-llhd-interface-signals='strict-llhd=true' %s | FileCheck %s

hw.module @comb_alloca_phi_ref_multi(in %cond : i1, in %in0 : i1, in %in1 : i1,
                                     out out : i1) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %comb = llhd.combinational -> i1 {
    %ptr0 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    %ptr1 = llvm.alloca %one x i1 : (i64) -> !llvm.ptr
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    llvm.store %in0, %ptr0 : i1, !llvm.ptr
    cf.br ^bb3(%ptr0 : !llvm.ptr)
  ^bb2:
    llvm.store %in1, %ptr1 : i1, !llvm.ptr
    cf.br ^bb3(%ptr1 : !llvm.ptr)
  ^bb3(%arg : !llvm.ptr):
    %ref = builtin.unrealized_conversion_cast %arg : !llvm.ptr to !llhd.ref<i1>
    %val = llhd.prb %ref : i1
    llhd.drv %ref, %val after %t0 : i1
    %load = llvm.load %arg : !llvm.ptr -> i1
    llhd.yield %load : i1
  }
  hw.output %comb : i1
}

// CHECK-NOT: llhd.
// CHECK-NOT: llhd_comb
