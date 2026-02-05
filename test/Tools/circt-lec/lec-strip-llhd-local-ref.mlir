// RUN: circt-lec --emit-mlir -c1=local_ref_a -c2=local_ref_b %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK-NOT: llhd.

module {
  hw.module @local_ref_a(in %a : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %0 = llhd.combinational -> i1 {
      %ptr = llvm.alloca %c1 x i1 : (i64) -> !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i1>
      llhd.drv %ref, %a after %t0 : i1
      %p = llhd.prb %ref : i1
      llhd.yield %p : i1
    }
    hw.output %0 : i1
  }
  hw.module @local_ref_b(in %a : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %0 = llhd.combinational -> i1 {
      %ptr = llvm.alloca %c1 x i1 : (i64) -> !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i1>
      llhd.drv %ref, %a after %t0 : i1
      %p = llhd.prb %ref : i1
      llhd.yield %p : i1
    }
    hw.output %0 : i1
  }
}
