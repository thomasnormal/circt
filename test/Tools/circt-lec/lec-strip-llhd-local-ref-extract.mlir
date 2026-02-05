// RUN: circt-lec --emit-mlir -c1=local_ref_extract_a -c2=local_ref_extract_b %s %s | FileCheck %s

// CHECK: smt.solver
// CHECK-NOT: llhd.

module {
  hw.module @local_ref_extract_a(in %a : i1, out o : i8) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c0_i3 = hw.constant 0 : i3
    %c0_i8 = hw.constant 0 : i8
    %0 = llhd.combinational -> i8 {
      %ptr = llvm.alloca %c1 x i8 : (i64) -> !llvm.ptr
      %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<i8>
      llhd.drv %ref, %c0_i8 after %t0 : i8
      %bit = llhd.sig.extract %ref from %c0_i3 : <i8> -> <i1>
      llhd.drv %bit, %a after %t0 : i1
      %p = llhd.prb %ref : i8
      llhd.yield %p : i8
    }
    hw.output %0 : i8
  }
  hw.module @local_ref_extract_b(in %a : i1, out o : i8) {
    %c0_i7 = hw.constant 0 : i7
    %0 = comb.concat %c0_i7, %a : i7, i1
    hw.output %0 : i8
  }
}
