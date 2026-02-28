// RUN: circt-opt %s --pass-pipeline='builtin.module(lower-lec-llvm)' | FileCheck %s
//
// Regression coverage: rewriteAllocaBackedLLHDRef must not use a cast op after
// erasing it while constructing dominance info.

hw.module @lower_lec_llvm_ref_alloca_dominance_regression(
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : i1) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  %ref0 = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  %ref1 = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %probe0 = llhd.prb %ref0 : !hw.struct<value: i1, unknown: i1>
  %probe1 = llhd.prb %ref1 : !hw.struct<value: i1, unknown: i1>
  %out0 = hw.struct_extract %probe0["value"] : !hw.struct<value: i1, unknown: i1>
  %out1 = hw.struct_extract %probe1["value"] : !hw.struct<value: i1, unknown: i1>
  %out = comb.and %out0, %out1 : i1
  hw.output %out : i1
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca_dominance_regression
// CHECK: llhd.sig
// CHECK-DAG: llhd.drv
// CHECK-DAG: llhd.prb
// CHECK-NOT: llvm.
