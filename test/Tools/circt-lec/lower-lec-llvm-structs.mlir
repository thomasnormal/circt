// RUN: circt-opt -lower-lec-llvm %s | FileCheck %s

hw.module @lower_lec_llvm_structs(in %in : !hw.struct<value: i1, unknown: i1>,
                                  out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
  %cast = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %cast : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_structs
// CHECK: hw.struct_create
// CHECK-NOT: llvm.
