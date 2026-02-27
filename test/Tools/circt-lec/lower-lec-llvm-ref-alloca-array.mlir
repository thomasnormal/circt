// RUN: circt-opt %s --lower-lec-llvm | FileCheck %s

hw.module @lower_lec_llvm_ref_alloca_array(
    in %in : !hw.array<2xstruct<value: i8, unknown: i8>>,
    out out : !hw.array<2xstruct<value: i8, unknown: i8>>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %c0 = hw.constant 0 : i1
  %c1 = hw.constant 1 : i1
  %undef_arr = llvm.mlir.undef : !llvm.array<2 x struct<(i8, i8)>>
  %undef_elem = llvm.mlir.undef : !llvm.struct<(i8, i8)>

  %in0 = hw.array_get %in[%c0] : !hw.array<2xstruct<value: i8, unknown: i8>>, i1
  %v0 = hw.struct_extract %in0["value"] : !hw.struct<value: i8, unknown: i8>
  %u0 = hw.struct_extract %in0["unknown"] : !hw.struct<value: i8, unknown: i8>
  %e00 = llvm.insertvalue %v0, %undef_elem[0] : !llvm.struct<(i8, i8)>
  %e01 = llvm.insertvalue %u0, %e00[1] : !llvm.struct<(i8, i8)>
  %arr0 = llvm.insertvalue %e01, %undef_arr[0] : !llvm.array<2 x struct<(i8, i8)>>

  %in1 = hw.array_get %in[%c1] : !hw.array<2xstruct<value: i8, unknown: i8>>, i1
  %v1 = hw.struct_extract %in1["value"] : !hw.struct<value: i8, unknown: i8>
  %u1 = hw.struct_extract %in1["unknown"] : !hw.struct<value: i8, unknown: i8>
  %e10 = llvm.insertvalue %v1, %undef_elem[0] : !llvm.struct<(i8, i8)>
  %e11 = llvm.insertvalue %u1, %e10[1] : !llvm.struct<(i8, i8)>
  %arr1 = llvm.insertvalue %e11, %arr0[1] : !llvm.array<2 x struct<(i8, i8)>>

  %ptr = llvm.alloca %one x !llvm.array<2 x struct<(i8, i8)>> : (i64) -> !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.array<2xstruct<value: i8, unknown: i8>>>
  llvm.store %arr1, %ptr : !llvm.array<2 x struct<(i8, i8)>>, !llvm.ptr
  %load = llvm.load %ptr : !llvm.ptr -> !llvm.array<2 x struct<(i8, i8)>>
  %cast = builtin.unrealized_conversion_cast %load : !llvm.array<2 x struct<(i8, i8)>> to !hw.array<2xstruct<value: i8, unknown: i8>>
  hw.output %cast : !hw.array<2xstruct<value: i8, unknown: i8>>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca_array
// CHECK: hw.output
// CHECK-NOT: llvm.
