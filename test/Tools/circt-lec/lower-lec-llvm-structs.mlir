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

hw.module @lower_lec_llvm_structs_multi_store(
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %tmp2 = llvm.insertvalue %unknown, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp3 = llvm.insertvalue %value, %tmp2[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  llvm.store %tmp3, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
  %cast = builtin.unrealized_conversion_cast %load : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %cast : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_structs_multi_store
// CHECK: hw.struct_create
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_structs_mux(
    in %cond : i1,
    in %lhs : !hw.struct<value: i1, unknown: i1>,
    in %rhs : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %lhs_value = hw.struct_extract %lhs["value"] : !hw.struct<value: i1, unknown: i1>
  %lhs_unknown = hw.struct_extract %lhs["unknown"] : !hw.struct<value: i1, unknown: i1>
  %lhs0 = llvm.insertvalue %lhs_value, %undef[0] : !llvm.struct<(i1, i1)>
  %lhs1 = llvm.insertvalue %lhs_unknown, %lhs0[1] : !llvm.struct<(i1, i1)>
  %rhs_value = hw.struct_extract %rhs["value"] : !hw.struct<value: i1, unknown: i1>
  %rhs_unknown = hw.struct_extract %rhs["unknown"] : !hw.struct<value: i1, unknown: i1>
  %rhs0 = llvm.insertvalue %rhs_value, %undef[0] : !llvm.struct<(i1, i1)>
  %rhs1 = llvm.insertvalue %rhs_unknown, %rhs0[1] : !llvm.struct<(i1, i1)>
  %mux = comb.mux %cond, %lhs1, %rhs1 : !llvm.struct<(i1, i1)>
  %cast = builtin.unrealized_conversion_cast %mux : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %cast : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_structs_mux
// CHECK: comb.mux
// CHECK: hw.struct_create
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_structs_select(
    in %cond : i1,
    in %lhs : !hw.struct<value: i1, unknown: i1>,
    in %rhs : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %lhs_value = hw.struct_extract %lhs["value"] : !hw.struct<value: i1, unknown: i1>
  %lhs_unknown = hw.struct_extract %lhs["unknown"] : !hw.struct<value: i1, unknown: i1>
  %lhs0 = llvm.insertvalue %lhs_value, %undef[0] : !llvm.struct<(i1, i1)>
  %lhs1 = llvm.insertvalue %lhs_unknown, %lhs0[1] : !llvm.struct<(i1, i1)>
  %rhs_value = hw.struct_extract %rhs["value"] : !hw.struct<value: i1, unknown: i1>
  %rhs_unknown = hw.struct_extract %rhs["unknown"] : !hw.struct<value: i1, unknown: i1>
  %rhs0 = llvm.insertvalue %rhs_value, %undef[0] : !llvm.struct<(i1, i1)>
  %rhs1 = llvm.insertvalue %rhs_unknown, %rhs0[1] : !llvm.struct<(i1, i1)>
  %sel = llvm.select %cond, %lhs1, %rhs1 : i1, !llvm.struct<(i1, i1)>
  %cast = builtin.unrealized_conversion_cast %sel : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %cast : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_structs_select
// CHECK-DAG: comb.mux
// CHECK-DAG: hw.struct_create
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_structs_partial_insert(
    in %lhs : !hw.struct<value: i1, unknown: i1>,
    in %rhs : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %lhs_value = hw.struct_extract %lhs["value"] : !hw.struct<value: i1, unknown: i1>
  %lhs_unknown = hw.struct_extract %lhs["unknown"] : !hw.struct<value: i1, unknown: i1>
  %lhs0 = llvm.insertvalue %lhs_value, %undef[0] : !llvm.struct<(i1, i1)>
  %lhs1 = llvm.insertvalue %lhs_unknown, %lhs0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  llvm.store %lhs1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %load = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i1, i1)>
  %rhs_value = hw.struct_extract %rhs["value"] : !hw.struct<value: i1, unknown: i1>
  %new = llvm.insertvalue %rhs_value, %load[0] : !llvm.struct<(i1, i1)>
  %cast = builtin.unrealized_conversion_cast %new : !llvm.struct<(i1, i1)> to !hw.struct<value: i1, unknown: i1>
  hw.output %cast : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_structs_partial_insert
// CHECK: hw.struct_create
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_ref_alloca(
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  llvm.store %tmp1, %ptr : !llvm.struct<(i1, i1)>, !llvm.ptr
  %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
  hw.output %probe : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca
// CHECK: llhd.sig
// CHECK-DAG: llhd.drv
// CHECK-DAG: llhd.prb
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_ref_alloca_cast(
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  %ptr_as1 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<1>
  %ptr_as0 = llvm.addrspacecast %ptr_as1 : !llvm.ptr<1> to !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %ptr_as0 : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  llvm.store %tmp1, %ptr_as0 : !llvm.struct<(i1, i1)>, !llvm.ptr
  %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
  hw.output %probe : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca_cast
// CHECK: llhd.sig
// CHECK-DAG: llhd.drv
// CHECK-DAG: llhd.prb
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_ref_alloca_block_arg(
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %outval = llhd.combinational -> !hw.struct<value: i1, unknown: i1> {
    %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
    cf.br ^bb1(%ptr : !llvm.ptr)

  ^bb1(%arg0: !llvm.ptr):
    %ref = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
    llvm.store %tmp1, %arg0 : !llvm.struct<(i1, i1)>, !llvm.ptr
    %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
    llhd.yield %probe : !hw.struct<value: i1, unknown: i1>
  }
  hw.output %outval : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca_block_arg
// CHECK: llhd.sig
// CHECK-DAG: llhd.drv
// CHECK-DAG: llhd.prb
// CHECK-NOT: llvm.

hw.module @lower_lec_llvm_ref_alloca_select(
    in %cond : i1,
    in %in : !hw.struct<value: i1, unknown: i1>,
    out out : !hw.struct<value: i1, unknown: i1>) {
  %one = llvm.mlir.constant(1 : i64) : i64
  %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
  %value = hw.struct_extract %in["value"] : !hw.struct<value: i1, unknown: i1>
  %unknown = hw.struct_extract %in["unknown"] : !hw.struct<value: i1, unknown: i1>
  %tmp0 = llvm.insertvalue %value, %undef[0] : !llvm.struct<(i1, i1)>
  %tmp1 = llvm.insertvalue %unknown, %tmp0[1] : !llvm.struct<(i1, i1)>
  %ptr = llvm.alloca %one x !llvm.struct<(i1, i1)> : (i64) -> !llvm.ptr
  %ptr_as1 = llvm.addrspacecast %ptr : !llvm.ptr to !llvm.ptr<1>
  %ptr_as0 = llvm.addrspacecast %ptr_as1 : !llvm.ptr<1> to !llvm.ptr
  %sel = llvm.select %cond, %ptr, %ptr_as0 : i1, !llvm.ptr
  %ref = builtin.unrealized_conversion_cast %sel : !llvm.ptr to !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  llvm.store %tmp1, %sel : !llvm.struct<(i1, i1)>, !llvm.ptr
  %probe = llhd.prb %ref : !hw.struct<value: i1, unknown: i1>
  hw.output %probe : !hw.struct<value: i1, unknown: i1>
}

// CHECK-LABEL: hw.module @lower_lec_llvm_ref_alloca_select
// CHECK: llhd.sig
// CHECK-DAG: llhd.drv
// CHECK-DAG: llhd.prb
// CHECK-NOT: llvm.
