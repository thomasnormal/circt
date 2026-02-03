// RUN: circt-opt %s --lower-lec-llvm | FileCheck %s

module {
  hw.module @llvm_mux_extract(
      in %cond : i1,
      in %a_val : i8,
      in %a_unk : i8,
      in %b_val : i8,
      in %b_unk : i8,
      out out : !hw.struct<value: i8, unknown: i8>) {
    %undef = llvm.mlir.undef : !llvm.struct<(i8, i8)>
    %a0 = llvm.insertvalue %a_val, %undef[0] : !llvm.struct<(i8, i8)>
    %a1 = llvm.insertvalue %a_unk, %a0[1] : !llvm.struct<(i8, i8)>
    %b0 = llvm.insertvalue %b_val, %undef[0] : !llvm.struct<(i8, i8)>
    %b1 = llvm.insertvalue %b_unk, %b0[1] : !llvm.struct<(i8, i8)>
    %sel = comb.mux %cond, %a1, %b1 : !llvm.struct<(i8, i8)>
    %val = llvm.extractvalue %sel[0] : !llvm.struct<(i8, i8)>
    %unk = llvm.extractvalue %sel[1] : !llvm.struct<(i8, i8)>
    %out = hw.struct_create (%val, %unk) : !hw.struct<value: i8, unknown: i8>
    hw.output %out : !hw.struct<value: i8, unknown: i8>
  }
}

// CHECK-NOT: llvm.
// CHECK: comb.mux
// CHECK: comb.mux
// CHECK: hw.struct_create
