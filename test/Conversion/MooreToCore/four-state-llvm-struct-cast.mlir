// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

hw.module @cast_from_llvm(out out : !hw.struct<value: i8, unknown: i8>) {
  %undef = llvm.mlir.undef : !llvm.struct<(i8, i8)>
  %val = hw.constant 90 : i8
  %unk = hw.constant 15 : i8
  %s0 = llvm.insertvalue %val, %undef[0] : !llvm.struct<(i8, i8)>
  %s1 = llvm.insertvalue %unk, %s0[1] : !llvm.struct<(i8, i8)>
  %hw = builtin.unrealized_conversion_cast %s1 : !llvm.struct<(i8, i8)> to !hw.struct<value: i8, unknown: i8>
  hw.output %hw : !hw.struct<value: i8, unknown: i8>
}

// CHECK-NOT: unrealized_conversion_cast
// CHECK: hw.aggregate_constant [90 : i8, 15 : i8]
