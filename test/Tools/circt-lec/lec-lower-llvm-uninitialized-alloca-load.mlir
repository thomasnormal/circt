// RUN: circt-opt %s --pass-pipeline='builtin.module(lower-lec-llvm)' | FileCheck %s

module {
  func.func @uninitialized_alloca_extract_value() -> i64 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %one x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr
    %agg = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %value = llvm.extractvalue %agg[0] : !llvm.struct<(i64, i64)>
    return %value : i64
  }

  func.func @uninitialized_alloca_extract_unknown() -> i64 {
    %one = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %one x !llvm.struct<(i64, i64)> : (i64) -> !llvm.ptr
    %agg = llvm.load %ptr : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %unknown = llvm.extractvalue %agg[1] : !llvm.struct<(i64, i64)>
    return %unknown : i64
  }
}

// CHECK-LABEL: func.func @uninitialized_alloca_extract_value
// CHECK-NOT: llvm.
// CHECK: hw.constant 0 : i64

// CHECK-LABEL: func.func @uninitialized_alloca_extract_unknown
// CHECK-NOT: llvm.
// CHECK: hw.constant -1 : i64
