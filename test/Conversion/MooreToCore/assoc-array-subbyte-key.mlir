// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr

//===----------------------------------------------------------------------===//
// Sub-byte integer key associative arrays: keySize truncation regression test
//
// Bug: `intTy.getWidth() / 8` truncated to 0 for sub-byte integer types
// (i1, i2, ..., i7), causing them to be treated as string-keyed arrays
// (keySize=0). The fix changed this to `(intTy.getWidth() + 7) / 8`.
//===----------------------------------------------------------------------===//

// --- i1 key: should get keySize=1, not 0 (string) ---

// CHECK-LABEL: hw.module @test_assoc_i1_key
hw.module @test_assoc_i1_key() {
  // CHECK-DAG: [[KEY_SIZE_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[VAL_SIZE_4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KEY_SIZE_1]], [[VAL_SIZE_4]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i32, i1>>
  moore.procedure initial {
    %key = moore.variable : <i1>
    %first = moore.assoc.first %aa, %key : <assoc_array<i32, i1>>, <i1>
    moore.return
  }
  hw.output
}

// --- i2 key: should get keySize=1, not 0 (string) ---

// CHECK-LABEL: hw.module @test_assoc_i2_key
hw.module @test_assoc_i2_key() {
  // CHECK-DAG: [[KEY_SIZE_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[VAL_SIZE_4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KEY_SIZE_1]], [[VAL_SIZE_4]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i32, i2>>
  moore.procedure initial {
    %key = moore.variable : <i2>
    %first = moore.assoc.first %aa, %key : <assoc_array<i32, i2>>, <i2>
    moore.return
  }
  hw.output
}

// --- i7 key: should get keySize=1, not 0 (string) ---

// CHECK-LABEL: hw.module @test_assoc_i7_key
hw.module @test_assoc_i7_key() {
  // CHECK-DAG: [[KEY_SIZE_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: [[VAL_SIZE_4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KEY_SIZE_1]], [[VAL_SIZE_4]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i32, i7>>
  moore.procedure initial {
    %key = moore.variable : <i7>
    %first = moore.assoc.first %aa, %key : <assoc_array<i32, i7>>, <i7>
    moore.return
  }
  hw.output
}

// --- i1 value: should also get valueSize=1, not 0 ---

// CHECK-LABEL: hw.module @test_assoc_i1_value
hw.module @test_assoc_i1_value() {
  // CHECK-DAG: [[KEY_SIZE_4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: [[VAL_SIZE_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KEY_SIZE_4]], [[VAL_SIZE_1]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i1, i32>>
  moore.procedure initial {
    %key = moore.variable : <i32>
    %first = moore.assoc.first %aa, %key : <assoc_array<i1, i32>>, <i32>
    moore.return
  }
  hw.output
}

// --- Both key and value sub-byte: i2 key, i1 value ---

// CHECK-LABEL: hw.module @test_assoc_both_subbyte
hw.module @test_assoc_both_subbyte() {
  // CHECK: [[SIZE_1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[SIZE_1]], [[SIZE_1]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i1, i2>>
  moore.procedure initial {
    %key = moore.variable : <i2>
    %first = moore.assoc.first %aa, %key : <assoc_array<i1, i2>>, <i2>
    moore.return
  }
  hw.output
}

// --- i9 key: should get keySize=2 (rounds up from 9 bits) ---

// CHECK-LABEL: hw.module @test_assoc_i9_key
hw.module @test_assoc_i9_key() {
  // CHECK-DAG: [[KEY_SIZE_2:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: [[VAL_SIZE_4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KEY_SIZE_2]], [[VAL_SIZE_4]]) : (i32, i32) -> !llvm.ptr
  %aa = moore.variable : <assoc_array<i32, i9>>
  moore.procedure initial {
    %key = moore.variable : <i9>
    %first = moore.assoc.first %aa, %key : <assoc_array<i32, i9>>, <i9>
    moore.return
  }
  hw.output
}
