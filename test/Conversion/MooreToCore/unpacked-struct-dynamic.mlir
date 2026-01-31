// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test unpacked struct containing dynamic types (strings)
// These should convert to LLVM struct types

// CHECK-LABEL: func.func @UnpackedStructWithString
func.func @UnpackedStructWithString() {
  // CHECK: llvm.mlir.zero : !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{a: string, b: string}>>
  return
}

// CHECK-LABEL: func.func @UnpackedStructWithMixedTypes
func.func @UnpackedStructWithMixedTypes() {
  // Unpacked struct with both string and integer fields
  // CHECK: llvm.mlir.zero : !llvm.struct<(i32, struct<(ptr, i64)>)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i32, struct<(ptr, i64)>)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{num: i32, name: string}>>
  return
}

// CHECK-LABEL: func.func @NestedUnpackedStructWithString
func.func @NestedUnpackedStructWithString() {
  // Nested unpacked struct containing strings
  // The nested struct with string becomes LLVM struct, so the outer struct also becomes LLVM struct
  // CHECK: llvm.mlir.zero : !llvm.struct<(struct<(i32, struct<(ptr, i64)>)>, i32)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(struct<(i32, struct<(ptr, i64)>)>, i32)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{inner: ustruct<{num: i32, name: string}>, count: i32}>>
  return
}

// Test that unpacked structs without dynamic types use llvm.alloca in functions
// (local variables have immediate memory semantics)
// CHECK-LABEL: func.func @UnpackedStructWithoutDynamic
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(i32, i32)>
// CHECK: llvm.store
// CHECK: return
func.func @UnpackedStructWithoutDynamic() {
  %var = moore.variable name "var" : <ustruct<{a: i32, b: i32}>>
  return
}
