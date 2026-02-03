// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test unpacked struct containing dynamic types (strings)
// These should convert to LLVM struct types

// CHECK-LABEL: func.func @UnpackedStructWithString
// CHECK-SAME: () -> !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
func.func @UnpackedStructWithString() -> !moore.ustruct<{a: string, b: string}> {
  // CHECK: llvm.mlir.zero : !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(struct<(ptr, i64)>, struct<(ptr, i64)>)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{a: string, b: string}>>
  %val = moore.read %var : <ustruct<{a: string, b: string}>>
  return %val : !moore.ustruct<{a: string, b: string}>
}

// CHECK-LABEL: func.func @UnpackedStructWithMixedTypes
// CHECK-SAME: () -> !llvm.struct<(i32, struct<(ptr, i64)>)>
func.func @UnpackedStructWithMixedTypes() -> !moore.ustruct<{num: i32, name: string}> {
  // Unpacked struct with both string and integer fields
  // CHECK: llvm.mlir.zero : !llvm.struct<(i32, struct<(ptr, i64)>)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(i32, struct<(ptr, i64)>)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{num: i32, name: string}>>
  %val = moore.read %var : <ustruct<{num: i32, name: string}>>
  return %val : !moore.ustruct<{num: i32, name: string}>
}

// CHECK-LABEL: func.func @NestedUnpackedStructWithString
// CHECK-SAME: () -> !llvm.struct<(struct<(i32, struct<(ptr, i64)>)>, i32)>
func.func @NestedUnpackedStructWithString() -> !moore.ustruct<{inner: ustruct<{num: i32, name: string}>, count: i32}> {
  // Nested unpacked struct containing strings
  // The nested struct with string becomes LLVM struct, so the outer struct also becomes LLVM struct
  // CHECK: llvm.mlir.zero : !llvm.struct<(struct<(i32, struct<(ptr, i64)>)>, i32)>
  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(struct<(i32, struct<(ptr, i64)>)>, i32)>
  // CHECK: llvm.store
  %var = moore.variable : <ustruct<{inner: ustruct<{num: i32, name: string}>, count: i32}>>
  %val = moore.read %var : <ustruct<{inner: ustruct<{num: i32, name: string}>, count: i32}>>
  return %val : !moore.ustruct<{inner: ustruct<{num: i32, name: string}>, count: i32}>
}

// Test that unpacked structs without dynamic types use llvm.alloca in functions
// (local variables have immediate memory semantics)
// The return type converts to hw.struct since there are no dynamic types
// CHECK-LABEL: func.func @UnpackedStructWithoutDynamic
// CHECK-SAME: () -> !hw.struct<a: i32, b: i32>
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(i32, i32)>
// CHECK: llvm.store
// CHECK: return
func.func @UnpackedStructWithoutDynamic() -> !moore.ustruct<{a: i32, b: i32}> {
  %var = moore.variable name "var" : <ustruct<{a: i32, b: i32}>>
  %val = moore.read %var : <ustruct<{a: i32, b: i32}>>
  return %val : !moore.ustruct<{a: i32, b: i32}>
}
