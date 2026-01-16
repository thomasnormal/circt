// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test StructExtractRefOp conversion for structs with dynamic fields (strings)
// vs structs with only static fields (integers).

// Test struct with string field - should use LLVM GEP
// CHECK-LABEL: func.func @test_struct_string_field
// CHECK: llvm.getelementptr
func.func @test_struct_string_field(%arg0: !moore.ref<ustruct<{a: string, b: i32}>>) -> !moore.ref<string> {
  %0 = moore.struct_extract_ref %arg0, "a" : <ustruct<{a: string, b: i32}>> -> <string>
  return %0 : !moore.ref<string>
}

// Test struct with string field - extracting integer field still uses LLVM GEP
// CHECK-LABEL: func.func @test_struct_string_field_extract_int
// CHECK: llvm.getelementptr
func.func @test_struct_string_field_extract_int(%arg0: !moore.ref<ustruct<{a: string, b: i32}>>) -> !moore.ref<i32> {
  %0 = moore.struct_extract_ref %arg0, "b" : <ustruct<{a: string, b: i32}>> -> <i32>
  return %0 : !moore.ref<i32>
}

// Test struct with only integer fields - should use SigStructExtract
// CHECK-LABEL: func.func @test_struct_integer_fields
// CHECK: llhd.sig.struct_extract
func.func @test_struct_integer_fields(%arg0: !moore.ref<ustruct<{a: i32, b: i32}>>) -> !moore.ref<i32> {
  %0 = moore.struct_extract_ref %arg0, "a" : <ustruct<{a: i32, b: i32}>> -> <i32>
  return %0 : !moore.ref<i32>
}

// Test packed struct with only integer fields - should use SigStructExtract
// CHECK-LABEL: func.func @test_packed_struct_integer_fields
// CHECK: llhd.sig.struct_extract
func.func @test_packed_struct_integer_fields(%arg0: !moore.ref<struct<{x: i16, y: i16}>>) -> !moore.ref<i16> {
  %0 = moore.struct_extract_ref %arg0, "x" : <struct<{x: i16, y: i16}>> -> <i16>
  return %0 : !moore.ref<i16>
}

// Test struct with multiple string fields - uses LLVM GEP
// CHECK-LABEL: func.func @test_struct_multiple_strings
// CHECK: llvm.getelementptr
func.func @test_struct_multiple_strings(%arg0: !moore.ref<ustruct<{name: string, value: string}>>) -> !moore.ref<string> {
  %0 = moore.struct_extract_ref %arg0, "name" : <ustruct<{name: string, value: string}>> -> <string>
  return %0 : !moore.ref<string>
}
