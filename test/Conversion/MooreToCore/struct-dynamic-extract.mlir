// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test StructExtractOp and StructCreateOp with dynamic fields (strings, classes)
// These types convert to LLVM structs and require LLVM ops instead of HW ops.

module {
  moore.class.classdecl @MyClass {
    moore.class.propertydecl @value : !moore.i32
  }

  // CHECK-LABEL: func.func @test_extract_class_field
  // CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, i32)>
  func.func @test_extract_class_field(%arg0: !moore.ustruct<{data: class<@MyClass>, count: i32}>) -> !moore.i32 {
    %0 = moore.struct_extract %arg0, "count" : ustruct<{data: class<@MyClass>, count: i32}> -> i32
    return %0 : !moore.i32
  }

  // CHECK-LABEL: func.func @test_extract_string_field
  // CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(struct<(ptr, i64)>, i32)>
  func.func @test_extract_string_field(%arg0: !moore.ustruct<{name: string, id: i32}>) -> !moore.string {
    %0 = moore.struct_extract %arg0, "name" : ustruct<{name: string, id: i32}> -> string
    return %0 : !moore.string
  }

  // CHECK-LABEL: func.func @test_extract_static
  // CHECK: hw.struct_extract {{.*}}["a"] : !hw.struct<a: i32, b: i32>
  func.func @test_extract_static(%arg0: !moore.ustruct<{a: i32, b: i32}>) -> !moore.i32 {
    %0 = moore.struct_extract %arg0, "a" : ustruct<{a: i32, b: i32}> -> i32
    return %0 : !moore.i32
  }

  // CHECK-LABEL: func.func @test_create_dynamic
  // CHECK: llvm.mlir.undef : !llvm.struct<(struct<(ptr, i64)>, i32)>
  // CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(struct<(ptr, i64)>, i32)>
  // CHECK: llvm.insertvalue {{.*}}[1] : !llvm.struct<(struct<(ptr, i64)>, i32)>
  func.func @test_create_dynamic(%name: !moore.string, %id: !moore.i32) -> !moore.ustruct<{name: string, id: i32}> {
    %0 = moore.struct_create %name, %id : !moore.string, !moore.i32 -> ustruct<{name: string, id: i32}>
    return %0 : !moore.ustruct<{name: string, id: i32}>
  }

  // CHECK-LABEL: func.func @test_create_static
  // CHECK: hw.struct_create
  func.func @test_create_static(%a: !moore.i32, %b: !moore.i32) -> !moore.ustruct<{a: i32, b: i32}> {
    %0 = moore.struct_create %a, %b : !moore.i32, !moore.i32 -> ustruct<{a: i32, b: i32}>
    return %0 : !moore.ustruct<{a: i32, b: i32}>
  }
}
