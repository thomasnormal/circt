// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test lowering of moore.class.null and moore.class_handle_cmp operations

// The class must be declared at the top level
moore.class.classdecl @MyClass {
  moore.class.propertydecl @prop : !moore.i32
}

/// Check that class.null lowers to llvm.mlir.zero

// CHECK-LABEL: func.func private @test_class_null
// CHECK:   %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   return %[[NULL]] : !llvm.ptr
// CHECK-NOT: moore.class.null

func.func private @test_class_null() -> !moore.class<@MyClass> {
  %null = moore.class.null : !moore.class<@MyClass>
  return %null : !moore.class<@MyClass>
}

/// Check that class_handle_cmp eq lowers to llvm.icmp eq

// CHECK-LABEL: func.func private @test_class_cmp_eq
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr)
// CHECK:   %[[CMP:.*]] = llvm.icmp "eq" %arg0, %arg1 : !llvm.ptr
// CHECK:   return %[[CMP]] : i1
// CHECK-NOT: moore.class_handle_cmp

func.func private @test_class_cmp_eq(%a: !moore.class<@MyClass>, %b: !moore.class<@MyClass>) -> !moore.i1 {
  %result = moore.class_handle_cmp eq %a, %b : !moore.class<@MyClass> -> i1
  return %result : !moore.i1
}

/// Check that class_handle_cmp ne lowers to llvm.icmp ne

// CHECK-LABEL: func.func private @test_class_cmp_ne
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr)
// CHECK:   %[[CMP:.*]] = llvm.icmp "ne" %arg0, %arg1 : !llvm.ptr
// CHECK:   return %[[CMP]] : i1
// CHECK-NOT: moore.class_handle_cmp

func.func private @test_class_cmp_ne(%a: !moore.class<@MyClass>, %b: !moore.class<@MyClass>) -> !moore.i1 {
  %result = moore.class_handle_cmp ne %a, %b : !moore.class<@MyClass> -> i1
  return %result : !moore.i1
}

/// Check a combination of null and comparison

// CHECK-LABEL: func.func private @test_null_cmp
// CHECK-SAME: (%arg0: !llvm.ptr)
// CHECK:   %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   %[[CMP:.*]] = llvm.icmp "eq" %arg0, %[[NULL]] : !llvm.ptr
// CHECK:   return %[[CMP]] : i1
// CHECK-NOT: moore.class.null
// CHECK-NOT: moore.class_handle_cmp

func.func private @test_null_cmp(%obj: !moore.class<@MyClass>) -> !moore.i1 {
  %null = moore.class.null : !moore.class<@MyClass>
  %is_null = moore.class_handle_cmp eq %obj, %null : !moore.class<@MyClass> -> i1
  return %is_null : !moore.i1
}
