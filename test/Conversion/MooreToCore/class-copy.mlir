// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test lowering of moore.class.copy operation

// The class must be declared at the top level.
// Class struct is: (type_id(i32), vtablePtr(ptr), prop(i32))
// Size: 4 + 8 + 4 = 16 bytes
moore.class.classdecl @MyClass {
  moore.class.propertydecl @prop : !moore.i32
}

/// Check that class.copy lowers to malloc + memcpy

// CHECK-LABEL: func.func private @test_class_copy
// CHECK-SAME: (%arg0: !llvm.ptr)
// CHECK:   %[[SIZE:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:   %[[NEWPTR:.*]] = llvm.call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   "llvm.intr.memcpy"(%[[NEWPTR]], %arg0, %[[SIZE]]) <{isVolatile = false}>
// CHECK:   return %[[NEWPTR]] : !llvm.ptr
// CHECK-NOT: moore.class.copy

func.func private @test_class_copy(%source: !moore.class<@MyClass>) -> !moore.class<@MyClass> {
  %copy = moore.class.copy %source : !moore.class<@MyClass>
  return %copy : !moore.class<@MyClass>
}
