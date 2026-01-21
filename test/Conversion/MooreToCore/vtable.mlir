// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that vtables and vtable_entry ops are erased during lowering.

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

moore.class.classdecl @virtualFunctionClass {
  moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>) -> ()
}
moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
  moore.class.methoddecl @testSubroutine : (!moore.class<@realFunctionClass>) -> ()
}
moore.class.classdecl @testClass implements [@realFunctionClass] {
  moore.class.methoddecl @subroutine -> @"testClass::subroutine" : (!moore.class<@testClass>) -> ()
  moore.class.methoddecl @testSubroutine -> @"testClass::testSubroutine" : (!moore.class<@testClass>) -> ()
}

moore.vtable @testClass::@vtable {
  moore.vtable @realFunctionClass::@vtable {
    moore.vtable @virtualFunctionClass::@vtable {
      moore.vtable_entry @subroutine -> @"testClass::subroutine"
    }
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  moore.vtable_entry @subroutine -> @"testClass::subroutine"
  moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
}

func.func private @"testClass::subroutine"(%arg0: !moore.class<@testClass>) {
  return
}
func.func private @"testClass::testSubroutine"(%arg0: !moore.class<@testClass>) {
  return
}

// Test vtable.load_method lowering to func.constant

// CHECK-LABEL: func.func @test_vtable_load_method
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr) -> ((!llvm.ptr) -> ())
// CHECK:         %[[FPTR:.*]] = constant @"testClass::subroutine" : (!llvm.ptr) -> ()
// CHECK:         return %[[FPTR]]

func.func @test_vtable_load_method(%obj: !moore.class<@testClass>) -> ((!moore.class<@testClass>) -> ()) {
  %fptr = moore.vtable.load_method %obj : @subroutine of <@testClass> -> (!moore.class<@testClass>) -> ()
  return %fptr : (!moore.class<@testClass>) -> ()
}

// Test vtable.load_method with nested vtable lookup (abstract base class case).
// When the static type is an abstract class without its own top-level vtable,
// we should find the vtable entry in the nested vtable inside a derived class vtable.

// CHECK-LABEL: func.func @test_vtable_load_method_nested
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr) -> ((!llvm.ptr) -> ())
// CHECK:         %[[FPTR:.*]] = constant @"testClass::subroutine" : (!llvm.ptr) -> ()
// CHECK:         return %[[FPTR]]

func.func @test_vtable_load_method_nested(%obj: !moore.class<@virtualFunctionClass>) -> ((!moore.class<@virtualFunctionClass>) -> ()) {
  // virtualFunctionClass has no top-level vtable, but testClass::@vtable contains
  // a nested @virtualFunctionClass::@vtable with the method entry.
  %fptr = moore.vtable.load_method %obj : @subroutine of <@virtualFunctionClass> -> (!moore.class<@virtualFunctionClass>) -> ()
  return %fptr : (!moore.class<@virtualFunctionClass>) -> ()
}
