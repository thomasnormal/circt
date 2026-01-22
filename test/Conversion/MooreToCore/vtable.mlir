// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that vtables and vtable_entry ops are converted to LLVM globals and
// dynamic dispatch code during lowering.

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

// CHECK: llvm.mlir.global internal @"testClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32} : !llvm.array<2 x ptr>

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

// Test vtable.load_method lowering to dynamic dispatch.
// The lowering performs runtime vtable lookup:
// 1. Load vtable pointer from object (at offset 1 after typeId)
// 2. GEP into vtable array at method's index
// 3. Load function pointer from vtable

// CHECK-LABEL: func.func @test_vtable_load_method
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr) -> ((!llvm.ptr) -> ())
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 0]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]

func.func @test_vtable_load_method(%obj: !moore.class<@testClass>) -> ((!moore.class<@testClass>) -> ()) {
  %fptr = moore.vtable.load_method %obj : @subroutine of <@testClass> -> (!moore.class<@testClass>) -> ()
  return %fptr : (!moore.class<@testClass>) -> ()
}

// Test vtable.load_method with abstract base class.
// The method index is determined by the class's methodToVtableIndex map.

// CHECK-LABEL: func.func @test_vtable_load_method_nested
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr) -> ((!llvm.ptr) -> ())
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][0, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 0]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]

func.func @test_vtable_load_method_nested(%obj: !moore.class<@virtualFunctionClass>) -> ((!moore.class<@virtualFunctionClass>) -> ()) {
  // virtualFunctionClass has no top-level vtable, but the method is looked up
  // via the class's methodToVtableIndex map which is populated from method declarations.
  %fptr = moore.vtable.load_method %obj : @subroutine of <@virtualFunctionClass> -> (!moore.class<@virtualFunctionClass>) -> ()
  return %fptr : (!moore.class<@virtualFunctionClass>) -> ()
}
