// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that vtables and vtable_entry ops are converted to LLVM globals and
// dynamic dispatch code during lowering.

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

// Check that both vtables have circt.vtable_entries (verifies the fix for
// the bug where ClassNewOpConversion would create a placeholder global
// without vtable entries before VTableOpConversion runs).
// CHECK-DAG: llvm.mlir.global internal @"SimpleClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = {{.*}}} : !llvm.array<1 x ptr>
// CHECK-DAG: llvm.mlir.global internal @"testClass::__vtable__"(#llvm.zero) {addr_space = 0 : i32, circt.vtable_entries = {{.*}}} : !llvm.array<2 x ptr>

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
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][{{%.+}}, 1]
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
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][{{%.+}}, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 0]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]

func.func @test_vtable_load_method_nested(%obj: !moore.class<@virtualFunctionClass>) -> ((!moore.class<@virtualFunctionClass>) -> ()) {
  // virtualFunctionClass has no top-level vtable, but the method is looked up
  // via the class's methodToVtableIndex map which is populated from method declarations.
  %fptr = moore.vtable.load_method %obj : @subroutine of <@virtualFunctionClass> -> (!moore.class<@virtualFunctionClass>) -> ()
  return %fptr : (!moore.class<@virtualFunctionClass>) -> ()
}

// Test vtable entries are populated even when ClassNewOpConversion creates
// the vtable global first (before VTableOpConversion runs).
// This tests the fix for the bug where the vtable global would be created
// without circt.vtable_entries attribute.

moore.class.classdecl @SimpleClass {
  moore.class.propertydecl @value : !moore.i32
  moore.class.methoddecl @get_value -> @"SimpleClass::get_value" : (!moore.class<@SimpleClass>) -> !moore.i32
}

moore.vtable @SimpleClass::@vtable {
  moore.vtable_entry @get_value -> @"SimpleClass::get_value"
}

func.func private @"SimpleClass::get_value"(%this: !moore.class<@SimpleClass>) -> !moore.i32 {
  %ref = moore.class.property_ref %this[@value] : <@SimpleClass> -> !moore.ref<i32>
  %val = moore.read %ref : <i32>
  return %val : !moore.i32
}

// Test that class.new before vtable still gets vtable entries populated.
// The order is intentional: new appears before vtable in IR.

// CHECK-LABEL: func.func @test_new_before_vtable
// CHECK:         llvm.call @malloc
func.func @test_new_before_vtable() -> !moore.class<@SimpleClass> {
  %obj = moore.class.new : <@SimpleClass>
  return %obj : !moore.class<@SimpleClass>
}
