// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// This test validates that class field indexing correctly accounts for:
// 1. System fields in root classes (typeId at [0], vtablePtr at [1])
// 2. Base class embedding in derived classes
// 3. Proper LLVM GEP index generation with leading 0 for pointer dereference

//===----------------------------------------------------------------------===//
// Root class field indexing
//===----------------------------------------------------------------------===//

// Root class layout: { i32 typeId, ptr vtablePtr, ...user_fields... }
// So user field N should be accessed at LLVM index N+2
moore.class.classdecl @RootClass {
  moore.class.propertydecl @field0 : !moore.i32  // LLVM index 2
  moore.class.propertydecl @field1 : !moore.i64  // LLVM index 3
  moore.class.propertydecl @field2 : !moore.i8   // LLVM index 4
}

// CHECK-LABEL: func.func @test_root_field0
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"RootClass"
func.func @test_root_field0(%obj: !moore.class<@RootClass>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %obj[@field0] : <@RootClass> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

// CHECK-LABEL: func.func @test_root_field1
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 3] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"RootClass"
func.func @test_root_field1(%obj: !moore.class<@RootClass>) -> !moore.ref<i64> {
  %ref = moore.class.property_ref %obj[@field1] : <@RootClass> -> !moore.ref<i64>
  return %ref : !moore.ref<i64>
}

// CHECK-LABEL: func.func @test_root_field2
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"RootClass"
func.func @test_root_field2(%obj: !moore.class<@RootClass>) -> !moore.ref<i8> {
  %ref = moore.class.property_ref %obj[@field2] : <@RootClass> -> !moore.ref<i8>
  return %ref : !moore.ref<i8>
}

//===----------------------------------------------------------------------===//
// Derived class accessing own fields
//===----------------------------------------------------------------------===//

// Derived class layout: { BaseStruct, ...derived_fields... }
// So derived field N should be accessed at LLVM index N+1 (after base at [0])
moore.class.classdecl @BaseForDerived {
  moore.class.propertydecl @base_x : !moore.i32
}

moore.class.classdecl @DerivedOwn extends @BaseForDerived {
  moore.class.propertydecl @derived_y : !moore.i64  // LLVM index 1
  moore.class.propertydecl @derived_z : !moore.i16  // LLVM index 2
}

// CHECK-LABEL: func.func @test_derived_own_field0
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"DerivedOwn"
func.func @test_derived_own_field0(%obj: !moore.class<@DerivedOwn>) -> !moore.ref<i64> {
  %ref = moore.class.property_ref %obj[@derived_y] : <@DerivedOwn> -> !moore.ref<i64>
  return %ref : !moore.ref<i64>
}

// CHECK-LABEL: func.func @test_derived_own_field1
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"DerivedOwn"
func.func @test_derived_own_field1(%obj: !moore.class<@DerivedOwn>) -> !moore.ref<i16> {
  %ref = moore.class.property_ref %obj[@derived_z] : <@DerivedOwn> -> !moore.ref<i16>
  return %ref : !moore.ref<i16>
}

//===----------------------------------------------------------------------===//
// Derived class accessing inherited fields
//===----------------------------------------------------------------------===//

// When accessing inherited field, path goes: [0 (ptr deref), 0 (into base), field_index]
// CHECK-LABEL: func.func @test_derived_inherited_field
// Inherited field base_x is at base[0].field[2] (base is at index 0 in derived, field at index 2 after typeId and vtablePtr in base)
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"DerivedOwn"
func.func @test_derived_inherited_field(%obj: !moore.class<@DerivedOwn>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %obj[@base_x] : <@DerivedOwn> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

//===----------------------------------------------------------------------===//
// Multi-level inheritance
//===----------------------------------------------------------------------===//

moore.class.classdecl @Level0 {
  moore.class.propertydecl @level0_field : !moore.i8
}

moore.class.classdecl @Level1 extends @Level0 {
  moore.class.propertydecl @level1_field : !moore.i16
}

moore.class.classdecl @Level2 extends @Level1 {
  moore.class.propertydecl @level2_field : !moore.i32
}

// Accessing Level2's own field
// CHECK-LABEL: func.func @test_multi_level_own
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
func.func @test_multi_level_own(%obj: !moore.class<@Level2>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %obj[@level2_field] : <@Level2> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

// Accessing Level1's field from Level2
// Path: [0 (ptr deref), 0 (into Level1 base), 1 (level1_field after Level0 base)]
// CHECK-LABEL: func.func @test_multi_level_parent
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
func.func @test_multi_level_parent(%obj: !moore.class<@Level2>) -> !moore.ref<i16> {
  %ref = moore.class.property_ref %obj[@level1_field] : <@Level2> -> !moore.ref<i16>
  return %ref : !moore.ref<i16>
}

// Accessing Level0's field from Level2
// Path: [0 (ptr deref), 0 (into Level1), 0 (into Level0), 2 (level0_field after typeId and vtablePtr)]
// CHECK-LABEL: func.func @test_multi_level_grandparent
// CHECK: llvm.getelementptr %arg0[{{%.+}}, 0, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
func.func @test_multi_level_grandparent(%obj: !moore.class<@Level2>) -> !moore.ref<i8> {
  %ref = moore.class.property_ref %obj[@level0_field] : <@Level2> -> !moore.ref<i8>
  return %ref : !moore.ref<i8>
}
