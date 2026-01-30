// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// This test validates class property access lowering to LLVM GEP operations

//===----------------------------------------------------------------------===//
// Simple class property access
//===----------------------------------------------------------------------===//

moore.class.classdecl @SimpleClass {
  moore.class.propertydecl @field1 : !moore.i32
  moore.class.propertydecl @field2 : !moore.l64
}

// CHECK-LABEL: func.func @test_simple_property_ref
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i32>
// The first index 0 dereferences the pointer, second index 2 accesses field1 (after typeId[0] and vtablePtr[1])
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SimpleClass"
// CHECK:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return [[CAST]] : !llhd.ref<i32>
func.func @test_simple_property_ref(%obj: !moore.class<@SimpleClass>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %obj[@field1] : <@SimpleClass> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

// CHECK-LABEL: func.func @test_second_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<!hw.struct<value: i64, unknown: i64>>
// The first index 0 dereferences the pointer, second index 3 accesses field2 (after typeId[0], vtablePtr[1], field1[2])
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SimpleClass"
// CHECK:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<!hw.struct<value: i64, unknown: i64>>
// CHECK:   return [[CAST]] : !llhd.ref<!hw.struct<value: i64, unknown: i64>>
func.func @test_second_property(%obj: !moore.class<@SimpleClass>) -> !moore.ref<l64> {
  %ref = moore.class.property_ref %obj[@field2] : <@SimpleClass> -> !moore.ref<l64>
  return %ref : !moore.ref<l64>
}

//===----------------------------------------------------------------------===//
// Inherited class property access
//===----------------------------------------------------------------------===//

moore.class.classdecl @BaseClass {
  moore.class.propertydecl @base_field : !moore.i16
}

moore.class.classdecl @DerivedClass extends @BaseClass {
  moore.class.propertydecl @derived_field : !moore.i32
}

// CHECK-LABEL: func.func @test_derived_class_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i32>
// For derived classes: first index 0 dereferences the pointer, second index 1 accesses derived_field (after base[0])
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"DerivedClass"
// CHECK:   [[CAST:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return [[CAST]] : !llhd.ref<i32>
func.func @test_derived_class_property(%obj: !moore.class<@DerivedClass>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %obj[@derived_field] : <@DerivedClass> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}


//===----------------------------------------------------------------------===//
// Property access with multiple fields
//===----------------------------------------------------------------------===//

moore.class.classdecl @MultiFieldClass {
  moore.class.propertydecl @a : !moore.i8
  moore.class.propertydecl @b : !moore.i16
  moore.class.propertydecl @c : !moore.i32
  moore.class.propertydecl @d : !moore.l64
}

// CHECK-LABEL: func.func @test_multi_field_access
// CHECK:   llvm.getelementptr
// CHECK:   llvm.getelementptr
// CHECK:   llvm.getelementptr
// CHECK:   return
func.func @test_multi_field_access(%obj: !moore.class<@MultiFieldClass>) -> (!moore.ref<i8>, !moore.ref<i32>, !moore.ref<l64>) {
  %a = moore.class.property_ref %obj[@a] : <@MultiFieldClass> -> !moore.ref<i8>
  %c = moore.class.property_ref %obj[@c] : <@MultiFieldClass> -> !moore.ref<i32>
  %d = moore.class.property_ref %obj[@d] : <@MultiFieldClass> -> !moore.ref<l64>
  return %a, %c, %d : !moore.ref<i8>, !moore.ref<i32>, !moore.ref<l64>
}
