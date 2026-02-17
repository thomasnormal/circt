// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test edge cases for class operations in Moore dialect

//===----------------------------------------------------------------------===//
// Class Hierarchy Declarations
//===----------------------------------------------------------------------===//

// Base class with properties
moore.class.classdecl @Base {
  moore.class.propertydecl @baseVal : !moore.i32
}

// First level derived class
moore.class.classdecl @Level1 extends @Base {
  moore.class.propertydecl @level1Val : !moore.i64
}

// Second level derived class
moore.class.classdecl @Level2 extends @Level1 {
  moore.class.propertydecl @level2Val : !moore.i16
}

// Third level derived class for chain testing
moore.class.classdecl @Level3 extends @Level2 {
  moore.class.propertydecl @level3Val : !moore.i8
}

//===----------------------------------------------------------------------===//
// Upcast Chain Tests (A -> B -> C)
//===----------------------------------------------------------------------===//

/// Test multi-level upcast chain: Level3 -> Level2 -> Level1 -> Base

// CHECK-LABEL: func.func private @test_upcast_chain
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK:   return %arg0 : !llvm.ptr
// CHECK-NOT: moore.class.upcast

func.func private @test_upcast_chain(%level3: !moore.class<@Level3>) -> !moore.class<@Base> {
  %level2 = moore.class.upcast %level3 : <@Level3> to <@Level2>
  %level1 = moore.class.upcast %level2 : <@Level2> to <@Level1>
  %base = moore.class.upcast %level1 : <@Level1> to <@Base>
  return %base : !moore.class<@Base>
}

/// Test direct multi-level upcast: Level3 -> Base (skip intermediates)

// CHECK-LABEL: func.func private @test_upcast_direct
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK:   return %arg0 : !llvm.ptr
// CHECK-NOT: moore.class.upcast

func.func private @test_upcast_direct(%level3: !moore.class<@Level3>) -> !moore.class<@Base> {
  %base = moore.class.upcast %level3 : <@Level3> to <@Base>
  return %base : !moore.class<@Base>
}

//===----------------------------------------------------------------------===//
// Multiple Class Instances in Same Function
//===----------------------------------------------------------------------===//

// Classes for multiple instance tests
moore.class.classdecl @ClassA {
  moore.class.propertydecl @valA : !moore.i32
}

moore.class.classdecl @ClassB {
  moore.class.propertydecl @valB : !moore.i64
}

/// Test creating multiple class instances of different types
/// ClassA: type_id(4) + vtable_ptr(8) + valA(4) = 16 bytes
/// ClassB: type_id(4) + vtable_ptr(8) + valB(8) = 20 bytes

// CHECK-LABEL: func.func private @test_multiple_new
// CHECK:   llvm.mlir.constant(16 : i64) : i64
// CHECK:   llvm.call @malloc
// CHECK:   llvm.mlir.constant(20 : i64) : i64
// CHECK:   llvm.call @malloc
// CHECK:   return
// CHECK-NOT: moore.class.new

func.func private @test_multiple_new() {
  %a = moore.class.new : <@ClassA>
  %b = moore.class.new : <@ClassB>
  return
}

//===----------------------------------------------------------------------===//
// Property Access Tests
//===----------------------------------------------------------------------===//

/// Test accessing own property at each level

// CHECK-LABEL: func.func private @test_own_property_level1
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i64> {
// For derived classes, own properties are at index 1 (after base at index 0).
// The leading 0 is for pointer dereference.
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[{{%.+}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level1"
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i64>
// CHECK:   return [[CONV]] : !llhd.ref<i64>
// CHECK-NOT: moore.class.property_ref

func.func private @test_own_property_level1(%obj: !moore.class<@Level1>) -> !moore.ref<i64> {
  %ref = moore.class.property_ref %obj[@level1Val] : <@Level1> -> !moore.ref<i64>
  return %ref : !moore.ref<i64>
}

// CHECK-LABEL: func.func private @test_own_property_level2
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i16> {
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[{{%.+}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i16>
// CHECK:   return [[CONV]] : !llhd.ref<i16>
// CHECK-NOT: moore.class.property_ref

func.func private @test_own_property_level2(%obj: !moore.class<@Level2>) -> !moore.ref<i16> {
  %ref = moore.class.property_ref %obj[@level2Val] : <@Level2> -> !moore.ref<i16>
  return %ref : !moore.ref<i16>
}

// CHECK-LABEL: func.func private @test_own_property_level3
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i8> {
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[{{%.+}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level3"
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i8>
// CHECK:   return [[CONV]] : !llhd.ref<i8>
// CHECK-NOT: moore.class.property_ref

func.func private @test_own_property_level3(%obj: !moore.class<@Level3>) -> !moore.ref<i8> {
  %ref = moore.class.property_ref %obj[@level3Val] : <@Level3> -> !moore.ref<i8>
  return %ref : !moore.ref<i8>
}

/// Test accessing base property via upcast then property_ref

// CHECK-LABEL: func.func private @test_upcast_then_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i32> {
// For root class Base, field baseVal is at index 2 (after typeId[0], vtablePtr[1]).
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[{{%.+}}, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Base", (i32, ptr, i32)>
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return [[CONV]] : !llhd.ref<i32>
// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.upcast

func.func private @test_upcast_then_property(%obj: !moore.class<@Level3>) -> !moore.ref<i32> {
  // Upcast to base, then access base property
  %base = moore.class.upcast %obj : <@Level3> to <@Base>
  %ref = moore.class.property_ref %base[@baseVal] : <@Base> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

//===----------------------------------------------------------------------===//
// Handle Comparison Edge Cases
//===----------------------------------------------------------------------===//

/// Test comparing handles from different hierarchy levels after upcast

// CHECK-LABEL: func.func private @test_cmp_after_upcast
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i1 {
// CHECK:   %[[CMP:.*]] = llvm.icmp "eq" %arg0, %arg1 : !llvm.ptr
// CHECK:   return %[[CMP]] : i1
// CHECK-NOT: moore.class_handle_cmp
// CHECK-NOT: moore.class.upcast

func.func private @test_cmp_after_upcast(%derived: !moore.class<@Level2>, %base: !moore.class<@Base>) -> !moore.i1 {
  %upcasted = moore.class.upcast %derived : <@Level2> to <@Base>
  %result = moore.class_handle_cmp eq %upcasted, %base : !moore.class<@Base> -> i1
  return %result : !moore.i1
}

/// Test comparing handle with null after new

// CHECK-LABEL: func.func private @test_new_not_null
// CHECK:   [[PTR:%.*]] = llvm.call @malloc
// CHECK:   [[NULL:%.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   [[CMP:%.*]] = llvm.icmp "ne" [[PTR]], [[NULL]] : !llvm.ptr
// CHECK:   return [[CMP]] : i1
// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.null
// CHECK-NOT: moore.class_handle_cmp

func.func private @test_new_not_null() -> !moore.i1 {
  %obj = moore.class.new : <@ClassA>
  %null = moore.class.null : !moore.class<@ClassA>
  %is_not_null = moore.class_handle_cmp ne %obj, %null : !moore.class<@ClassA> -> i1
  return %is_not_null : !moore.i1
}

/// Test null equality (null == null should be true)
/// The constant folding optimization simplifies this to just return true.

// CHECK-LABEL: func.func private @test_null_equals_null
// CHECK:   [[NULL1:%.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   [[NULL2:%.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   [[CMP:%.*]] = llvm.icmp "eq" [[NULL1]], [[NULL2]] : !llvm.ptr
// CHECK:   return [[CMP]] : i1
// CHECK-NOT: moore.class.null
// CHECK-NOT: moore.class_handle_cmp

func.func private @test_null_equals_null() -> !moore.i1 {
  %null1 = moore.class.null : !moore.class<@ClassA>
  %null2 = moore.class.null : !moore.class<@ClassA>
  %result = moore.class_handle_cmp eq %null1, %null2 : !moore.class<@ClassA> -> i1
  return %result : !moore.i1
}

//===----------------------------------------------------------------------===//
// Combined Operations
//===----------------------------------------------------------------------===//

/// Test full workflow: new -> property access -> upcast -> compare

// CHECK-LABEL: func.func private @test_full_workflow
// CHECK:   [[PTR:%.*]] = llvm.call @malloc
// CHECK:   llvm.getelementptr [[PTR]]
// CHECK:   [[NULL:%.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   [[CMP:%.*]] = llvm.icmp "ne" [[PTR]], [[NULL]] : !llvm.ptr
// CHECK:   return [[CMP]] : i1

func.func private @test_full_workflow() -> !moore.i1 {
  // Create a Level1 object
  %obj = moore.class.new : <@Level1>
  // Access its own property
  %propRef = moore.class.property_ref %obj[@level1Val] : <@Level1> -> !moore.ref<i64>
  // Upcast to base
  %base = moore.class.upcast %obj : <@Level1> to <@Base>
  // Compare with null
  %null = moore.class.null : !moore.class<@Base>
  %is_valid = moore.class_handle_cmp ne %base, %null : !moore.class<@Base> -> i1
  return %is_valid : !moore.i1
}

//===----------------------------------------------------------------------===//
// Memory Size Calculations for Complex Hierarchies
//===----------------------------------------------------------------------===//

/// Test that Level3 allocation includes all inherited properties

// CHECK-LABEL: func.func private @test_level3_size
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(27 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return
// CHECK-NOT: moore.class.new

func.func private @test_level3_size() {
  // Level3 struct is nested: Level3(Level2(Level1(Base(i32, ptr, i32), i64), i16), i8)
  // Base: i32 + ptr(8) + i32 = 16 bytes
  // Level1: Base(16) + i64 = 24 bytes
  // Level2: Level1(24) + i16 = 26 bytes
  // Level3: Level2(26) + i8 = 27 bytes (packed size)
  %obj = moore.class.new : <@Level3>
  return
}
