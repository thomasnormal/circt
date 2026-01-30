// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

/// Check that a classdecl gets noop'd and handles are lowered to !llvm.ptr

// CHECK-LABEL:   func.func @ClassType(%arg0: !llvm.ptr) {
// CHECK:    return
// CHECK:  }
// CHECK-NOT: moore.class.classdecl
// CHECK-NOT: moore.class<@PropertyCombo>

moore.class.classdecl @PropertyCombo {
  moore.class.propertydecl @pubAutoI32   : !moore.i32
  moore.class.propertydecl @protStatL18  : !moore.l18
  moore.class.propertydecl @localAutoI32 : !moore.i32
}

func.func @ClassType(%arg0: !moore.class<@PropertyCombo>) {
  return
}

/// Check that new lowers to malloc

// malloc should be declared in the LLVM dialect.
// Class C struct is: (type_id(i32), vtablePtr(ptr), a(i32), b(struct<i32,i32>), c(struct<i32,i32>))
// Size: 4 + 8 + 4 + 8 + 8 = 32 bytes
// CHECK-LABEL: func.func private @test_new2
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

// Allocate a new instance; should lower to llvm.call @malloc(i64).
func.func private @test_new2() {
  %h = moore.class.new : <@C>
  return
}
// Minimal class so the identified struct has a concrete body.
// l32 is 4-value logic which is lowered to struct<(i32, i32)>
moore.class.classdecl @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that new lowers to malloc with inheritance without shadowing
/// D struct is: (C(32), d(struct<i32,i32>), e(struct<i64,i64>), f(i16))
/// Size: 32 + 8 + 16 + 2 = 58 bytes

// CHECK-LABEL: func.func private @test_new3
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(58 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_new3() {
  %h = moore.class.new : <@D>
  return
}
moore.class.classdecl @D extends @C {
  moore.class.propertydecl @d : !moore.l32
  moore.class.propertydecl @e : !moore.l64
  moore.class.propertydecl @f : !moore.i16
}

/// Check that new lowers to malloc with inheritance & shadowing
/// E struct is: (C(32), a(i32), b(struct<i32,i32>), c(struct<i32,i32>))
/// Size: 32 + 4 + 8 + 8 = 52 bytes

// CHECK-LABEL: func.func private @test_new4
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(52 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_new4() {
  %h = moore.class.new : <@E>
  return
}
moore.class.classdecl @E extends @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that upcast lowers to no-op

// CHECK-LABEL: func.func private @test_new5
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK:   return %arg0 : !llvm.ptr

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.upcast
// CHECK-NOT: moore.class.classdecl

func.func private @test_new5(%arg0: !moore.class<@F>) -> !moore.class<@C> {
  %upcast = moore.class.upcast %arg0 : <@F> to <@C>
  return %upcast : !moore.class<@C>
}
moore.class.classdecl @F extends @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that property_ref lowers to GEP

// CHECK-LABEL: func.func private @test_new6
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i32> {
// G extends C (derived), so layout is {base_C, d, e, f}. Accessing d at index 1.
// First index 0 dereferences the pointer, second index 1 accesses field d.
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"G"
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return [[CONV]] : !llhd.ref<i32>

// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.classdecl

func.func private @test_new6(%arg0: !moore.class<@G>) -> !moore.ref<i32> {
  %gep = moore.class.property_ref %arg0[@d] : <@G> -> !moore.ref<i32>
  return %gep : !moore.ref<i32>
}
moore.class.classdecl @G extends @C {
  moore.class.propertydecl @d : !moore.i32
  moore.class.propertydecl @e : !moore.l32
  moore.class.propertydecl @f : !moore.l32
}

/// Check that dynamic cast lowers to runtime type check

// CHECK-LABEL: func.func private @test_dyn_cast
// CHECK-SAME: (%arg0: !llvm.ptr) -> (!llvm.ptr, i1) {
// CHECK-DAG:   [[ZERO:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:   [[RTTI:%.+]] = llvm.load %arg0 : !llvm.ptr -> i32
// CHECK:   [[CALL:%.+]] = llvm.call @__moore_dyn_cast_check({{.*}}) : (i32, i32, i32) -> i1
// CHECK:   [[NOTNULL:%.+]] = llvm.icmp "ne" %arg0, [[ZERO]] : !llvm.ptr
// CHECK:   [[SUCCESS:%.+]] = llvm.and [[NOTNULL]], [[CALL]] : i1
// CHECK:   return %arg0, [[SUCCESS]] : !llvm.ptr, i1

// CHECK-NOT: moore.class.dyn_cast
// CHECK-NOT: moore.class.classdecl

moore.class.classdecl @BaseClass {
  moore.class.propertydecl @x : !moore.i32
}

moore.class.classdecl @DerivedClass extends @BaseClass {
  moore.class.propertydecl @y : !moore.i32
}

func.func private @test_dyn_cast(%arg0: !moore.class<@BaseClass>) -> (!moore.class<@DerivedClass>, i1) {
  %result, %success = moore.class.dyn_cast %arg0 : <@BaseClass> to <@DerivedClass>
  return %result, %success : !moore.class<@DerivedClass>, i1
}

/// Check that class with struct property computes size correctly
/// (regression test for DataLayout crash with hw.struct types)

// CHECK-LABEL: func.func private @test_struct_property
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_struct_property() {
  %h = moore.class.new : <@ClassWithStruct>
  return
}
// Class with a struct property - must be converted to pure LLVM types
// for DataLayout::getTypeSize() to work correctly.
// Struct is: (type_id(i32), vtablePtr(ptr), x(i32), data(struct<i32,i32>))
// Size: 4 + 8 + 4 + 8 = 24 bytes
moore.class.classdecl @ClassWithStruct {
  moore.class.propertydecl @x : !moore.i32
  moore.class.propertydecl @data : !moore.ustruct<{field1: i32, field2: i32}>
}

/// Check that class with time property computes size correctly
/// (regression test for llhd.time DataLayout crash - time is i64)

// CHECK-LABEL: func.func private @test_class_with_time
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_class_with_time() {
  %h = moore.class.new : <@ClassWithTime>
  return
}
// Class with a time property - llhd.time requires special handling because
// DataLayout doesn't support it. Time is {i64 realTime, i32 delta, i32 epsilon} = 16 bytes.
// Struct is: (type_id(i32), vtablePtr(ptr), x(i32), timestamp(time)) = 4 + 8 + 4 + 16 = 32 bytes total.
moore.class.classdecl @ClassWithTime {
  moore.class.propertydecl @x : !moore.i32
  moore.class.propertydecl @timestamp : !moore.time
}

/// Check that class with multiple time properties computes size correctly

// CHECK-LABEL: func.func private @test_class_with_multiple_times
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_class_with_multiple_times() {
  %h = moore.class.new : <@ClassWithMultipleTimes>
  return
}
// Class with multiple time properties: type_id(4) + vtablePtr(8) + start(i64=8) + end(i64=8) + count(4) = 32 bytes
moore.class.classdecl @ClassWithMultipleTimes {
  moore.class.propertydecl @start_time : !moore.time
  moore.class.propertydecl @end_time : !moore.time
  moore.class.propertydecl @count : !moore.i32
}

/// Check that class with nested struct containing time computes size correctly

// CHECK-LABEL: func.func private @test_class_with_time_struct
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(28 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_class_with_time_struct() {
  %h = moore.class.new : <@ClassWithTimeStruct>
  return
}
// Class with a struct property containing time:
// type_id(4) + vtablePtr(8) + id(4) + access_record{timestamp(i64=8) + count(4)} = 28 bytes
moore.class.classdecl @ClassWithTimeStruct {
  moore.class.propertydecl @id : !moore.i32
  moore.class.propertydecl @access_record : !moore.ustruct<{timestamp: time, count: i32}>
}
