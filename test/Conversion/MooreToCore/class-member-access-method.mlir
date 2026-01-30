// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// This test validates class member variable access from within class methods.
// The key issue is that when class methods are converted to LLVM, the 'this'
// pointer (block argument) gets remapped but operations inside methods may
// still reference the OLD/invalidated block arguments. The fix ensures
// getConvertedOperand() properly handles remapped block arguments.

//===----------------------------------------------------------------------===//
// Class declarations
//===----------------------------------------------------------------------===//

moore.class.classdecl @SimpleCounter {
  moore.class.propertydecl @count : !moore.i32
  moore.class.propertydecl @enabled : !moore.i1
}

moore.class.classdecl @DataContainer {
  moore.class.propertydecl @value1 : !moore.i32
  moore.class.propertydecl @value2 : !moore.i64
  moore.class.propertydecl @flag : !moore.l1
}

moore.class.classdecl @BaseCounter {
  moore.class.propertydecl @base_count : !moore.i16
}

moore.class.classdecl @DerivedCounter extends @BaseCounter {
  moore.class.propertydecl @derived_count : !moore.i32
  moore.class.propertydecl @multiplier : !moore.i8
}

//===----------------------------------------------------------------------===//
// Simple class member read from method
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_simple_member_read
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr) -> i32
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SimpleCounter"
// CHECK:   %[[RESULT:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
// CHECK:   return %[[RESULT]] : i32
func.func @test_simple_member_read(%this: !moore.class<@SimpleCounter>) -> !moore.i32 {
  %ref = moore.class.property_ref %this[@count] : <@SimpleCounter> -> !moore.ref<i32>
  %val = moore.read %ref : !moore.ref<i32>
  return %val : !moore.i32
}

//===----------------------------------------------------------------------===//
// Multiple members accessed in single method
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_multiple_member_access
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr) -> (i32, i64)
// CHECK:   llvm.getelementptr %[[THIS]][0, 2]
// CHECK:   llvm.getelementptr %[[THIS]][0, 3]
// CHECK:   return
func.func @test_multiple_member_access(%this: !moore.class<@DataContainer>) -> (!moore.i32, !moore.i64) {
  %ref1 = moore.class.property_ref %this[@value1] : <@DataContainer> -> !moore.ref<i32>
  %val1 = moore.read %ref1 : !moore.ref<i32>
  %ref2 = moore.class.property_ref %this[@value2] : <@DataContainer> -> !moore.ref<i64>
  %val2 = moore.read %ref2 : !moore.ref<i64>
  return %val1, %val2 : !moore.i32, !moore.i64
}

//===----------------------------------------------------------------------===//
// Member write from method
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_member_write
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr, %[[VALUE:arg1]]: i32)
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SimpleCounter"
// CHECK:   return
func.func @test_member_write(%this: !moore.class<@SimpleCounter>, %value: !moore.i32) {
  %ref = moore.class.property_ref %this[@count] : <@SimpleCounter> -> !moore.ref<i32>
  moore.blocking_assign %ref, %value : i32
  return
}

//===----------------------------------------------------------------------===//
// Derived class member access
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_derived_member_access
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr) -> (i32, i8)
// CHECK:   llvm.getelementptr %[[THIS]][0, 1]
// CHECK:   llvm.getelementptr %[[THIS]][0, 2]
// CHECK:   return
func.func @test_derived_member_access(%this: !moore.class<@DerivedCounter>) -> (!moore.i32, !moore.i8) {
  %ref1 = moore.class.property_ref %this[@derived_count] : <@DerivedCounter> -> !moore.ref<i32>
  %val1 = moore.read %ref1 : !moore.ref<i32>
  %ref2 = moore.class.property_ref %this[@multiplier] : <@DerivedCounter> -> !moore.ref<i8>
  %val2 = moore.read %ref2 : !moore.ref<i8>
  return %val1, %val2 : !moore.i32, !moore.i8
}

//===----------------------------------------------------------------------===//
// Member access with computation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_member_computation
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr, %[[INC:arg1]]: i32) -> i32
// CHECK:   llvm.getelementptr
// CHECK:   comb.add
// CHECK:   return
func.func @test_member_computation(%this: !moore.class<@SimpleCounter>, %increment: !moore.i32) -> !moore.i32 {
  %ref = moore.class.property_ref %this[@count] : <@SimpleCounter> -> !moore.ref<i32>
  %val = moore.read %ref : !moore.ref<i32>
  %result = moore.add %val, %increment : !moore.i32
  return %result : !moore.i32
}

//===----------------------------------------------------------------------===//
// Member access returning reference (for use in array locators, etc.)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_member_ref_return
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr) -> !llhd.ref<i32>
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"SimpleCounter"
// CHECK:   %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return %[[CAST]] : !llhd.ref<i32>
func.func @test_member_ref_return(%this: !moore.class<@SimpleCounter>) -> !moore.ref<i32> {
  %ref = moore.class.property_ref %this[@count] : <@SimpleCounter> -> !moore.ref<i32>
  return %ref : !moore.ref<i32>
}

//===----------------------------------------------------------------------===//
// Multiple function arguments with class handle
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_multi_arg_method
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr, %[[OTHER:arg1]]: !llvm.ptr) -> i32
// CHECK:   llvm.getelementptr %[[THIS]]
// CHECK:   llvm.getelementptr %[[OTHER]]
// CHECK:   comb.add
// CHECK:   return
func.func @test_multi_arg_method(%this: !moore.class<@SimpleCounter>, %other: !moore.class<@SimpleCounter>) -> !moore.i32 {
  %ref1 = moore.class.property_ref %this[@count] : <@SimpleCounter> -> !moore.ref<i32>
  %val1 = moore.read %ref1 : !moore.ref<i32>
  %ref2 = moore.class.property_ref %other[@count] : <@SimpleCounter> -> !moore.ref<i32>
  %val2 = moore.read %ref2 : !moore.ref<i32>
  %result = moore.add %val1, %val2 : !moore.i32
  return %result : !moore.i32
}

//===----------------------------------------------------------------------===//
// Class with queue property - tests array locator inside class context
//===----------------------------------------------------------------------===//

moore.class.classdecl @QueueHolder {
  moore.class.propertydecl @items : !moore.queue<i32, 0>
  moore.class.propertydecl @threshold : !moore.i32
}

// This is the key test case for the bug fix:
// Array locator inside a class method that accesses class members.
// The 'this' pointer is a block argument that gets remapped during
// function signature conversion. The array locator predicate body
// needs to properly resolve references to the remapped block argument.

// CHECK-LABEL: func.func @test_class_method_array_locator
// CHECK-SAME: (%[[THIS:arg0]]: !llvm.ptr)
// CHECK:   llvm.getelementptr %[[THIS]]
// CHECK:   scf.for
// CHECK:   scf.if
func.func @test_class_method_array_locator(%this: !moore.class<@QueueHolder>) -> !moore.queue<i32, 0> {
  // Access the items queue from 'this'
  %items_ref = moore.class.property_ref %this[@items] : <@QueueHolder> -> !moore.ref<queue<i32, 0>>
  %items = moore.read %items_ref : <queue<i32, 0>>

  // Access the threshold from 'this'
  %threshold_ref = moore.class.property_ref %this[@threshold] : <@QueueHolder> -> !moore.ref<i32>
  %threshold = moore.read %threshold_ref : <i32>

  // Array locator that uses external values (threshold from class member)
  // This is where the bug would manifest: the predicate body references
  // %threshold which was derived from %this, and %this is a block argument
  // that gets remapped during conversion.
  %result = moore.array.locator all, elements %items : queue<i32, 0> -> <i32, 0> {
  ^bb0(%item: !moore.i32):
    %cond = moore.sgt %item, %threshold : i32 -> i1
    moore.array.locator.yield %cond : i1
  }

  return %result : !moore.queue<i32, 0>
}
