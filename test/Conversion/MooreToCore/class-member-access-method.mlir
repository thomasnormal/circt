// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test class member read from method - verifies block argument remapping
// for class method contexts where 'this' pointer needs to be correctly
// remapped after function signature conversion.
//
// This tests the fix for the issue where reading class member variables
// from methods (other than constructor) would cause simulation termination
// due to invalid pointer access. The problem was that when class methods
// are converted to LLVM, the 'this' pointer (block argument) gets remapped
// but operations inside methods still referenced old/invalidated block
// arguments.

//===----------------------------------------------------------------------===//
// Test class definition
//===----------------------------------------------------------------------===//

moore.class.classdecl @TestClass {
  moore.class.propertydecl @value : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test method that reads member variable
//===----------------------------------------------------------------------===//

// This function simulates a class method that accesses 'this.value'.
// The 'this' pointer is passed as the first argument (block argument 0).
// The fix ensures we find the new block argument at the corresponding
// position in the converted function's entry block.

// CHECK-LABEL: func.func @test_get_value
// CHECK-SAME: (%[[THIS:.*]]: !llvm.ptr) -> i32
// CHECK:   %[[GEP:.*]] = llvm.getelementptr %[[THIS]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass"
// CHECK:   %[[LOAD:.*]] = llvm.load %[[GEP]] : !llvm.ptr -> i32
// CHECK:   return %[[LOAD]] : i32
func.func @test_get_value(%this: !moore.class<@TestClass>) -> !moore.i32 {
  %ref = moore.class.property_ref %this[@value] : <@TestClass> -> !moore.ref<i32>
  %val = moore.read %ref : !moore.ref<i32>
  return %val : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test write to member variable (for completeness)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_set_value
// CHECK-SAME: (%[[THIS2:.*]]: !llvm.ptr, %[[VAL:.*]]: i32)
// CHECK:   %[[GEP2:.*]] = llvm.getelementptr %[[THIS2]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<"TestClass"
// CHECK:   llvm.store %[[VAL]], %[[GEP2]] : i32, !llvm.ptr
func.func @test_set_value(%this: !moore.class<@TestClass>, %newval: !moore.i32) {
  %ref = moore.class.property_ref %this[@value] : <@TestClass> -> !moore.ref<i32>
  moore.blocking_assign %ref, %newval : i32
  return
}
