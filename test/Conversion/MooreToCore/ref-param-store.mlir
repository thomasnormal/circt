// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that class output parameters use llvm.store instead of llhd.drv
// This validates the AssignOpConversion fix that detects function ref
// parameters with class types and uses LLVM store operations.

//===----------------------------------------------------------------------===//
// Class declarations for testing
//===----------------------------------------------------------------------===//

moore.class.classdecl @RefTestClass {
  moore.class.propertydecl @value : !moore.i32
}

moore.class.classdecl @AnotherClass {
  moore.class.propertydecl @data : !moore.i64
}

//===----------------------------------------------------------------------===//
// Test: Assignment to class ref parameter should use llvm.store, not llhd.drv
//===----------------------------------------------------------------------===//

// The key fix is that AssignOpConversion detects when the destination is a
// ref-typed block argument in a function context with a class (llvm.ptr) type
// and uses llvm.store instead of llhd.drv. The interpreter doesn't track refs
// passed through function calls as signals for class memory.

// CHECK-LABEL: func.func @test_class_output_param
// CHECK-SAME: (%[[REF:arg0]]: !llhd.ref<!llvm.ptr>, %[[OBJ:arg1]]: !llvm.ptr)
// CHECK:   %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[REF]] : !llhd.ref<!llvm.ptr> to !llvm.ptr
// CHECK:   llvm.store %[[OBJ]], %[[PTR]] : !llvm.ptr, !llvm.ptr
// CHECK:   return
// CHECK-NOT: llhd.drv
func.func @test_class_output_param(%ref: !moore.ref<class<@RefTestClass>>, %obj: !moore.class<@RefTestClass>) {
  moore.blocking_assign %ref, %obj : class<@RefTestClass>
  return
}

//===----------------------------------------------------------------------===//
// Test: Multiple class ref parameters
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_multiple_class_refs
// CHECK-SAME: (%[[REF1:arg0]]: !llhd.ref<!llvm.ptr>, %[[REF2:arg1]]: !llhd.ref<!llvm.ptr>, %[[OBJ1:arg2]]: !llvm.ptr, %[[OBJ2:arg3]]: !llvm.ptr)
// CHECK:   builtin.unrealized_conversion_cast %[[REF1]] : !llhd.ref<!llvm.ptr> to !llvm.ptr
// CHECK:   llvm.store %[[OBJ1]]
// CHECK:   builtin.unrealized_conversion_cast %[[REF2]] : !llhd.ref<!llvm.ptr> to !llvm.ptr
// CHECK:   llvm.store %[[OBJ2]]
// CHECK:   return
// CHECK-NOT: llhd.drv
func.func @test_multiple_class_refs(%ref1: !moore.ref<class<@RefTestClass>>, %ref2: !moore.ref<class<@AnotherClass>>, %obj1: !moore.class<@RefTestClass>, %obj2: !moore.class<@AnotherClass>) {
  moore.blocking_assign %ref1, %obj1 : class<@RefTestClass>
  moore.blocking_assign %ref2, %obj2 : class<@AnotherClass>
  return
}

//===----------------------------------------------------------------------===//
// Test: Class ref param after creating new instance
//===----------------------------------------------------------------------===//

// This pattern is common in UVM: create a new object and assign to output ref

// CHECK-LABEL: func.func @test_class_ref_with_new
// CHECK-SAME: (%[[REF:arg0]]: !llhd.ref<!llvm.ptr>)
// CHECK:   llvm.call @malloc
// CHECK:   %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[REF]] : !llhd.ref<!llvm.ptr> to !llvm.ptr
// CHECK:   llvm.store %{{.*}}, %[[PTR]] : !llvm.ptr, !llvm.ptr
// CHECK:   return
// CHECK-NOT: llhd.drv
func.func @test_class_ref_with_new(%ref: !moore.ref<class<@RefTestClass>>) {
  %obj = moore.class.new : <@RefTestClass>
  moore.blocking_assign %ref, %obj : class<@RefTestClass>
  return
}

//===----------------------------------------------------------------------===//
// Test: Conditional assignment to class ref param
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_class_ref_conditional
// CHECK-SAME: (%[[REF:arg0]]: !llhd.ref<!llvm.ptr>, %[[OBJ1:arg1]]: !llvm.ptr, %[[OBJ2:arg2]]: !llvm.ptr, %[[COND:arg3]]: i1)
// CHECK:   scf.if
// CHECK:     builtin.unrealized_conversion_cast
// CHECK:     llvm.store
// CHECK:   else
// CHECK:     builtin.unrealized_conversion_cast
// CHECK:     llvm.store
// CHECK-NOT: llhd.drv
func.func @test_class_ref_conditional(%ref: !moore.ref<class<@RefTestClass>>, %obj1: !moore.class<@RefTestClass>, %obj2: !moore.class<@RefTestClass>, %cond: !moore.i1) {
  %c = moore.conversion %cond : !moore.i1 -> i1
  scf.if %c {
    moore.blocking_assign %ref, %obj1 : class<@RefTestClass>
  } else {
    moore.blocking_assign %ref, %obj2 : class<@RefTestClass>
  }
  return
}
