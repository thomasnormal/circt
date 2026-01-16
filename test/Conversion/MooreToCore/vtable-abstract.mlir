// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test VTable Operations with Abstract Classes
//===----------------------------------------------------------------------===//

// This file tests vtable.load_method calls through abstract class handles.
// Abstract classes don't have their own top-level vtables, but their vtable
// segments appear nested inside concrete derived class vtables.

//===----------------------------------------------------------------------===//
// Class Hierarchy: Abstract base -> Concrete derived
//===----------------------------------------------------------------------===//

// Abstract base class with pure virtual method (get_type_name has no impl)
moore.class.classdecl @AbstractBase {
  moore.class.methoddecl @get_type_name : (!moore.class<@AbstractBase>) -> !moore.i32
  moore.class.methoddecl @initialize -> @"AbstractBase::initialize" : (!moore.class<@AbstractBase>) -> ()
}

// Concrete derived class implementing all methods
moore.class.classdecl @ConcreteChild extends @AbstractBase {
  moore.class.methoddecl @get_type_name -> @"ConcreteChild::get_type_name" : (!moore.class<@ConcreteChild>) -> !moore.i32
  moore.class.methoddecl @initialize -> @"ConcreteChild::initialize" : (!moore.class<@ConcreteChild>) -> ()
}

//===----------------------------------------------------------------------===//
// VTable Declarations (only for concrete classes)
//===----------------------------------------------------------------------===//

// VTable for ConcreteChild - contains nested vtable for AbstractBase
moore.vtable @ConcreteChild::@vtable {
  // Nested vtable for AbstractBase (abstract class doesn't get its own top-level vtable)
  moore.vtable @AbstractBase::@vtable {
    moore.vtable_entry @get_type_name -> @"ConcreteChild::get_type_name"
    moore.vtable_entry @initialize -> @"ConcreteChild::initialize"
  }
  moore.vtable_entry @get_type_name -> @"ConcreteChild::get_type_name"
  moore.vtable_entry @initialize -> @"ConcreteChild::initialize"
}

//===----------------------------------------------------------------------===//
// Method Implementations
//===----------------------------------------------------------------------===//

func.func private @"AbstractBase::initialize"(%this: !moore.class<@AbstractBase>) {
  return
}

func.func private @"ConcreteChild::get_type_name"(%this: !moore.class<@ConcreteChild>) -> !moore.i32 {
  %c1 = moore.constant 1 : !moore.i32
  return %c1 : !moore.i32
}

func.func private @"ConcreteChild::initialize"(%this: !moore.class<@ConcreteChild>) {
  return
}

//===----------------------------------------------------------------------===//
// Test 1: VTable load through abstract class handle
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_abstract_class_vtable_load
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         %[[FPTR:.*]] = constant @"ConcreteChild::initialize" : (!llvm.ptr) -> ()
// CHECK:         return

func.func @test_abstract_class_vtable_load(%obj: !moore.class<@AbstractBase>) {
  // Load method through abstract class handle - should find method in nested vtable
  %fptr = moore.vtable.load_method %obj : @initialize of <@AbstractBase> -> (!moore.class<@AbstractBase>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 2: VTable load through concrete class handle (should still work)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_concrete_class_vtable_load
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         %[[FPTR:.*]] = constant @"ConcreteChild::initialize" : (!llvm.ptr) -> ()
// CHECK:         return

func.func @test_concrete_class_vtable_load(%obj: !moore.class<@ConcreteChild>) {
  // Load method through concrete class handle - should work via top-level vtable
  %fptr = moore.vtable.load_method %obj : @initialize of <@ConcreteChild> -> (!moore.class<@ConcreteChild>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 3: VTable load and call through abstract class handle
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_abstract_vtable_load_and_call
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         %[[FPTR:.*]] = constant @"ConcreteChild::initialize" : (!llvm.ptr) -> ()
// CHECK:         call_indirect %[[FPTR]](%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return

func.func @test_abstract_vtable_load_and_call(%obj: !moore.class<@AbstractBase>) {
  %fptr = moore.vtable.load_method %obj : @initialize of <@AbstractBase> -> (!moore.class<@AbstractBase>) -> ()
  func.call_indirect %fptr(%obj) : (!moore.class<@AbstractBase>) -> ()
  return
}
