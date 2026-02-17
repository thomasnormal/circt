// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// VTable Operations Comprehensive Tests
//===----------------------------------------------------------------------===//

// This file contains comprehensive tests for vtable-related operations
// in the MooreToCore conversion pass. It tests VTableOp, VTableEntryOp,
// and VTableLoadMethodOp lowering.

//===----------------------------------------------------------------------===//
// Class Hierarchy Declarations - Simple Inheritance
//===----------------------------------------------------------------------===//

// Base interface with virtual method declarations
moore.class.classdecl @IAnimal {
  moore.class.methoddecl @speak : (!moore.class<@IAnimal>) -> ()
  moore.class.methoddecl @getAge : (!moore.class<@IAnimal>) -> !moore.i32
}

// Derived class that implements IAnimal
moore.class.classdecl @Dog implements [@IAnimal] {
  moore.class.propertydecl @breed : !moore.i32
  moore.class.methoddecl @speak -> @"Dog::speak" : (!moore.class<@Dog>) -> ()
  moore.class.methoddecl @getAge -> @"Dog::getAge" : (!moore.class<@Dog>) -> !moore.i32
}

// Another derived class that implements IAnimal
moore.class.classdecl @Cat implements [@IAnimal] {
  moore.class.methoddecl @speak -> @"Cat::speak" : (!moore.class<@Cat>) -> ()
  moore.class.methoddecl @getAge -> @"Cat::getAge" : (!moore.class<@Cat>) -> !moore.i32
}

// Multi-level inheritance: GoldenRetriever extends Dog which implements IAnimal
moore.class.classdecl @GoldenRetriever extends @Dog {
  moore.class.methoddecl @speak -> @"GoldenRetriever::speak" : (!moore.class<@GoldenRetriever>) -> ()
  moore.class.methoddecl @getAge -> @"GoldenRetriever::getAge" : (!moore.class<@GoldenRetriever>) -> !moore.i32
}

//===----------------------------------------------------------------------===//
// VTable Declarations
//===----------------------------------------------------------------------===//

// VTable for Dog class
moore.vtable @Dog::@vtable {
  moore.vtable @IAnimal::@vtable {
    moore.vtable_entry @speak -> @"Dog::speak"
    moore.vtable_entry @getAge -> @"Dog::getAge"
  }
  moore.vtable_entry @speak -> @"Dog::speak"
  moore.vtable_entry @getAge -> @"Dog::getAge"
}

// VTable for Cat class
moore.vtable @Cat::@vtable {
  moore.vtable @IAnimal::@vtable {
    moore.vtable_entry @speak -> @"Cat::speak"
    moore.vtable_entry @getAge -> @"Cat::getAge"
  }
  moore.vtable_entry @speak -> @"Cat::speak"
  moore.vtable_entry @getAge -> @"Cat::getAge"
}

// VTable for GoldenRetriever class (3-level inheritance)
moore.vtable @GoldenRetriever::@vtable {
  moore.vtable @Dog::@vtable {
    moore.vtable @IAnimal::@vtable {
      moore.vtable_entry @speak -> @"GoldenRetriever::speak"
      moore.vtable_entry @getAge -> @"GoldenRetriever::getAge"
    }
    moore.vtable_entry @speak -> @"GoldenRetriever::speak"
    moore.vtable_entry @getAge -> @"GoldenRetriever::getAge"
  }
  moore.vtable_entry @speak -> @"GoldenRetriever::speak"
  moore.vtable_entry @getAge -> @"GoldenRetriever::getAge"
}

//===----------------------------------------------------------------------===//
// Method Implementations
//===----------------------------------------------------------------------===//

func.func private @"Dog::speak"(%this: !moore.class<@Dog>) {
  return
}

func.func private @"Dog::getAge"(%this: !moore.class<@Dog>) -> !moore.i32 {
  %c42 = moore.constant 42 : !moore.i32
  return %c42 : !moore.i32
}

func.func private @"Cat::speak"(%this: !moore.class<@Cat>) {
  return
}

func.func private @"Cat::getAge"(%this: !moore.class<@Cat>) -> !moore.i32 {
  %c10 = moore.constant 10 : !moore.i32
  return %c10 : !moore.i32
}

func.func private @"GoldenRetriever::speak"(%this: !moore.class<@GoldenRetriever>) {
  return
}

func.func private @"GoldenRetriever::getAge"(%this: !moore.class<@GoldenRetriever>) -> !moore.i32 {
  %c5 = moore.constant 5 : !moore.i32
  return %c5 : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test 1: VTable and VTableEntry ops are erased during lowering
//===----------------------------------------------------------------------===//

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

//===----------------------------------------------------------------------===//
// Test 2: Basic VTableLoadMethodOp - void method
//===----------------------------------------------------------------------===//

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_load_void_method
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_load_void_method(%obj: !moore.class<@Dog>) {
  %fptr = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 3: VTableLoadMethodOp - method with return value
//===----------------------------------------------------------------------===//

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_load_method_with_return
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_load_method_with_return(%obj: !moore.class<@Dog>) {
  %fptr = moore.vtable.load_method %obj : @getAge of <@Dog> -> (!moore.class<@Dog>) -> !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Loading different methods from same object
//===----------------------------------------------------------------------===//

// When the results are not used, the constants get dead-code-eliminated.
// CHECK-LABEL: func.func @test_load_multiple_methods
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_load_multiple_methods(%obj: !moore.class<@Dog>) {
  %fptr1 = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  %fptr2 = moore.vtable.load_method %obj : @getAge of <@Dog> -> (!moore.class<@Dog>) -> !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Loading same method from different classes (polymorphism)
//===----------------------------------------------------------------------===//

// When the results are not used, the constants get dead-code-eliminated.
// CHECK-LABEL: func.func @test_polymorphic_method_load
// CHECK-SAME:    (%[[DOG:.*]]: !llvm.ptr, %[[CAT:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_polymorphic_method_load(%dog: !moore.class<@Dog>, %cat: !moore.class<@Cat>) {
  %fptr1 = moore.vtable.load_method %dog : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  %fptr2 = moore.vtable.load_method %cat : @speak of <@Cat> -> (!moore.class<@Cat>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Multi-level inheritance (3 levels deep)
//===----------------------------------------------------------------------===//

// When the results are not used, the constants get dead-code-eliminated.
// CHECK-LABEL: func.func @test_multilevel_inheritance
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_multilevel_inheritance(%obj: !moore.class<@GoldenRetriever>) {
  // speak is overridden in GoldenRetriever
  %fptr1 = moore.vtable.load_method %obj : @speak of <@GoldenRetriever> -> (!moore.class<@GoldenRetriever>) -> ()
  // getAge is overridden in GoldenRetriever
  %fptr2 = moore.vtable.load_method %obj : @getAge of <@GoldenRetriever> -> (!moore.class<@GoldenRetriever>) -> !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// Test 7: Loading method and calling it via func.call_indirect
// With dynamic dispatch, this performs runtime vtable lookup
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_load_and_call_method
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][{{%.+}}, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 0]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]
// CHECK:         call_indirect {{.*}}(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return

func.func @test_load_and_call_method(%obj: !moore.class<@Dog>) {
  %fptr = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  func.call_indirect %fptr(%obj) : (!moore.class<@Dog>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 8: Loading method with return value and using the result
// With dynamic dispatch, this performs runtime vtable lookup
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_load_call_use_result
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr) -> i32
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][{{%.+}}, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 1]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]
// CHECK:         %[[RESULT:.*]] = call_indirect {{.*}}(%[[OBJ]]) : (!llvm.ptr) -> i32
// CHECK:         return %[[RESULT]] : i32

func.func @test_load_call_use_result(%obj: !moore.class<@Dog>) -> !moore.i32 {
  %fptr = moore.vtable.load_method %obj : @getAge of <@Dog> -> (!moore.class<@Dog>) -> !moore.i32
  %result = func.call_indirect %fptr(%obj) : (!moore.class<@Dog>) -> !moore.i32
  return %result : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test 9: Method dispatch in control flow
// With dynamic dispatch, vtable loads happen in each branch
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_method_in_control_flow
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[COND:.*]]: i1)
// CHECK:         cf.cond_br %[[COND]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK:       ^[[BB1]]:
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         call_indirect
// CHECK:         cf.br ^[[BB3:.*]]
// CHECK:       ^[[BB2]]:
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         call_indirect
// CHECK:         cf.br ^[[BB3]]
// CHECK:       ^[[BB3]]:
// CHECK:         return

func.func @test_method_in_control_flow(%obj: !moore.class<@Dog>, %cond: i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %fptr1 = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  func.call_indirect %fptr1(%obj) : (!moore.class<@Dog>) -> ()
  cf.br ^bb3
^bb2:
  %fptr2 = moore.vtable.load_method %obj : @getAge of <@Dog> -> (!moore.class<@Dog>) -> !moore.i32
  %age = func.call_indirect %fptr2(%obj) : (!moore.class<@Dog>) -> !moore.i32
  cf.br ^bb3
^bb3:
  return
}

//===----------------------------------------------------------------------===//
// Test 10: Multiple virtual calls in sequence
// Each call loads from vtable dynamically
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_multiple_virtual_calls
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// First vtable load (speak at index 0)
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr{{.*}}[0, 0]
// CHECK:         llvm.load
// Second vtable load (getAge at index 1)
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr{{.*}}[0, 1]
// CHECK:         llvm.load
// Three indirect calls
// CHECK:         call_indirect
// CHECK:         call_indirect
// CHECK:         call_indirect
// CHECK:         return

func.func @test_multiple_virtual_calls(%obj: !moore.class<@Dog>) {
  %fptr1 = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  %fptr2 = moore.vtable.load_method %obj : @getAge of <@Dog> -> (!moore.class<@Dog>) -> !moore.i32
  func.call_indirect %fptr1(%obj) : (!moore.class<@Dog>) -> ()
  %age = func.call_indirect %fptr2(%obj) : (!moore.class<@Dog>) -> !moore.i32
  func.call_indirect %fptr1(%obj) : (!moore.class<@Dog>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 11: Polymorphic virtual calls with different implementations
// Each object's vtable is accessed independently
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_polymorphic_calls
// CHECK-SAME:    (%[[DOG:.*]]: !llvm.ptr, %[[CAT:.*]]: !llvm.ptr, %[[GOLDEN:.*]]: !llvm.ptr)
// First vtable load for Dog
// CHECK:         llvm.getelementptr %[[DOG]]
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// Second vtable load for Cat
// CHECK:         llvm.getelementptr %[[CAT]]
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// Third vtable load for GoldenRetriever
// CHECK:         llvm.getelementptr %[[GOLDEN]]
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// Three indirect calls
// CHECK:         call_indirect
// CHECK:         call_indirect
// CHECK:         call_indirect
// CHECK:         return

func.func @test_polymorphic_calls(%dog: !moore.class<@Dog>, %cat: !moore.class<@Cat>, %golden: !moore.class<@GoldenRetriever>) {
  %fptr1 = moore.vtable.load_method %dog : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  %fptr2 = moore.vtable.load_method %cat : @speak of <@Cat> -> (!moore.class<@Cat>) -> ()
  %fptr3 = moore.vtable.load_method %golden : @speak of <@GoldenRetriever> -> (!moore.class<@GoldenRetriever>) -> ()
  func.call_indirect %fptr1(%dog) : (!moore.class<@Dog>) -> ()
  func.call_indirect %fptr2(%cat) : (!moore.class<@Cat>) -> ()
  func.call_indirect %fptr3(%golden) : (!moore.class<@GoldenRetriever>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 12: Compare loaded function pointers (same method from same class)
//===----------------------------------------------------------------------===//

// When the results are not used, the constants get dead-code-eliminated.
// CHECK-LABEL: func.func @test_same_method_twice
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_same_method_twice(%obj: !moore.class<@Dog>) {
  // Load the same method twice - results unused so get eliminated
  %fptr1 = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  %fptr2 = moore.vtable.load_method %obj : @speak of <@Dog> -> (!moore.class<@Dog>) -> ()
  return
}
