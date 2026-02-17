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

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_abstract_class_vtable_load
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_abstract_class_vtable_load(%obj: !moore.class<@AbstractBase>) {
  // Load method through abstract class handle - should find method in nested vtable
  %fptr = moore.vtable.load_method %obj : @initialize of <@AbstractBase> -> (!moore.class<@AbstractBase>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 2: VTable load through concrete class handle (should still work)
//===----------------------------------------------------------------------===//

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_concrete_class_vtable_load
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_concrete_class_vtable_load(%obj: !moore.class<@ConcreteChild>) {
  // Load method through concrete class handle - should work via top-level vtable
  %fptr = moore.vtable.load_method %obj : @initialize of <@ConcreteChild> -> (!moore.class<@ConcreteChild>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 3: VTable load and call through abstract class handle
// With dynamic dispatch, this performs runtime vtable lookup
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_abstract_vtable_load_and_call
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         %[[VTABLE_PTR_PTR:.*]] = llvm.getelementptr %[[OBJ]][{{%.+}}, 1]
// CHECK:         %[[VTABLE_PTR:.*]] = llvm.load %[[VTABLE_PTR_PTR]]
// CHECK:         %[[FUNC_PTR_PTR:.*]] = llvm.getelementptr %[[VTABLE_PTR]][0, 1]
// CHECK:         %[[FUNC_PTR:.*]] = llvm.load %[[FUNC_PTR_PTR]]
// CHECK:         call_indirect {{.*}}(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return

func.func @test_abstract_vtable_load_and_call(%obj: !moore.class<@AbstractBase>) {
  %fptr = moore.vtable.load_method %obj : @initialize of <@AbstractBase> -> (!moore.class<@AbstractBase>) -> ()
  func.call_indirect %fptr(%obj) : (!moore.class<@AbstractBase>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Class with no vtable segment - method found via global vtable search
//===----------------------------------------------------------------------===//

// This tests the scenario where a class (MiddleClass) extends a base class
// but has no vtable segment of its own in any vtable. The method should still
// be found by searching all available vtables.

// Base class with method
moore.class.classdecl @BaseWithMethod {
  moore.class.methoddecl @get_name -> @"BaseWithMethod::get_name" : (!moore.class<@BaseWithMethod>) -> !moore.i32
}

// Middle class extends base but has no derived classes that create a vtable
// segment for it specifically
moore.class.classdecl @MiddleClass extends @BaseWithMethod {
  moore.class.methoddecl @get_name -> @"BaseWithMethod::get_name" : (!moore.class<@MiddleClass>) -> !moore.i32
}

// Another concrete class that creates a vtable (but not for MiddleClass)
moore.class.classdecl @OtherConcrete extends @BaseWithMethod {
  moore.class.methoddecl @get_name -> @"OtherConcrete::get_name" : (!moore.class<@OtherConcrete>) -> !moore.i32
}

// VTable for OtherConcrete - does NOT contain a segment for MiddleClass
moore.vtable @OtherConcrete::@vtable {
  moore.vtable @BaseWithMethod::@vtable {
    moore.vtable_entry @get_name -> @"OtherConcrete::get_name"
  }
  moore.vtable_entry @get_name -> @"OtherConcrete::get_name"
}

func.func private @"BaseWithMethod::get_name"(%this: !moore.class<@BaseWithMethod>) -> !moore.i32 {
  %c0 = moore.constant 0 : !moore.i32
  return %c0 : !moore.i32
}

func.func private @"OtherConcrete::get_name"(%this: !moore.class<@OtherConcrete>) -> !moore.i32 {
  %c1 = moore.constant 1 : !moore.i32
  return %c1 : !moore.i32
}

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_no_vtable_segment_fallback
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_no_vtable_segment_fallback(%obj: !moore.class<@MiddleClass>) {
  // MiddleClass has no vtable segment anywhere, but get_name should be found
  // by searching all vtables in the module
  %fptr = moore.vtable.load_method %obj : @get_name of <@MiddleClass> -> (!moore.class<@MiddleClass>) -> !moore.i32
  return
}

//===----------------------------------------------------------------------===//
// Test 5: Multiple intermediate classes in hierarchy (A -> B -> C)
// where B has no vtable segment of its own
//===----------------------------------------------------------------------===//

// This tests a deeper hierarchy where MiddleAbstract doesn't have a vtable segment
// but its parent TopBase does, and its child LeafConcrete creates the actual vtable.

// Top-level base class
moore.class.classdecl @TopBase {
  moore.class.methoddecl @get_id -> @"TopBase::get_id" : (!moore.class<@TopBase>) -> !moore.i32
  moore.class.methoddecl @describe -> @"TopBase::describe" : (!moore.class<@TopBase>) -> !moore.i32
}

// Middle abstract class - extends TopBase but has no vtable segment
moore.class.classdecl @MiddleAbstract extends @TopBase {
  moore.class.methoddecl @get_id -> @"MiddleAbstract::get_id" : (!moore.class<@MiddleAbstract>) -> !moore.i32
  moore.class.methoddecl @describe -> @"MiddleAbstract::describe" : (!moore.class<@MiddleAbstract>) -> !moore.i32
  moore.class.methoddecl @process : (!moore.class<@MiddleAbstract>) -> ()
}

// Leaf concrete class - creates vtable for itself and TopBase (but not MiddleAbstract)
moore.class.classdecl @LeafConcrete extends @MiddleAbstract {
  moore.class.methoddecl @get_id -> @"LeafConcrete::get_id" : (!moore.class<@LeafConcrete>) -> !moore.i32
  moore.class.methoddecl @describe -> @"LeafConcrete::describe" : (!moore.class<@LeafConcrete>) -> !moore.i32
  moore.class.methoddecl @process -> @"LeafConcrete::process" : (!moore.class<@LeafConcrete>) -> ()
}

// VTable for LeafConcrete - contains segment for TopBase but NOT for MiddleAbstract
moore.vtable @LeafConcrete::@vtable {
  moore.vtable @TopBase::@vtable {
    moore.vtable_entry @get_id -> @"LeafConcrete::get_id"
    moore.vtable_entry @describe -> @"LeafConcrete::describe"
  }
  moore.vtable_entry @get_id -> @"LeafConcrete::get_id"
  moore.vtable_entry @describe -> @"LeafConcrete::describe"
  moore.vtable_entry @process -> @"LeafConcrete::process"
}

func.func private @"TopBase::get_id"(%this: !moore.class<@TopBase>) -> !moore.i32 {
  %c10 = moore.constant 10 : !moore.i32
  return %c10 : !moore.i32
}

func.func private @"TopBase::describe"(%this: !moore.class<@TopBase>) -> !moore.i32 {
  %c100 = moore.constant 100 : !moore.i32
  return %c100 : !moore.i32
}

func.func private @"MiddleAbstract::get_id"(%this: !moore.class<@MiddleAbstract>) -> !moore.i32 {
  %c20 = moore.constant 20 : !moore.i32
  return %c20 : !moore.i32
}

func.func private @"MiddleAbstract::describe"(%this: !moore.class<@MiddleAbstract>) -> !moore.i32 {
  %c200 = moore.constant 200 : !moore.i32
  return %c200 : !moore.i32
}

func.func private @"LeafConcrete::get_id"(%this: !moore.class<@LeafConcrete>) -> !moore.i32 {
  %c30 = moore.constant 30 : !moore.i32
  return %c30 : !moore.i32
}

func.func private @"LeafConcrete::describe"(%this: !moore.class<@LeafConcrete>) -> !moore.i32 {
  %c300 = moore.constant 300 : !moore.i32
  return %c300 : !moore.i32
}

func.func private @"LeafConcrete::process"(%this: !moore.class<@LeafConcrete>) {
  return
}

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_deep_hierarchy_middle_no_vtable
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_deep_hierarchy_middle_no_vtable(%obj: !moore.class<@MiddleAbstract>) {
  // MiddleAbstract has no vtable segment, but get_id should be found by
  // searching all vtables - will find it in LeafConcrete's vtable
  %fptr = moore.vtable.load_method %obj : @get_id of <@MiddleAbstract> -> (!moore.class<@MiddleAbstract>) -> !moore.i32
  return
}

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_deep_hierarchy_process_method
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_deep_hierarchy_process_method(%obj: !moore.class<@MiddleAbstract>) {
  // The process method is only declared in MiddleAbstract but implemented in LeafConcrete
  // This tests that methods unique to middle classes can still be found
  %fptr = moore.vtable.load_method %obj : @process of <@MiddleAbstract> -> (!moore.class<@MiddleAbstract>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test 6: Multiple vtables where method could be in any of them
//===----------------------------------------------------------------------===//

// This tests that when multiple concrete classes exist, the fallback search
// finds the method in any available vtable.

// Interface-like base class
moore.class.classdecl @Printable {
  moore.class.methoddecl @print -> @"Printable::print" : (!moore.class<@Printable>) -> ()
}

// First concrete implementation
moore.class.classdecl @Document extends @Printable {
  moore.class.methoddecl @print -> @"Document::print" : (!moore.class<@Document>) -> ()
}

// Second concrete implementation
moore.class.classdecl @Report extends @Printable {
  moore.class.methoddecl @print -> @"Report::print" : (!moore.class<@Report>) -> ()
}

// VTable for Document
moore.vtable @Document::@vtable {
  moore.vtable @Printable::@vtable {
    moore.vtable_entry @print -> @"Document::print"
  }
  moore.vtable_entry @print -> @"Document::print"
}

// VTable for Report
moore.vtable @Report::@vtable {
  moore.vtable @Printable::@vtable {
    moore.vtable_entry @print -> @"Report::print"
  }
  moore.vtable_entry @print -> @"Report::print"
}

func.func private @"Printable::print"(%this: !moore.class<@Printable>) {
  return
}

func.func private @"Document::print"(%this: !moore.class<@Document>) {
  return
}

func.func private @"Report::print"(%this: !moore.class<@Report>) {
  return
}

// When the result is not used, the constant gets dead-code-eliminated.
// CHECK-LABEL: func.func @test_multiple_vtables_available
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         return

func.func @test_multiple_vtables_available(%obj: !moore.class<@Printable>) {
  // Printable has no direct vtable, but both Document and Report have vtables
  // with Printable segments. The fallback search should find print in one of them.
  // (It will find the first one encountered - Document in this case)
  %fptr = moore.vtable.load_method %obj : @print of <@Printable> -> (!moore.class<@Printable>) -> ()
  return
}
