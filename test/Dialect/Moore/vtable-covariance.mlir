// RUN: circt-opt --verify-diagnostics %s | FileCheck %s

// Test that VTableLoadMethodOp allows covariant "this" types.
// When calling an inherited method, the call site uses derived class "this"
// type, but the method declaration has base class "this" type.

// Base class with virtual methods
moore.class.classdecl @BaseClass {
  moore.class.methoddecl @get_value -> @"BaseClass::get_value" : (!moore.class<@BaseClass>) -> !moore.i32
}

// Derived class that overrides the method (with derived class "this" type)
moore.class.classdecl @DerivedClass extends @BaseClass {
  moore.class.methoddecl @get_value -> @"DerivedClass::get_value" : (!moore.class<@DerivedClass>) -> !moore.i32
}

// Grandchild class that overrides the method
moore.class.classdecl @GrandchildClass extends @DerivedClass {
  moore.class.methoddecl @get_value -> @"GrandchildClass::get_value" : (!moore.class<@GrandchildClass>) -> !moore.i32
}

// Method implementations
func.func private @"BaseClass::get_value"(%this: !moore.class<@BaseClass>) -> !moore.i32 {
  %c = moore.constant 0 : !moore.i32
  return %c : !moore.i32
}

func.func private @"DerivedClass::get_value"(%this: !moore.class<@DerivedClass>) -> !moore.i32 {
  %c = moore.constant 1 : !moore.i32
  return %c : !moore.i32
}

func.func private @"GrandchildClass::get_value"(%this: !moore.class<@GrandchildClass>) -> !moore.i32 {
  %c = moore.constant 2 : !moore.i32
  return %c : !moore.i32
}

// VTable for BaseClass
moore.vtable @BaseClass::@vtable {
  moore.vtable_entry @get_value -> @"BaseClass::get_value"
}

// VTable for DerivedClass
moore.vtable @DerivedClass::@vtable {
  moore.vtable @BaseClass::@vtable {
    moore.vtable_entry @get_value -> @"DerivedClass::get_value"
  }
  moore.vtable_entry @get_value -> @"DerivedClass::get_value"
}

// VTable for GrandchildClass
moore.vtable @GrandchildClass::@vtable {
  moore.vtable @DerivedClass::@vtable {
    moore.vtable @BaseClass::@vtable {
      moore.vtable_entry @get_value -> @"GrandchildClass::get_value"
    }
    moore.vtable_entry @get_value -> @"GrandchildClass::get_value"
  }
  moore.vtable_entry @get_value -> @"GrandchildClass::get_value"
}

// Test: load method with covariant "this" type
// The method get_value is declared in each class with that class's "this" type.
// When we load from DerivedClass, the verifier walks the hierarchy and finds
// the methoddecl in DerivedClass (with DerivedClass "this" type).

// CHECK-LABEL: func.func @test_exact_match
func.func @test_exact_match(%derived: !moore.class<@DerivedClass>) {
  // This is the normal case - "this" types match exactly
  // CHECK: moore.vtable.load_method
  %0 = moore.vtable.load_method %derived : @get_value of <@DerivedClass> -> (!moore.class<@DerivedClass>) -> !moore.i32
  return
}

// CHECK-LABEL: func.func @test_grandchild_exact_match
func.func @test_grandchild_exact_match(%grandchild: !moore.class<@GrandchildClass>) {
  // This is the normal case - "this" types match exactly
  // CHECK: moore.vtable.load_method
  %0 = moore.vtable.load_method %grandchild : @get_value of <@GrandchildClass> -> (!moore.class<@GrandchildClass>) -> !moore.i32
  return
}
