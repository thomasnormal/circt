// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Test: Inherited virtual methods in vtable generation
//===----------------------------------------------------------------------===//
//
// Problem: When a derived class inherits virtual methods from a base class
// without overriding them, those methods were NOT being registered in the
// derived class's vtable.
//
// Root cause: classAST.members() only returns explicitly defined members,
// not inherited ones.
//
// Fix: After Pass 2, walk up the inheritance chain using getBaseClass() and
// for each inherited virtual method not overridden in the current class,
// create a ClassMethodDeclOp pointing to the base class's function.
//===----------------------------------------------------------------------===//

// Base class with virtual methods
// CHECK-LABEL: moore.class.classdecl @BaseClass {
// CHECK:   moore.class.methoddecl @get_name -> @"BaseClass::get_name"
// CHECK:   moore.class.methoddecl @display -> @"BaseClass::display"
// CHECK: }
class BaseClass;
    virtual function string get_name();
        return "BaseClass";
    endfunction

    virtual function void display();
        $display("Base display");
    endfunction
endclass

// Derived class that only overrides get_name but NOT display
// The vtable should contain BOTH methods:
// - get_name: points to DerivedClass::get_name (overridden)
// - display: points to BaseClass::display (inherited)
// CHECK-LABEL: moore.class.classdecl @DerivedClass extends @BaseClass {
// CHECK:   moore.class.methoddecl @get_name -> @"DerivedClass::get_name"
// CHECK:   moore.class.methoddecl @display -> @"BaseClass::display"
// CHECK: }
class DerivedClass extends BaseClass;
    virtual function string get_name();
        return "DerivedClass";
    endfunction
    // Note: display() is NOT overridden - it should be inherited
endclass

// Third-level derived class that doesn't override anything
// Should inherit BOTH methods from the chain
// CHECK-LABEL: moore.class.classdecl @GrandchildClass extends @DerivedClass {
// CHECK:   moore.class.methoddecl @get_name -> @"DerivedClass::get_name"
// CHECK:   moore.class.methoddecl @display -> @"BaseClass::display"
// CHECK: }
class GrandchildClass extends DerivedClass;
    // No overrides - both methods should be inherited
endclass

// Test class with multiple virtual methods at different levels
// CHECK-LABEL: moore.class.classdecl @Level0 {
// CHECK:   moore.class.methoddecl @method_a -> @"Level0::method_a"
// CHECK:   moore.class.methoddecl @method_b -> @"Level0::method_b"
// CHECK: }
class Level0;
    virtual function int method_a();
        return 0;
    endfunction

    virtual function int method_b();
        return 0;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @Level1 extends @Level0 {
// CHECK:   moore.class.methoddecl @method_a -> @"Level1::method_a"
// CHECK:   moore.class.methoddecl @method_c -> @"Level1::method_c"
// CHECK:   moore.class.methoddecl @method_b -> @"Level0::method_b"
// CHECK: }
class Level1 extends Level0;
    // Override method_a
    virtual function int method_a();
        return 1;
    endfunction

    // Add new method_c
    virtual function int method_c();
        return 1;
    endfunction
    // method_b should be inherited from Level0
endclass

// CHECK-LABEL: moore.class.classdecl @Level2 extends @Level1 {
// CHECK:   moore.class.methoddecl @method_b -> @"Level2::method_b"
// CHECK:   moore.class.methoddecl @method_a -> @"Level1::method_a"
// CHECK:   moore.class.methoddecl @method_c -> @"Level1::method_c"
// CHECK: }
class Level2 extends Level1;
    // Override method_b (originally from Level0)
    virtual function int method_b();
        return 2;
    endfunction
    // method_a should be inherited from Level1
    // method_c should be inherited from Level1
endclass

// Test that virtual dispatch works correctly
// CHECK-LABEL: moore.module @VirtualDispatchTest() {
module VirtualDispatchTest;
    BaseClass base_handle;
    DerivedClass derived_handle;
    GrandchildClass grandchild_handle;

    initial begin
        derived_handle = new;
        grandchild_handle = new;

        // Test virtual dispatch through base class handle
        // The display method should dispatch to BaseClass::display
        // even though called on a DerivedClass object
        // CHECK: moore.vtable.load_method
        base_handle = derived_handle;
        base_handle.display();

        // Test virtual dispatch on grandchild
        // CHECK: moore.vtable.load_method
        base_handle = grandchild_handle;
        base_handle.display();

        // Call overridden method
        // CHECK: moore.vtable.load_method
        $display(base_handle.get_name());
    end
endmodule
