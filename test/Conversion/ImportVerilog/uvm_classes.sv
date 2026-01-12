// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// UVM Class Parsing Feature Tests
//===----------------------------------------------------------------------===//

/// Test generic class specialization mapping
/// This tests that specialized classes derived from parameterized base classes
/// are properly recognized. For example, uvm_pool_18 should be recognized as
/// derived from uvm_pool#(KEY, T).

// CHECK-LABEL: moore.class.classdecl @uvm_pool {
// CHECK: }
class uvm_pool #(type KEY=int, type T=int);
endclass

// CHECK-LABEL: moore.class.classdecl @uvm_pool_int_string extends @uvm_pool {
// CHECK: }
class uvm_pool_int_string extends uvm_pool#(int, string);
endclass

// CHECK-LABEL: moore.class.classdecl @uvm_pool_string_int extends @uvm_pool {
// CHECK: }
class uvm_pool_string_int extends uvm_pool#(string, int);
endclass

// Test that multiple specializations are recognized
// CHECK-LABEL: moore.module @GenericClassSpecializationTest() {
module GenericClassSpecializationTest;
    // CHECK: moore.variable : <!moore.class<@uvm_pool_int_string>>
    uvm_pool_int_string pool1;
    // CHECK: moore.variable : <!moore.class<@uvm_pool_string_int>>
    uvm_pool_string_int pool2;
endmodule

/// Test class handle comparison operations (== and !=)
/// These are used extensively in UVM for comparing object handles.

// CHECK-LABEL: moore.class.classdecl @HandleTestClass {
// CHECK:   moore.class.propertydecl @value : !moore.i32
// CHECK: }
class HandleTestClass;
    int value;
endclass

// CHECK-LABEL: moore.module @ClassHandleComparisonTest() {
// CHECK:   [[H1:%.+]] = moore.variable : <!moore.class<@HandleTestClass>>
// CHECK:   [[H2:%.+]] = moore.variable : <!moore.class<@HandleTestClass>>
// CHECK:   [[RESULT:%.+]] = moore.variable : <!moore.i32>
// CHECK:   moore.procedure initial {
module ClassHandleComparisonTest;
    HandleTestClass h1;
    HandleTestClass h2;
    int result;

    initial begin
        h1 = new;
        h2 = new;
        // Test handle equality (==)
        // Note: Class handle comparison currently emits a warning and returns a constant.
        // This tests that the comparison is recognized and handled.
        // CHECK: moore.constant
        if (h1 == h2) begin
            result = 1;
        end
        // Test handle inequality (!=)
        // CHECK: moore.constant
        if (h1 != h2) begin
            result = 2;
        end
        // Test comparison with null
        // CHECK: moore.constant
        if (h1 == null) begin
            result = 0;
        end
        // CHECK: moore.constant
        if (h1 != null) begin
            result = 3;
        end
    end
endmodule

/// Test nested parameterized class specialization
/// This is common in UVM for type-parameterized pools and queues.

// CHECK-LABEL: moore.class.classdecl @GenericContainer {
// CHECK:   moore.class.propertydecl @data : !moore.i32
// CHECK: }
class GenericContainer #(type T=int);
    T data;
endclass

// CHECK-LABEL: moore.class.classdecl @SpecializedContainer extends @GenericContainer {
// CHECK:   moore.class.propertydecl @extra : !moore.l8
// CHECK: }
class SpecializedContainer extends GenericContainer#(int);
    logic [7:0] extra;
endclass

// CHECK-LABEL: moore.module @NestedSpecializationTest() {
module NestedSpecializationTest;
    // CHECK: moore.variable : <!moore.class<@GenericContainer>>
    GenericContainer#(int) generic_inst;
    // CHECK: moore.variable : <!moore.class<@SpecializedContainer>>
    SpecializedContainer specialized_inst;

    initial begin
        generic_inst = new;
        specialized_inst = new;
        // Access data through specialized instance (requires upcast)
        // CHECK: moore.class.upcast
        specialized_inst.data = 42;
    end
endmodule

/// Test UVM-style polymorphic class hierarchy
// CHECK-LABEL: moore.class.classdecl @uvm_object {
// CHECK: }
class uvm_object;
endclass

// CHECK-LABEL: moore.class.classdecl @uvm_component extends @uvm_object {
// CHECK: }
class uvm_component extends uvm_object;
endclass

// CHECK-LABEL: moore.class.classdecl @my_driver extends @uvm_component {
// CHECK: }
class my_driver extends uvm_component;
endclass

// CHECK-LABEL: moore.module @UVMHierarchyTest() {
module UVMHierarchyTest;
    // CHECK: moore.variable : <!moore.class<@my_driver>>
    my_driver drv;
    // CHECK: moore.variable : <!moore.class<@uvm_component>>
    uvm_component comp;

    initial begin
        drv = new;
        // Polymorphic assignment - derived to base
        // CHECK: moore.class.upcast
        comp = drv;
    end
endmodule
