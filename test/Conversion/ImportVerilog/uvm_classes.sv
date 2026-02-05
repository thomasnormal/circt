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

// CHECK-DAG: moore.class.classdecl @uvm_pool {
// CHECK-DAG: moore.class.classdecl @uvm_pool_0 {
class uvm_pool #(type KEY=int, type T=int);
endclass

// Note: The exact suffix (_0, _1, etc.) assigned to each specialization depends
// on the order of class processing, which may vary. Use regex to match either.
// CHECK-DAG: moore.class.classdecl @uvm_pool_int_string extends @uvm_pool{{(_0)?}} {
class uvm_pool_int_string extends uvm_pool#(int, string);
endclass

// Note: Different specialization of uvm_pool creates a different base class
// CHECK-DAG: moore.class.classdecl @uvm_pool_string_int extends @uvm_pool{{(_0)?}} {
class uvm_pool_string_int extends uvm_pool#(string, int);
endclass

// Test that multiple specializations are recognized
// CHECK-LABEL: moore.module @GenericClassSpecializationTest() {
module GenericClassSpecializationTest;
    // CHECK: moore.variable : <class<@uvm_pool_int_string>>
    uvm_pool_int_string pool1;
    // CHECK: moore.variable : <class<@uvm_pool_string_int>>
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
// CHECK:   [[H1:%.+]] = moore.variable : <class<@HandleTestClass>>
// CHECK:   [[H2:%.+]] = moore.variable : <class<@HandleTestClass>>
// CHECK:   [[RESULT:%.+]] = moore.variable : <i32>
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
    // CHECK: moore.variable : <class<@GenericContainer>>
    GenericContainer#(int) generic_inst;
    // CHECK: moore.variable : <class<@SpecializedContainer>>
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
    // CHECK: moore.variable : <class<@my_driver>>
    my_driver drv;
    // CHECK: moore.variable : <class<@uvm_component>>
    uvm_component comp;

    initial begin
        drv = new;
        // Polymorphic assignment - derived to base
        // CHECK: moore.class.upcast
        comp = drv;
    end
endmodule

//===----------------------------------------------------------------------===//
// Test parameterized class this_type pattern (common in UVM)
//===----------------------------------------------------------------------===//

/// Test the UVM this_type pattern where a parameterized class uses a typedef
/// for self-referential return types. This pattern is extremely common in UVM:
///
/// class uvm_pool #(type KEY=int, T=uvm_void) extends uvm_object;
///   typedef uvm_pool #(KEY,T) this_type;
///   virtual function this_type get_global_pool();
///     return ...;
///   endfunction
/// endclass

class uvm_void;
endclass

class uvm_object_base;
endclass

// CHECK-LABEL: moore.class.classdecl @this_type_pool extends @uvm_object_base {
// CHECK:   moore.class.methoddecl @get_global_pool -> @"this_type_pool::get_global_pool"
// CHECK: }
class this_type_pool #(type KEY=int, type T=uvm_void) extends uvm_object_base;
    typedef this_type_pool #(KEY, T) this_type;

    // Static pool instance
    static this_type m_global;

    // Virtual function returning this_type
    // This pattern previously caused the error:
    // "receiver class @"this_type_pool_N" is not the same as, or derived from,
    // expected base class @this_type_pool"
    virtual function this_type get_global_pool();
        if (m_global == null)
            m_global = new;
        return m_global;
    endfunction
endclass

// A second specialization of this_type_pool is created for the this_type typedef.
// The exact suffix (_0, _1, etc.) depends on the order of class processing.
// Static member m_global is a global variable, not a class property.
// CHECK-LABEL: moore.class.classdecl @this_type_pool_{{[0-9]+}} extends @uvm_object_base {
// CHECK:   moore.class.methoddecl @get_global_pool
// CHECK: }

// CHECK-LABEL: moore.module @ThisTypePatternTest() {
// CHECK:   %pool_handle = moore.variable : <class<@this_type_pool{{(_[0-9]+)?}}>>
// CHECK:   moore.procedure initial {
// CHECK:     %[[HANDLE:.*]] = moore.read %pool_handle
// CHECK:     %[[METHOD:.*]] = moore.vtable.load_method %[[HANDLE]] : @get_global_pool of <@this_type_pool[[SRC_SUFFIX:(_[0-9]+)?]]> -> (!moore.class<@this_type_pool[[SRC_SUFFIX]]>) -> !moore.class<@this_type_pool[[RET_SUFFIX:(_[0-9]+)?]]>
// CHECK:     %[[RESULT:.*]] = func.call_indirect %[[METHOD]](%[[HANDLE]]) : (!moore.class<@this_type_pool[[SRC_SUFFIX]]>) -> !moore.class<@this_type_pool[[RET_SUFFIX]]>
// The key fix: conversion between this_type_pool specializations is allowed.
// CHECK:     %[[CONV:.*]] = moore.conversion %[[RESULT]] : !moore.class<@this_type_pool[[RET_SUFFIX]]> -> !moore.class<@this_type_pool[[SRC_SUFFIX]]>
// CHECK:     moore.blocking_assign %pool_handle, %[[CONV]]
// CHECK:   }
// CHECK: }
module ThisTypePatternTest;
    // Use the default specialization
    this_type_pool#(int, uvm_void) pool_handle;

    initial begin
        // Call the virtual function returning this_type
        pool_handle = pool_handle.get_global_pool();
    end
endmodule
