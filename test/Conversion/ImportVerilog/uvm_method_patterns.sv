// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// UVM Class Method Pattern Tests
// Testing: virtual methods, super., this., polymorphism, new(), static methods
//===----------------------------------------------------------------------===//

/// Test 1: Virtual methods with basic inheritance
// CHECK-LABEL: moore.class.classdecl @base_class {

class base_class;
    virtual function void do_something();
        // Base implementation
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @derived_class extends @base_class {

class derived_class extends base_class;
    virtual function void do_something();
        // Overridden implementation
    endfunction
endclass

/// Test 2: Parameterized classes
// CHECK-LABEL: moore.class.classdecl @parameterized_class {
// CHECK:   moore.class.propertydecl @data : !moore.i32

class parameterized_class #(type T = int);
    T data;
endclass

// Module to instantiate parameterized_class
module test_parameterized;
    parameterized_class#(int) obj;
endmodule

/// Test 3: super. calls
// CHECK-LABEL: moore.class.classdecl @base_with_method {

class base_with_method;
    int value;

    virtual function void set_value(int v);
        value = v;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @derived_with_super extends @base_with_method {

class derived_with_super extends base_with_method;
    virtual function void set_value(int v);
        super.set_value(v * 2);  // Call base class method
    endfunction
endclass

/// Test 4: this. references
// CHECK-LABEL: moore.class.classdecl @class_with_this {

class class_with_this;
    int data;

    function void set_data(int data);
        this.data = data;  // Disambiguate member from parameter
    endfunction

    function int get_data();
        return this.data;
    endfunction
endclass

/// Test 5: Polymorphism - virtual method dispatch
// CHECK-LABEL: moore.module @test_polymorphism

module test_polymorphism;
    base_class handle;
    derived_class obj;

    initial begin
        obj = new;
        handle = obj;  // Upcast
        // CHECK: moore.vtable.load_method
        handle.do_something();  // Should call derived implementation
    end
endmodule

/// Test 6: Constructor chaining with super.new()
// CHECK-LABEL: moore.class.classdecl @base_with_ctor {

class base_with_ctor;
    int x;
    function new(int val);
        x = val;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @derived_with_ctor extends @base_with_ctor {

class derived_with_ctor extends base_with_ctor;
    int y;
    function new(int val1, int val2);
        super.new(val1);
        y = val2;
    endfunction
endclass

/// Test 7: Static methods
// CHECK-LABEL: moore.class.classdecl @class_with_static {

class class_with_static;
    static int shared_counter;

    static function int get_counter();
        return shared_counter;
    endfunction

    static function void increment();
        shared_counter++;
    endfunction
endclass

/// Test 8: Static method calls from module
// CHECK-LABEL: moore.module @test_static_method

module test_static_method;
    int result;

    initial begin
        class_with_static::increment();
        result = class_with_static::get_counter();
    end
endmodule

/// Test 9: Complex polymorphism with multiple levels
// CHECK-LABEL: moore.class.classdecl @level1_base {

class level1_base;
    virtual function int compute(int x);
        return x;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @level2_middle extends @level1_base {

class level2_middle extends level1_base;
    virtual function int compute(int x);
        return super.compute(x) * 2;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @level3_derived extends @level2_middle {

class level3_derived extends level2_middle;
    virtual function int compute(int x);
        return super.compute(x) + 10;
    endfunction
endclass

/// Test 10: UVM factory pattern simulation
// CHECK-LABEL: moore.class.classdecl @uvm_test {

class uvm_test;
    static string type_name = "uvm_test";

    static function uvm_test create(string name);
        uvm_test obj = new;
        return obj;
    endfunction
endclass

// CHECK-LABEL: moore.class.classdecl @my_test extends @uvm_test {

class my_test extends uvm_test;
    // In real UVM, `uvm_component_utils would generate type registration
    // We test basic inheritance pattern here
endclass

/// Test 11: Mixed virtual and non-virtual methods
// CHECK-LABEL: moore.class.classdecl @mixed_methods {

class mixed_methods;
    int data;

    // Non-virtual method (direct call)
    function void concrete_method();
        data = 1;
    endfunction

    // Virtual method (indirect call)
    virtual function void virtual_method();
        data = 2;
    endfunction
endclass

/// Test 12: this. in complex expressions
// CHECK-LABEL: moore.class.classdecl @complex_this {

class complex_this;
    int x, y;

    function void update(int x, int y);
        this.x = x + this.y;
        this.y = y + this.x;
    endfunction

    function int sum();
        return this.x + this.y;
    endfunction
endclass

/// Test 13: Parameterized class with methods
// CHECK-LABEL: moore.class.classdecl @generic_container {

class generic_container #(type T = int);
    T data;

    function void set(T value);
        data = value;
    endfunction

    function T get();
        return data;
    endfunction
endclass

// Module to instantiate generic_container
module test_generic;
    generic_container#(int) obj;
endmodule

/// Test 14: Static property access in instance method
// CHECK-LABEL: moore.class.classdecl @mixed_static_instance {

class mixed_static_instance;
    static int class_count;
    int instance_id;

    function new();
        instance_id = class_count;
        class_count++;
    endfunction

    function int get_id();
        return instance_id;
    endfunction
endclass

/// Test 15: Virtual method with this. reference
// CHECK-LABEL: moore.class.classdecl @virtual_with_this {

class virtual_with_this;
    int value;

    virtual function void process();
        this.value = this.compute();
    endfunction

    virtual function int compute();
        return this.value * 2;
    endfunction
endclass

/// Test 16: super. with multiple inheritance levels
// CHECK-LABEL: moore.module @test_super_chain

module test_super_chain;
    level3_derived obj;
    int result;

    initial begin
        obj = new;
        // Should call: level3(level2(level1(5) * 2) + 10)
        // = level3((5 * 2) + 10) = level3(20) = 20
        result = obj.compute(5);
    end
endmodule

/// Test 17: Constructor without parameters
// CHECK-LABEL: moore.class.classdecl @simple_ctor {

class simple_ctor;
    int x;

    function new();
        x = 42;
    endfunction
endclass

/// Test 18: Multiple constructors (should fail - SV doesn't support overloading)
// This test documents expected behavior

/// Test 19: Protected methods
// CHECK-LABEL: moore.class.classdecl @with_protected {

class with_protected;
    protected int data;

    protected function void internal_process();
        data = 100;
    endfunction

    function void public_interface();
        internal_process();
    endfunction
endclass

/// Test 20: Local (private) methods
// CHECK-LABEL: moore.class.classdecl @with_local {

class with_local;
    local int secret;

    local function void private_method();
        secret = 99;
    endfunction

    function void use_private();
        private_method();
    endfunction
endclass

/// Test 21: Extern method declarations
// CHECK-LABEL: moore.class.classdecl @class_with_extern {

class class_with_extern;
    int data;

    // Extern method declaration
    extern function void set_data(int value);
    extern function int get_data();

    // Extern virtual method
    extern virtual function int compute(int x);
endclass

// Extern method implementations
function void class_with_extern::set_data(int value);
    data = value;
endfunction

function int class_with_extern::get_data();
    return data;
endfunction

function int class_with_extern::compute(int x);
    return x * data;
endfunction

// CHECK-LABEL: moore.module @test_extern
module test_extern;
    initial begin
        class_with_extern obj;
        int result;
        obj = new;
        obj.set_data(10);
        result = obj.get_data();
        result = obj.compute(5);
    end
endmodule
