// RUN: circt-verilog --ir-moore %s 2>&1
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// UVM Class Method Edge Cases and Advanced Patterns
// Testing problematic patterns and edge cases
//===----------------------------------------------------------------------===//

/// Test 1: Multiple super calls in same method
class multi_level_base;
    int value;
    virtual function void init(int x);
        value = x;
    endfunction
endclass

class multi_level_mid extends multi_level_base;
    int mid_value;
    virtual function void init(int x);
        super.init(x);
        mid_value = x * 2;
    endfunction
endclass

class multi_level_top extends multi_level_mid;
    int top_value;
    virtual function void init(int x);
        super.init(x);
        top_value = x * 3;
    endfunction
endclass

/// Test 2: this. in nested function calls
class nested_this_calls;
    int a, b, c;

    function void method_a(int val);
        this.a = val;
        this.method_b(val + 1);
    endfunction

    function void method_b(int val);
        this.b = val;
        this.method_c(val + 1);
    endfunction

    function void method_c(int val);
        this.c = val;
    endfunction
endclass

/// Test 3: Parameterized class with virtual methods
class param_virtual #(type T = int);
    T data;

    virtual function void set(T value);
        data = value;
    endfunction

    virtual function T get();
        return data;
    endfunction
endclass

class param_virtual_derived #(type T = int) extends param_virtual#(T);
    virtual function void set(T value);
        super.set(value);
    endfunction
endclass

/// Test 4: Static method accessing static property
class static_complex;
    static int counter;
    static int max_count;

    static function bit check_limit();
        return counter < max_count;
    endfunction

    static function void safe_increment();
        if (check_limit()) begin
            counter++;
        end
    endfunction
endclass

/// Test 5: Virtual method returning class handle
class returnable_base;
    int id;
    virtual function returnable_base clone();
        returnable_base obj = new;
        obj.id = this.id;
        return obj;
    endfunction
endclass

class returnable_derived extends returnable_base;
    int extra;
    virtual function returnable_base clone();
        returnable_derived obj = new;
        obj.id = this.id;
        obj.extra = this.extra;
        return obj;
    endfunction
endclass

/// Test 6: Constructor calling virtual method (dangerous pattern)
class ctor_virtual_base;
    int value;

    function new();
        this.initialize();
    endfunction

    virtual function void initialize();
        value = 0;
    endfunction
endclass

class ctor_virtual_derived extends ctor_virtual_base;
    int derived_value;

    function new();
        super.new();
        derived_value = 100;
    endfunction

    virtual function void initialize();
        super.initialize();
        value = 42;
    endfunction
endclass

/// Test 7: Recursive method calls
class recursive_class;
    int depth;

    function int factorial(int n);
        if (n <= 1)
            return 1;
        else
            return n * this.factorial(n - 1);
    endfunction
endclass

/// Test 8: Method with multiple return points
class multi_return;
    int state;

    function int check_state();
        if (state == 0)
            return -1;
        else if (state < 10)
            return 0;
        else if (state < 100)
            return 1;
        else
            return 2;
    endfunction
endclass

/// Test 9: Polymorphism with type parameters
class poly_param_base #(type T);
    virtual function T transform(T inp);
        return inp;
    endfunction
endclass

class poly_param_derived #(type T) extends poly_param_base#(T);
    virtual function T transform(T inp);
        // In real code, would do something different
        return super.transform(inp);
    endfunction
endclass

/// Test 10: this. with array properties
class array_this;
    int arr[10];

    function void set_element(int idx, int val);
        this.arr[idx] = val;
    endfunction

    function int get_element(int idx);
        return this.arr[idx];
    endfunction
endclass

/// Test 11: Virtual method with default parameter values
class default_params;
    virtual function int add(int a, int b = 10);
        return a + b;
    endfunction
endclass

class default_params_derived extends default_params;
    virtual function int add(int a, int b = 20);
        return super.add(a, b) * 2;
    endfunction
endclass

/// Test 12: Method shadowing (non-virtual)
class method_shadow_base;
    function int compute();
        return 1;
    endfunction
endclass

class method_shadow_derived extends method_shadow_base;
    // Shadows base method (non-virtual)
    function int compute();
        return 2;
    endfunction
endclass

/// Test 13: Complex constructor with multiple super calls
class complex_ctor_base;
    int x;
    function new(int val);
        x = val;
    endfunction
endclass

class complex_ctor_derived extends complex_ctor_base;
    int y;
    function new(int val1, int val2);
        super.new(val1);
        y = val2;
        this.initialize();
    endfunction

    function void initialize();
        y = y * 2;
    endfunction
endclass

/// Test 14: this. in ternary expressions
class ternary_this;
    int a, b;
    bit sel;

    function int get_selected();
        return this.sel ? this.a : this.b;
    endfunction
endclass

/// Test 15: Virtual pure function (interface class pattern)
virtual class abstract_base;
    pure virtual function int compute();
endclass

class concrete_impl extends abstract_base;
    virtual function int compute();
        return 42;
    endfunction
endclass

/// Test 16: Static property in parameterized class
class static_param #(type T = int);
    static T shared_data;

    static function void set_shared(T val);
        shared_data = val;
    endfunction

    static function T get_shared();
        return shared_data;
    endfunction
endclass

/// Test 17: this. with dynamic array
class dynamic_array_this;
    int dyn_arr[];

    function void allocate(int size);
        this.dyn_arr = new[size];
    endfunction

    function void fill(int value);
        foreach(this.dyn_arr[i])
            this.dyn_arr[i] = value;
    endfunction
endclass

/// Test 18: Method returning this (fluent interface pattern)
class fluent_interface;
    int value;

    function fluent_interface set_value(int v);
        this.value = v;
        return this;
    endfunction

    function fluent_interface increment();
        this.value++;
        return this;
    endfunction
endclass

/// Test 19: super. in non-virtual method
class super_nonvirtual_base;
    function int compute();
        return 10;
    endfunction
endclass

class super_nonvirtual_derived extends super_nonvirtual_base;
    function int compute();
        return super.compute() + 5;
    endfunction
endclass

/// Test 20: Complex vtable scenario with multiple inheritance levels
class vtable_l1;
    virtual function void f1();
    endfunction
    virtual function void f2();
    endfunction
endclass

class vtable_l2 extends vtable_l1;
    virtual function void f1();
    endfunction
    virtual function void f3();
    endfunction
endclass

class vtable_l3 extends vtable_l2;
    virtual function void f2();
    endfunction
    virtual function void f3();
    endfunction
endclass

/// Test module to exercise vtable
module test_vtable;
    vtable_l1 base_handle;
    vtable_l3 derived_obj;

    initial begin
        derived_obj = new;
        base_handle = derived_obj;
        base_handle.f1();  // Should call vtable_l2::f1
        base_handle.f2();  // Should call vtable_l3::f2
    end
endmodule

/// Test 21: Method overloading (should fail in SV)
// SystemVerilog doesn't support method overloading
// class overload_test;
//     function int compute(int x);
//         return x;
//     endfunction
//     function int compute(int x, int y);  // ERROR
//         return x + y;
//     endfunction
// endclass

/// Test 22: Const methods (not supported in SV)
// class const_method_test;
//     int value;
//     function int get_value() const;  // ERROR: const not supported
//         return value;
//     endfunction
// endclass

/// Test 23: Accessing static member through instance
module test_static_via_instance;
    static_complex obj1, obj2;

    initial begin
        obj1 = new;
        // Note: In SV, can access static through instance (though not recommended)
        // This is a language quirk
        // obj1.counter = 5;  // Accesses static through instance
        static_complex::counter = 5;  // Proper way
    end
endmodule

/// Test 24: Nested class with methods
class outer_class;
    int outer_data;

    class inner_class;
        int inner_data;

        function void inner_method();
            inner_data = 99;
        endfunction
    endclass

    function void outer_method();
        outer_data = 88;
    endfunction
endclass

/// Test 25: Package-scoped class with methods
package test_pkg;
    class pkg_class;
        static int pkg_counter;

        static function void increment_pkg();
            pkg_counter++;
        endfunction
    endclass
endpackage

module test_package_class;
    initial begin
        test_pkg::pkg_class::increment_pkg();
    end
endmodule
