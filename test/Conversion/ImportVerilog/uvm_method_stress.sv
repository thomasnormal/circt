// RUN: circt-verilog --ir-moore %s 2>&1
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// UVM Class Method Stress Tests
// Tests corner cases and potential failure modes
//===----------------------------------------------------------------------===//

/// Test 1: Very deep inheritance (5 levels)
class l1_base;
    virtual function int f();
        return 1;
    endfunction
endclass

class l2_d1 extends l1_base;
    virtual function int f();
        return super.f() + 1;
    endfunction
endclass

class l3_d2 extends l2_d1;
    virtual function int f();
        return super.f() + 1;
    endfunction
endclass

class l4_d3 extends l3_d2;
    virtual function int f();
        return super.f() + 1;
    endfunction
endclass

class l5_d4 extends l4_d3;
    virtual function int f();
        return super.f() + 1;
    endfunction
endclass

/// Test 2: Multiple virtual methods in same class
class multi_virtual;
    virtual function void a();
    endfunction
    virtual function void b();
    endfunction
    virtual function void c();
    endfunction
    virtual function void d();
    endfunction
    virtual function void e();
    endfunction
endclass

/// Test 3: Mix of virtual and non-virtual
class mixed_virtual;
    function void concrete1();
    endfunction

    virtual function void virtual1();
    endfunction

    function void concrete2();
    endfunction

    virtual function void virtual2();
    endfunction
endclass

/// Test 4: this. with complex expressions
class complex_this_expr;
    int a, b, c;
    int arr[10];

    function int evaluate();
        return (this.a + this.b) * this.c;
    endfunction

    function void assign_chain();
        this.a = 5;
        this.b = 5;
        this.c = 5;
    endfunction

    function int indexed_this();
        return this.arr[this.a];
    endfunction
endclass

/// Test 5: super. through multiple levels
class super_chain_base;
    int value;
    virtual function void set(int v);
        value = v;
    endfunction
endclass

class super_chain_mid extends super_chain_base;
    virtual function void set(int v);
        super.set(v * 2);
    endfunction
endclass

class super_chain_top extends super_chain_mid;
    virtual function void set(int v);
        super.set(v + 10);
    endfunction
endclass

/// Test 6: Calling other virtual methods from virtual method
class virtual_calls_virtual;
    virtual function int helper();
        return 42;
    endfunction

    virtual function int main_func();
        return this.helper() * 2;
    endfunction
endclass

class virtual_calls_virtual_d extends virtual_calls_virtual;
    virtual function int helper();
        return 100;
    endfunction

    virtual function int main_func();
        return this.helper() + super.main_func();
    endfunction
endclass

/// Test 7: Static method calling static method
class static_chain;
    static int counter;

    static function void increment();
        counter++;
    endfunction

    static function void increment_twice();
        increment();
        increment();
    endfunction

    static function int get_value();
        return counter;
    endfunction
endclass

/// Test 8: Parameterized class with multiple type parameters
class multi_param #(type T1 = int, type T2 = bit, type T3 = logic);
    T1 data1;
    T2 data2;
    T3 data3;

    function void set_all(T1 d1, T2 d2, T3 d3);
        this.data1 = d1;
        this.data2 = d2;
        this.data3 = d3;
    endfunction
endclass

/// Test 9: Parameterized virtual methods
class param_virtual #(type T);
    T data;

    virtual function void set(T val);
        this.data = val;
    endfunction

    virtual function T get();
        return this.data;
    endfunction

    virtual function T transform(T val);
        return val;
    endfunction
endclass

class param_virtual_derived #(type T) extends param_virtual#(T);
    virtual function void set(T val);
        super.set(val);
    endfunction

    virtual function T transform(T val);
        return super.transform(val);
    endfunction
endclass

/// Test 10: Constructor with complex initialization
class complex_ctor;
    int a, b, c;
    int sum;

    function new(int x, int y, int z);
        this.a = x;
        this.b = y;
        this.c = z;
        this.sum = this.a + this.b + this.c;
    endfunction
endclass

/// Test 11: Diamond problem simulation (through interfaces)
interface class interface1;
    pure virtual function void method1();
endclass

interface class interface2;
    pure virtual function void method2();
endclass

class diamond_impl implements interface1, interface2;
    virtual function void method1();
    endfunction

    virtual function void method2();
    endfunction
endclass

/// Test 12: Method with many parameters
class many_params;
    function int compute(int a, int b, int c, int d, int e, int f);
        return a + b + c + d + e + f;
    endfunction
endclass

/// Test 13: Method returning large struct (package array)
class large_return;
    typedef struct {
        int a;
        int b;
        int c;
        int d;
    } my_struct;

    function my_struct make_struct(int val);
        my_struct s;
        s.a = val;
        s.b = val * 2;
        s.c = val * 3;
        s.d = val * 4;
        return s;
    endfunction
endclass

/// Test 14: Void function with side effects
class side_effects;
    int state;

    function void modify_state();
        this.state++;
    endfunction

    function void chain_modifications();
        this.modify_state();
        this.modify_state();
        this.modify_state();
    endfunction
endclass

/// Test 15: Virtual method in constructor
class ctor_calls_virtual;
    int value;

    function new();
        this.initialize();
    endfunction

    virtual function void initialize();
        this.value = 0;
    endfunction
endclass

class ctor_calls_virtual_d extends ctor_calls_virtual;
    function new();
        super.new();
    endfunction

    virtual function void initialize();
        this.value = 42;
    endfunction
endclass

/// Test 16: Nested parameterized classes
class outer_param #(type T);
    T data;

    class inner_param #(type U);
        U inner_data;

        function void set_inner(U val);
            this.inner_data = val;
        endfunction
    endclass
endclass

/// Test 17: Class with both static and instance methods accessing same property
class mixed_access;
    static int shared;
    int inst_value;

    static function void set_shared(int val);
        shared = val;
    endfunction

    function void set_instance(int val);
        this.inst_value = val;
    endfunction

    function int get_both();
        return shared + this.inst_value;
    endfunction
endclass

/// Test 18: Multiple methods with same name in hierarchy (shadowing)
class shadow_base;
    function int compute();
        return 1;
    endfunction
endclass

class shadow_mid extends shadow_base;
    function int compute();
        return 2;
    endfunction
endclass

class shadow_top extends shadow_mid;
    function int compute();
        return 3;
    endfunction
endclass

/// Test 19: Task methods (non-blocking)
class with_tasks;
    int counter;

    task increment_delayed();
        counter++;
    endtask

    task run();
        increment_delayed();
    endtask
endclass

/// Test 20: Abstract class with mix of pure and concrete
virtual class abstract_mix;
    int concrete_data;

    pure virtual function int abstract_method();

    virtual function void concrete_method();
        concrete_data = 0;
    endfunction
endclass

class concrete_mix extends abstract_mix;
    virtual function int abstract_method();
        return this.concrete_data;
    endfunction

    virtual function void concrete_method();
        super.concrete_method();
        concrete_data = 100;
    endfunction
endclass

/// Module to exercise stress tests
module stress_test;
    l5_d4 deep_obj;
    multi_virtual mv_obj;
    super_chain_top chain_obj;
    virtual_calls_virtual_d vcv_obj;
    param_virtual_derived#(int) pv_obj;
    diamond_impl di_obj;
    complex_ctor cc_obj;

    int result;

    initial begin
        // Test deep hierarchy
        deep_obj = new;
        result = deep_obj.f();  // Should return 5

        // Test multiple virtuals
        mv_obj = new;
        mv_obj.a();
        mv_obj.e();

        // Test super chain
        chain_obj = new;
        chain_obj.set(5);

        // Test virtual calling virtual
        vcv_obj = new;
        result = vcv_obj.main_func();

        // Test parameterized virtual
        pv_obj = new;
        pv_obj.set(42);
        result = pv_obj.get();

        // Test diamond
        di_obj = new;
        di_obj.method1();
        di_obj.method2();

        // Test complex constructor
        cc_obj = new(1, 2, 3);
    end
endmodule
