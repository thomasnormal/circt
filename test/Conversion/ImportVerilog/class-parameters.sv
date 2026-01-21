// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s
// Tests for class parameter handling

// Test 1: Simple parameterized class (parameter is not a property)
// The parameter value should be inlined/constant-folded at use sites.
// CHECK: moore.module @test_simple_param
module test_simple_param;
  class test_cls #(parameter a = 12);
  endclass

  test_cls #(34) test_obj;

  initial begin
    // TODO: Class parameters accessed via instance should be treated as constants
    // $display("a = %d", test_obj.a);  // This currently fails MLIR verification
  end
endmodule

// Test 2: Parameterized class with type parameter
// CHECK: moore.module @test_type_param
module test_type_param;
  class my_class #(type T = int, int WIDTH = 8);
    T data;
  endclass

  my_class #(bit, 16) inst;

  initial begin
    inst = new;
    inst.data = 1;
  end
endmodule

// Test 3: Parameterized class extending another parameterized class
// CHECK: moore.module @test_param_extend
module test_param_extend;
  class base_cls #(int b = 20);
    int a;
  endclass

  class ext_cls #(int e = 25) extends base_cls #(5);
    int c;
  endclass

  ext_cls #(15) inst;

  initial begin
    inst = new;
  end
endmodule
