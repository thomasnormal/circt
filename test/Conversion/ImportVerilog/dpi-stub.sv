// RUN: circt-verilog --ir-moore %s | FileCheck %s

// Test that DPI-C imports generate external function declarations
// that can be linked with C implementations later.

// CHECK-DAG: func.func private @my_c_func(!moore.i32) -> !moore.i32
import "DPI-C" function int my_c_func(int x);

// CHECK-DAG: func.func private @get_name() -> !moore.string
import "DPI-C" function string get_name();

// CHECK-DAG: func.func private @void_func(!moore.i32)
import "DPI-C" function void void_func(int x);

module test;
  initial begin
    int result;
    string name;
    // CHECK: func.call @my_c_func
    result = my_c_func(42);
    // CHECK: func.call @get_name
    name = get_name();
    // CHECK: func.call @void_func
    void_func(result);
  end
endmodule
