// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI-C imports are lowered to runtime stub calls.

import "DPI-C" function int my_c_func(int x);
import "DPI-C" function string get_name();
import "DPI-C" function void void_func(int x);

module test;
  initial begin
    int result;
    string name;
    int input_val;
    input_val = 42;
    // CHECK: remark: DPI-C import 'my_c_func' will use runtime stub (link with MooreRuntime)
    result = my_c_func(input_val);
    // CHECK: remark: DPI-C import 'get_name' will use runtime stub (link with MooreRuntime)
    name = get_name();
    // CHECK: remark: DPI-C import 'void_func' will use runtime stub (link with MooreRuntime)
    void_func(result);
    // CHECK: func.call @my_c_func
    // CHECK: func.call @get_name
    // CHECK: func.call @void_func
    // Use the values to prevent optimization
    $display("result=%d name=%s", result, name);
  end
endmodule
