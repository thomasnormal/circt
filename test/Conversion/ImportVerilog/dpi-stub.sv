// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI-C imports return meaningful stub values for UVM compatibility.
// DPI-C calls are not yet implemented but return appropriate defaults:
// - int types: return 0
// - string types: return empty string
// - void functions: no-op

import "DPI-C" function int my_c_func(int x);
import "DPI-C" function string get_name();
import "DPI-C" function void void_func(int x);

module test;
  initial begin
    int result;
    string name;
    int input_val;
    input_val = 42;
    // CHECK: remark: DPI-C imports not yet supported; call to 'my_c_func' skipped
    result = my_c_func(input_val);
    // CHECK: remark: DPI-C imports not yet supported; call to 'get_name' skipped
    name = get_name();
    // CHECK: remark: DPI-C imports not yet supported; call to 'void_func' skipped
    void_func(result);
    // Use the values to prevent optimization
    $display("result=%d name=%s", result, name);
  end
endmodule

// Check that stub values are returned (empty string for get_name):
// CHECK: moore.constant_string "" : i8
// CHECK: moore.int_to_string
