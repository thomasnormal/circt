// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI-C imports are lowered to runtime stub calls with remarks.

// DPI-C function declaration at package level
import "DPI-C" function int c_add(int a, int b);

// DPI-C function without arguments
import "DPI-C" function void c_void_func();

// DPI-C function with return value
import "DPI-C" function string c_get_string();

// DPI-C function returning chandle (opaque C pointer)
// This is commonly used by UVM for regex compilation (uvm_re_comp)
import "DPI-C" function chandle c_get_handle();

// CHECK: remark: DPI-C import 'c_add' will use runtime stub (link with MooreRuntime)
// CHECK: remark: DPI-C import 'c_void_func' will use runtime stub (link with MooreRuntime)
// CHECK: remark: DPI-C import 'c_get_string' will use runtime stub (link with MooreRuntime)
// CHECK: remark: DPI-C import 'c_get_handle' will use runtime stub (link with MooreRuntime)

// CHECK: moore.module @DPITest
// Check DPI calls are emitted:
// CHECK: func.call @c_add
// CHECK: func.call @c_void_func
// CHECK: func.call @c_get_string
// CHECK: func.call @c_get_handle
module DPITest;
  initial begin
    int result;
    string s;
    chandle h;
    result = c_add(1, 2);
    c_void_func();
    s = c_get_string();
    h = c_get_handle();
    // Use values to prevent DCE
    $display("result=%d s=%s h=%p", result, s, h);
  end
endmodule
