// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI-C imports are skipped with a remark, not a crash or hard error.
// DPI-C imports are not yet supported but should not block compilation.
// Instead of calling the DPI functions, meaningful default values are returned:
// - int types: return 0
// - string types: return empty string
// - void functions: no-op

// DPI-C function declaration at package level
import "DPI-C" function int c_add(int a, int b);

// DPI-C function without arguments
import "DPI-C" function void c_void_func();

// DPI-C function with return value
import "DPI-C" function string c_get_string();

// CHECK: remark: DPI-C imports not yet supported; call to 'c_add' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_void_func' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_get_string' skipped

// CHECK: moore.module @DPITest
module DPITest;
  initial begin
    int result;
    string s;
    result = c_add(1, 2);
    c_void_func();
    s = c_get_string();
    // Use values to prevent DCE
    $display("result=%d s=%s", result, s);
  end
endmodule

// Check stub values are generated:
// CHECK: moore.constant_string "" : i8
// CHECK: moore.int_to_string
