// RUN: circt-translate --import-verilog %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test that DPI-C imports are skipped with a remark, not a crash or hard error.
// DPI-C imports are not yet supported but should not block compilation.

// DPI-C function declaration at package level
import "DPI-C" function int c_add(int a, int b);

// DPI-C function without arguments
import "DPI-C" function void c_void_func();

// DPI-C function with return value
import "DPI-C" function string c_get_string();

// CHECK: remark: DPI-C imports not yet supported; call to 'c_add' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_void_func' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_get_string' skipped

// The DPI-C function declarations should still be emitted as external functions.
// CHECK: func.func private @c_add(!moore.i32, !moore.i32) -> !moore.i32
// CHECK: func.func private @c_void_func()
// CHECK: func.func private @c_get_string() -> !moore.string

// CHECK: moore.module @DPITest
module DPITest;
  initial begin
    int result;
    string s;
    result = c_add(1, 2);
    c_void_func();
    s = c_get_string();
  end
endmodule
