// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI-C imports are skipped with a remark, not a crash or hard error.
// DPI-C imports are not yet supported but should not block compilation.
// Instead of calling the DPI functions, meaningful default values are returned:
// - int types: return 0
// - string types: return empty string
// - chandle types: return null (0 converted to chandle)
// - void functions: no-op

// DPI-C function declaration at package level
import "DPI-C" function int c_add(int a, int b);

// DPI-C function without arguments
import "DPI-C" function void c_void_func();

// DPI-C function with return value
import "DPI-C" function string c_get_string();

// DPI-C function returning chandle (opaque C pointer)
// This is commonly used by UVM for regex compilation (uvm_re_comp)
import "DPI-C" function chandle c_get_handle();

// CHECK: remark: DPI-C imports not yet supported; call to 'c_add' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_void_func' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_get_string' skipped
// CHECK: remark: DPI-C imports not yet supported; call to 'c_get_handle' skipped

// CHECK: moore.module @DPITest
// Check stub values are generated (constants may be hoisted to module scope):
// CHECK: moore.constant 0 : i64
// CHECK: moore.constant 0 : i32
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

// Check string stub generation inside procedure:
// CHECK: moore.constant_string "" : i8
// CHECK: moore.int_to_string
// CHECK: moore.conversion %{{.*}} : !moore.i64 -> !moore.chandle
