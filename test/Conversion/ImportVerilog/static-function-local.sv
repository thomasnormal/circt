// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// Test that static function-local variables with EXPLICIT 'static' keyword
// are converted to global variables. These persist across function calls,
// unlike automatic (default) variables which are re-initialized on each call.

package test_pkg;
  // Function with a static local variable - should become a global variable.
  // The global variable name includes the function name to ensure uniqueness.
  // CHECK-LABEL: func.func private @"test_pkg::get_counter"()
  // CHECK: moore.get_global_variable @"test_pkg::get_counter::count" : <i32>
  function automatic int get_counter();
    static int count = 0;
    count = count + 1;
    return count;
  endfunction

  // CHECK: moore.global_variable @"test_pkg::get_counter::count" : !moore.i32 init {
  // CHECK:   %[[ZERO:.*]] = moore.constant 0 : i32
  // CHECK:   moore.yield %[[ZERO]]
  // CHECK: }

  // Function with automatic (default) local variable - should remain local.
  // CHECK-LABEL: func.func private @"test_pkg::get_value"()
  // CHECK: %value = moore.variable
  function automatic int get_value();
    int value = 42;  // automatic variable (default in automatic function)
    return value;
  endfunction

  // Static function with explicit static local variable.
  // Note: Only variables with EXPLICIT 'static' keyword become globals.
  // CHECK-LABEL: func.func private @"test_pkg::static_func"()
  // CHECK: moore.get_global_variable @"test_pkg::static_func::static_var" : <i32>
  function static int static_func();
    static int static_var = 10;
    return static_var;
  endfunction

  // CHECK: moore.global_variable @"test_pkg::static_func::static_var" : !moore.i32 init {
  // CHECK:   %[[TEN:.*]] = moore.constant 10 : i32
  // CHECK:   moore.yield %[[TEN]]
  // CHECK: }

  // Static function with implicit static local variable (no 'static' keyword).
  // This should NOT become a global variable because there's no explicit
  // 'static' keyword on the variable declaration.
  // CHECK-LABEL: func.func private @"test_pkg::implicit_static_func"()
  // CHECK: %x = moore.variable
  function static int implicit_static_func();
    int x = 5;  // Implicitly static, but no explicit 'static' keyword
    return x;
  endfunction
endpackage

// Test module that uses the static function-local variable pattern similar to
// UVM's uvm_component_registry::get() singleton pattern.
module test_static_local;
  import test_pkg::*;

  initial begin
    int a, b, c;
    a = get_counter();  // Should return 1
    b = get_counter();  // Should return 2 (static count persists)
    c = get_counter();  // Should return 3
  end
endmodule
