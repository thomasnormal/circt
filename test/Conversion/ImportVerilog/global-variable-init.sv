// RUN: circt-translate --import-verilog %s | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog --ir-hw %s | FileCheck %s --check-prefix=HW
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

//===----------------------------------------------------------------------===//
// Test: Global variable initialization with function call
//===----------------------------------------------------------------------===//
//
// Problem: When a package-level global variable has an initializer that calls
// a function, the initialization was being discarded during MooreToCore
// lowering. This caused the global to remain null/zero at runtime.
//
// Root cause: GlobalVariableOpConversion in MooreToCore.cpp was creating an
// LLVM global with zero initialization, completely ignoring the init region.
//
// Fix: Generate a global constructor function that executes the init region
// code at program startup and stores the result to the global. Register this
// constructor with llvm.mlir.global_ctors.
//===----------------------------------------------------------------------===//

package test_pkg;
  class my_class;
    int value;
    function new();
      value = 42;
    endfunction
  endclass

  function my_class create_instance();
    my_class obj;
    obj = new;
    return obj;
  endfunction

  // Package-level global variable with function call initializer
  // MOORE: moore.global_variable @"test_pkg::global_obj"
  // MOORE-SAME: init
  // MOORE: func.call @"test_pkg::create_instance"
  // MOORE: moore.yield
  my_class global_obj = create_instance();
endpackage

// HW: llvm.mlir.global internal @"test_pkg::global_obj"
// HW: llvm.mlir.global_ctors ctors = [@"__moore_global_init_test_pkg::global_obj"]
// HW: llvm.func internal @"__moore_global_init_test_pkg::global_obj"()
// HW: func.call @"test_pkg::create_instance"
// HW: llvm.store

module test;
  import test_pkg::*;

  initial begin
    if (global_obj != null)
      $display("SUCCESS: global_obj initialized, value = %0d", global_obj.value);
    else
      $display("FAIL: global_obj is null");
  end
endmodule
