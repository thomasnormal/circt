// RUN: circt-verilog --parse-only --uvm-path=%S/../../../lib/Runtime/uvm %s
// RUN: circt-verilog --ir-moore --uvm-path=%S/../../../lib/Runtime/uvm %s 2>&1 | FileCheck %s --check-prefix=MOORE
// REQUIRES: slang
// XFAIL: *

//===----------------------------------------------------------------------===//
// Test uvm_get_report_object global function
//===----------------------------------------------------------------------===//
//
// This test verifies that the global uvm_get_report_object() function works
// correctly in CIRCT's UVM stubs. This function is commonly used in UVM macros
// like `uvm_info, `uvm_warning, etc.
//
// In the real UVM library, uvm_get_report_object() can trigger recursive
// function calls through uvm_coreservice_t::get() -> uvm_init(). CIRCT's
// UVM stubs provide a non-recursive implementation that returns uvm_root.
//
//===----------------------------------------------------------------------===//

`timescale 1ns/1ps

`include "uvm_macros.svh"

// MOORE: module {

module test_get_report_object;
  import uvm_pkg::*;

  initial begin
    uvm_report_object ro;
    int verbosity;

    // Test 1: Call the global uvm_get_report_object function
    ro = uvm_get_report_object();

    // Test 2: Call the method version on the returned object
    ro = ro.uvm_get_report_object();

    // Test 3: Get verbosity level through the report object
    verbosity = ro.get_report_verbosity_level();

    // Test 4: Use in UVM macros (these expand to use uvm_get_report_object)
    `uvm_info("TEST", "Testing uvm_get_report_object", UVM_LOW)

    $display("Verbosity level: %0d", verbosity);
  end
endmodule

// MOORE-DAG: func.func private @"uvm_pkg::uvm_get_report_object"()
// MOORE-DAG: func.func private @"uvm_pkg::uvm_report_object::uvm_get_report_object"
// MOORE-DAG: moore.module @test_get_report_object()
