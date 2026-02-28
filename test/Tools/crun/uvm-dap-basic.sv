// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test UVM Data Access Policy (DAP) classes.
// Verifies set_before_get_dap and get_to_lock_dap behavior.

// CHECK: [TEST] set_before_get: PASS
// CHECK: [TEST] get_to_lock: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class dap_test extends uvm_test;
    `uvm_component_utils(dap_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_set_before_get_dap #(int) sbg;
      uvm_get_to_lock_dap #(int) gtl;
      int val;

      phase.raise_objection(this);

      // Test 1: set_before_get_dap — set then get
      sbg = new("sbg");
      sbg.set(42);
      val = sbg.get();
      if (val == 42)
        `uvm_info("TEST", "set_before_get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("set_before_get: FAIL (got %0d)", val))

      // Test 2: get_to_lock_dap — set, get (locks), set again (should be locked)
      gtl = new("gtl");
      gtl.set(10);
      val = gtl.get();  // This should lock it
      gtl.set(20);      // This should be ignored or error (locked)
      val = gtl.get();
      if (val == 10)
        `uvm_info("TEST", "get_to_lock: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("get_to_lock: FAIL (got %0d, expected 10)", val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("dap_test");
endmodule
