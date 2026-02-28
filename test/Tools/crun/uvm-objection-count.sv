// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test objection counting: per-component and total.
// Verifies raise/drop/get_objection_count/get_objection_total.

// CHECK: [TEST] per-component count: PASS
// CHECK: [TEST] total count: PASS
// CHECK: [TEST] drop reduces count: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class obj_count_test extends uvm_test;
    `uvm_component_utils(obj_count_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_objection obj;
      phase.raise_objection(this);

      obj = new("test_obj");

      // Raise multiple times
      obj.raise_objection(this, "", 2);
      obj.raise_objection(this, "", 1);

      // Test 1: per-component count
      if (obj.get_objection_count(this) == 3)
        `uvm_info("TEST", "per-component count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("per-component count: FAIL got %0d", obj.get_objection_count(this)))

      // Test 2: total count
      if (obj.get_objection_total() == 3)
        `uvm_info("TEST", "total count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("total count: FAIL got %0d", obj.get_objection_total()))

      // Test 3: drop
      obj.drop_objection(this, "", 1);
      if (obj.get_objection_count(this) == 2)
        `uvm_info("TEST", "drop reduces count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("drop reduces count: FAIL got %0d", obj.get_objection_count(this)))

      // Clean up remaining objections
      obj.drop_objection(this, "", 2);

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("obj_count_test");
endmodule
