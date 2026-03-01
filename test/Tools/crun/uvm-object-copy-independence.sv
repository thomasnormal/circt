// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test object copy independence: modify original, copy unchanged and vice versa.

// CHECK: [TEST] copy unchanged after orig modified: PASS
// CHECK: [TEST] orig unchanged after copy modified: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_copy_obj extends uvm_object;
    int x;
    int y;

    `uvm_object_utils_begin(edge_copy_obj)
      `uvm_field_int(x, UVM_ALL_ON)
      `uvm_field_int(y, UVM_ALL_ON)
    `uvm_object_utils_end

    function new(string name = "edge_copy_obj");
      super.new(name);
    endfunction
  endclass

  class edge_copy_indep_test extends uvm_test;
    `uvm_component_utils(edge_copy_indep_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_copy_obj orig, copied;
      phase.raise_objection(this);

      orig = new("orig");
      orig.x = 10;
      orig.y = 20;

      copied = new("copied");
      copied.copy(orig);

      // Modify original, check copy
      orig.x = 999;
      if (copied.x == 10 && copied.y == 20)
        `uvm_info("TEST", "copy unchanged after orig modified: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("copy changed: x=%0d y=%0d: FAIL", copied.x, copied.y))

      // Modify copy, check original
      copied.y = 888;
      if (orig.y == 20)
        `uvm_info("TEST", "orig unchanged after copy modified: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("orig changed: y=%0d: FAIL", orig.y))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_copy_indep_test");
endmodule
