// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sequence_item clone: clone has same values, is independent.

// CHECK: [TEST] clone has same values: PASS
// CHECK: [TEST] clone is independent: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_clone_item extends uvm_sequence_item;
    `uvm_object_utils(edge_clone_item)
    int addr;
    int data;

    function new(string name = "edge_clone_item");
      super.new(name);
    endfunction
  endclass

  class edge_clone_test extends uvm_test;
    `uvm_component_utils(edge_clone_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_clone_item orig, cloned;
      uvm_object tmp;
      phase.raise_objection(this);

      orig = edge_clone_item::type_id::create("orig");
      orig.addr = 100;
      orig.data = 200;

      tmp = orig.clone();
      $cast(cloned, tmp);

      // Verify clone has same values
      if (cloned.addr == 100 && cloned.data == 200)
        `uvm_info("TEST", "clone has same values: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "clone has same values: FAIL")

      // Modify original, verify clone unchanged
      orig.addr = 999;
      if (cloned.addr == 100)
        `uvm_info("TEST", "clone is independent: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("clone is independent: FAIL (addr=%0d)", cloned.addr))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_clone_test");
endmodule
