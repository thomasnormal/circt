// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test factory with parameterized class.

// CHECK: [TEST] parameterized create default: PASS
// CHECK: [TEST] parameterized create wide: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_param_item #(int W = 8) extends uvm_object;
    `uvm_object_param_utils(edge_param_item#(W))
    bit [W-1:0] data;

    function new(string name = "edge_param_item");
      super.new(name);
    endfunction

    function int get_width();
      return W;
    endfunction
  endclass

  class edge_factory_param_test extends uvm_test;
    `uvm_component_utils(edge_factory_param_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      edge_param_item #(8) item8;
      edge_param_item #(16) item16;
      phase.raise_objection(this);

      // Create default width
      item8 = edge_param_item#(8)::type_id::create("item8");
      if (item8 != null && item8.get_width() == 8)
        `uvm_info("TEST", "parameterized create default: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "parameterized create default: FAIL")

      // Create wide
      item16 = edge_param_item#(16)::type_id::create("item16");
      if (item16 != null && item16.get_width() == 16)
        `uvm_info("TEST", "parameterized create wide: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "parameterized create wide: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_factory_param_test");
endmodule
