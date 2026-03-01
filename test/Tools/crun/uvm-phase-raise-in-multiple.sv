// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test dual objection: sim ends only after BOTH components drop.

// CHECK: sub dropped at time
// CHECK: parent dropped at time
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_sub_comp extends uvm_component;
    `uvm_component_utils(edge_sub_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      #20;
      `uvm_info("TEST", $sformatf("sub dropped at time %0t: PASS", $time), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  class edge_phase_multi_obj_test extends uvm_test;
    `uvm_component_utils(edge_phase_multi_obj_test)
    edge_sub_comp sub;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sub = edge_sub_comp::type_id::create("sub", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      #30;
      `uvm_info("TEST", $sformatf("parent dropped at time %0t: PASS", $time), UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_phase_multi_obj_test");
endmodule
