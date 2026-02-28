// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test that run_phase starts at simulation time 0.

// CHECK: [TEST] run_phase starts at time 0: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_phase_zero_test extends uvm_test;
    `uvm_component_utils(edge_phase_zero_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      if ($time == 0)
        `uvm_info("TEST", "run_phase starts at time 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("run_phase starts at time 0: FAIL ($time=%0t)", $time))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_phase_zero_test");
endmodule
