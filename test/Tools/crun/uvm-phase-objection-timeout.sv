// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test that simulation ends after objection drain time expires.
// Raises objection, sets drain time, drops objection, verifies extract_phase runs.

// CHECK: [TEST] run_phase started: PASS
// CHECK: [TEST] extract_phase reached: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class objection_timeout_test extends uvm_test;
    `uvm_component_utils(objection_timeout_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      phase.get_objection().set_drain_time(this, 100);
      `uvm_info("TEST", "run_phase started: PASS", UVM_LOW)
      #50;
      phase.drop_objection(this);
    endtask

    function void extract_phase(uvm_phase phase);
      `uvm_info("TEST", "extract_phase reached: PASS", UVM_LOW)
    endfunction
  endclass

  initial run_test("objection_timeout_test");
endmodule
