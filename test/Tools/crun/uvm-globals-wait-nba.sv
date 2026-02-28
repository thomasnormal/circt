// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_wait_for_nba_region utility.
// Verifies it completes without hanging.

// CHECK: [TEST] wait_for_nba completes: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class nba_test extends uvm_test;
    `uvm_component_utils(nba_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      uvm_wait_for_nba_region();
      `uvm_info("TEST", "wait_for_nba completes: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("nba_test");
endmodule
