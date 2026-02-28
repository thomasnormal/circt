// RUN: crun %s --top uvm_basic_tb -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test basic UVM test execution through crun.
// Verifies that UVM auto-include correctly detects UVM references in the source
// and injects the UVM package and include directories.

// CHECK: [TEST] UVM basic test running via crun
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps

module uvm_basic_tb;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  class crun_uvm_test extends uvm_test;
    `uvm_component_utils(crun_uvm_test)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("TEST", "UVM basic test running via crun", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("crun_uvm_test");
endmodule
