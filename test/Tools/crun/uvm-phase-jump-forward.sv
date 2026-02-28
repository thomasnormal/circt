// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test phase.jump() to skip forward to extract_phase.
// Verifies that extract_phase runs after the jump.

// CHECK: [TEST] run_phase jumping: PASS
// CHECK: [TEST] extract_phase reached: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class phase_jump_test extends uvm_test;
    `uvm_component_utils(phase_jump_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("TEST", "run_phase jumping: PASS", UVM_LOW)
      phase.jump(uvm_extract_phase::get());
      phase.drop_objection(this);
    endtask

    function void extract_phase(uvm_phase phase);
      `uvm_info("TEST", "extract_phase reached: PASS", UVM_LOW)
    endfunction
  endclass

  initial run_test("phase_jump_test");
endmodule
