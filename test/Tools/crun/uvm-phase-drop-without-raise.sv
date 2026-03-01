// RUN: not crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: drop objection without raising should issue UVM_FATAL
// with OBJTN_ZERO and terminate the simulation (matching UVM reference
// behavior in uvm_objection.svh lines 625-634).
//
// The "not" prefix expects a non-zero exit code from crun.

// CHECK: UVM_FATAL
// CHECK-SAME: OBJTN_ZERO
// CHECK-NOT: [AFTER] THIS SHOULD NEVER PRINT

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_phase_drop_no_raise_test extends uvm_test;
    `uvm_component_utils(neg_phase_drop_no_raise_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      // Raise once so the phase doesn't end immediately
      phase.raise_objection(this);

      // Drop (balanced)
      phase.drop_objection(this);

      // Extra drop — should trigger OBJTN_ZERO fatal
      phase.drop_objection(this);

      // This line should never execute
      $display("[AFTER] THIS SHOULD NEVER PRINT");
    endtask
  endclass

  initial run_test("neg_phase_drop_no_raise_test");
endmodule
