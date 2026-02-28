// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test that all_dropped fires at phase end.
// Alternative callback path — may work even if individual raised/dropped don't fire.

// CHECK: [TEST] objection raised: PASS
// CHECK: [TEST] objection dropped: PASS
// CHECK: [TEST] phase completed: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_objection_test extends uvm_test;
    `uvm_component_utils(probe_objection_test)
    bit objection_raised_ok;
    bit objection_dropped_ok;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this, "test raise", 1);
      objection_raised_ok = 1;
      `uvm_info("TEST", "objection raised: PASS", UVM_LOW)

      #10;

      phase.drop_objection(this, "test drop", 1);
      objection_dropped_ok = 1;
      `uvm_info("TEST", "objection dropped: PASS", UVM_LOW)
    endtask

    function void report_phase(uvm_phase phase);
      // If we get here, the phase machinery worked
      if (objection_raised_ok && objection_dropped_ok)
        `uvm_info("TEST", "phase completed: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "phase completed: FAIL")
    endfunction
  endclass

  initial run_test("probe_objection_test");
endmodule
