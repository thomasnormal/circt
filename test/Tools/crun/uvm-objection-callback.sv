// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test objection callbacks: raised, dropped, all_dropped.
// Verifies callbacks fire when objections are raised/dropped.

// CHECK: [TEST] raised callback fired: PASS
// CHECK: [TEST] dropped callback fired: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class obj_cb_test extends uvm_test;
    `uvm_component_utils(obj_cb_test)

    bit raised_seen;
    bit dropped_seen;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      raised_seen = 0;
      dropped_seen = 0;
    endfunction

    function void raised(uvm_objection objection, uvm_object source_obj,
                         string description, int count);
      raised_seen = 1;
    endfunction

    function void dropped(uvm_objection objection, uvm_object source_obj,
                          string description, int count);
      dropped_seen = 1;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      // The phase objection raise above should have triggered raised()
      if (raised_seen)
        `uvm_info("TEST", "raised callback fired: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "raised callback fired: FAIL")

      phase.drop_objection(this);

      // Give a delta for dropped callback
      #0;
      if (dropped_seen)
        `uvm_info("TEST", "dropped callback fired: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "dropped callback fired: FAIL")
    endtask
  endclass

  initial run_test("obj_cb_test");
endmodule
