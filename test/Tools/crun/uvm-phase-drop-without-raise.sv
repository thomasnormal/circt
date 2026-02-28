// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: drop objection without raising. Should produce error, not crash.

// CHECK: [TEST] survived drop without raise: PASS
// CHECK: [circt-sim] Simulation completed

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
      uvm_report_server srv;

      // Raise once so the phase doesn't end immediately
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Drop without raising (extra drop) — should produce error but not crash
      phase.drop_objection(this);
      phase.drop_objection(this);

      // If we get here, the simulator survived
      `uvm_info("TEST", "survived drop without raise: PASS", UVM_LOW)
    endtask
  endclass

  initial run_test("neg_phase_drop_no_raise_test");
endmodule
