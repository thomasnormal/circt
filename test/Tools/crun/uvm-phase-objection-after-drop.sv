// RUN: crun %s --top tb_top -v 0 --max-time 100000 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: raise objection after already dropping. Tests re-raise behavior.

// CHECK: [TEST] re-raise after drop: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class neg_phase_reraise_test extends uvm_test;
    `uvm_component_utils(neg_phase_reraise_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_report_server srv;
      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // First raise and drop
      phase.raise_objection(this);
      @(posedge clk);
      phase.drop_objection(this);

      // Re-raise after drop — may extend the phase or be ignored
      phase.raise_objection(this);
      @(posedge clk);
      `uvm_info("TEST", "re-raise after drop: PASS", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_phase_reraise_test");
endmodule
