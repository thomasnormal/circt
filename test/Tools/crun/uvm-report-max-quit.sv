// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test set_max_quit_count() terminates simulation after N errors.
// Set max quit to 3, issue 3 UVM_ERRORs, verify simulation ends.
// We check that report_phase still runs (cleanup phase).

// CHECK: [TEST] issuing errors
// CHECK: [TEST] report_phase reached: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class max_quit_test extends uvm_test;
    `uvm_component_utils(max_quit_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_report_server srv;
      srv = get_report_server();
      srv.set_max_quit_count(3);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      `uvm_info("TEST", "issuing errors", UVM_LOW)
      `uvm_error("QUIT_TEST", "error 1 of 3")
      `uvm_error("QUIT_TEST", "error 2 of 3")
      `uvm_error("QUIT_TEST", "error 3 of 3")
      // Simulation should end here due to max_quit

      phase.drop_objection(this);
    endtask

    function void report_phase(uvm_phase phase);
      `uvm_info("TEST", "report_phase reached: PASS", UVM_LOW)
    endfunction
  endclass

  initial run_test("max_quit_test");
endmodule
