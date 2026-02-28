// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: get_child("nonexistent"). Should return null, not crash.

// CHECK: [TEST] get nonexistent child returns null: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_comp_get_child_test extends uvm_test;
    `uvm_component_utils(neg_comp_get_child_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_component child;
      uvm_report_server srv;
      phase.raise_objection(this);

      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Try to get a child that doesn't exist
      child = get_child("nonexistent_child_xyz");

      if (child == null)
        `uvm_info("TEST", "get nonexistent child returns null: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get nonexistent child: FAIL (got non-null)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_comp_get_child_test");
endmodule
