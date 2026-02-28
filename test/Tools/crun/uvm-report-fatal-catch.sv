// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: issue UVM_FATAL, catch with report_catcher to downgrade to UVM_ERROR.
// Simulation should continue past the fatal.

// CHECK: [TEST] fatal was caught and downgraded: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_fatal_catcher extends uvm_report_catcher;
    `uvm_object_utils(neg_fatal_catcher)
    int caught_count;

    function new(string name = "neg_fatal_catcher");
      super.new(name);
      caught_count = 0;
    endfunction

    function action_e catch_action();
      if (get_severity() == UVM_FATAL) begin
        set_severity(UVM_ERROR);
        caught_count++;
        return THROW;
      end
      return THROW;
    endfunction
  endclass

  class neg_report_fatal_catch_test extends uvm_test;
    `uvm_component_utils(neg_report_fatal_catch_test)
    neg_fatal_catcher catcher;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      uvm_report_server srv;
      super.build_phase(phase);
      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

      // Install catcher to intercept fatals
      catcher = neg_fatal_catcher::type_id::create("catcher");
      uvm_report_cb::add(null, catcher);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      // Issue a UVM_FATAL — should be caught and downgraded
      `uvm_fatal("TEST", "this fatal should be caught")

      // If we get here, the catcher worked
      if (catcher.caught_count > 0)
        `uvm_info("TEST", "fatal was caught and downgraded: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "fatal catch: FAIL (catcher count is 0)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_report_fatal_catch_test");
endmodule
