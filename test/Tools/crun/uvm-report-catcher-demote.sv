// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test report catcher that demotes UVM_ERROR to UVM_WARNING.
// Register catcher, issue UVM_ERROR, verify it becomes UVM_WARNING.

// CHECK: [TEST] catcher registered: PASS
// CHECK: [TEST] error demoted to warning: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class demote_catcher extends uvm_report_catcher;
    `uvm_object_utils(demote_catcher)
    int caught_count;
    function new(string name = "demote_catcher");
      super.new(name);
      caught_count = 0;
    endfunction
    function action_e catch_action();
      if (get_severity() == UVM_ERROR && get_id() == "DEMOTE_ME") begin
        set_severity(UVM_WARNING);
        caught_count++;
      end
      return THROW;
    endfunction
  endclass

  class catcher_demote_test extends uvm_test;
    `uvm_component_utils(catcher_demote_test)
    demote_catcher catcher;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      catcher = new("catcher");
      uvm_report_cb::add(null, catcher);
      `uvm_info("TEST", "catcher registered: PASS", UVM_LOW)
    endfunction

    task run_phase(uvm_phase phase);
      int err_count_before, err_count_after;
      phase.raise_objection(this);

      err_count_before = get_report_server().get_severity_count(UVM_ERROR);
      `uvm_error("DEMOTE_ME", "this should be demoted")
      err_count_after = get_report_server().get_severity_count(UVM_ERROR);

      if (err_count_after == err_count_before)
        `uvm_info("TEST", "error demoted to warning: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "error demoted to warning: FAIL (error count increased)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("catcher_demote_test");
endmodule
