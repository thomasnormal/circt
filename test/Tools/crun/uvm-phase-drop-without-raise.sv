// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: drop objection without raising. Should produce error, not crash.
// Use a report catcher to demote OBJTN_ZERO from UVM_FATAL to UVM_WARNING
// (UVM 1.1d terminates on UVM_FATAL regardless of set_max_quit_count).

// CHECK: [TEST] survived drop without raise: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class objtn_catcher extends uvm_report_catcher;
    function new(string name = "objtn_catcher");
      super.new(name);
    endfunction
    function action_e catch();
      if (get_id() == "OBJTN_ZERO")
        set_severity(UVM_WARNING);
      return THROW;
    endfunction
  endclass

  class neg_phase_drop_no_raise_test extends uvm_test;
    `uvm_component_utils(neg_phase_drop_no_raise_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      objtn_catcher catcher;

      // Raise once so the phase doesn't end immediately
      phase.raise_objection(this);

      // Install catcher to demote OBJTN_ZERO fatal
      catcher = new();
      uvm_report_cb::add(null, catcher);

      // Drop without raising (extra drop) — should produce error but not crash
      phase.drop_objection(this);
      phase.drop_objection(this);

      // If we get here, the simulator survived
      `uvm_info("TEST", "survived drop without raise: PASS", UVM_LOW)
    endtask
  endclass

  initial run_test("neg_phase_drop_no_raise_test");
endmodule
