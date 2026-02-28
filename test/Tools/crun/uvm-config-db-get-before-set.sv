// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: get() before any set(). Should return 0 (not found), not crash.

// CHECK: [TEST] get before set returns 0: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_config_get_before_set_test extends uvm_test;
    `uvm_component_utils(neg_config_get_before_set_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      phase.raise_objection(this);

      // Try to get a key that was never set
      val = 999;
      ok = uvm_config_db#(int)::get(this, "", "never_set_key", val);

      if (!ok)
        `uvm_info("TEST", "get before set returns 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("get before set: FAIL (ok=%0b val=%0d)", ok, val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_config_get_before_set_test");
endmodule
