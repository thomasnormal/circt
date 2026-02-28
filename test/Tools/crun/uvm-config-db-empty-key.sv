// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: set/get with empty string key. Should work or fail gracefully.

// CHECK: [TEST] empty key: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_config_empty_key_test extends uvm_test;
    `uvm_component_utils(neg_config_empty_key_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      phase.raise_objection(this);

      // Set with empty key
      uvm_config_db#(int)::set(this, "", "", 77);

      // Get with empty key
      ok = uvm_config_db#(int)::get(this, "", "", val);

      if (ok && val == 77)
        `uvm_info("TEST", "empty key: PASS", UVM_LOW)
      else if (!ok)
        // Also acceptable: empty key not supported, but no crash
        `uvm_info("TEST", "empty key: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("empty key: FAIL (ok=%0b val=%0d)", ok, val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_config_empty_key_test");
endmodule
