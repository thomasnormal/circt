// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: set/get with null context. Should work without crashing.

// CHECK: [TEST] null context set/get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_config_null_ctx_test extends uvm_test;
    `uvm_component_utils(neg_config_null_ctx_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit ok;
      phase.raise_objection(this);

      // Set with null context
      uvm_config_db#(int)::set(null, "", "null_ctx_key", 42);

      // Get with null context
      ok = uvm_config_db#(int)::get(null, "", "null_ctx_key", val);

      if (ok && val == 42)
        `uvm_info("TEST", "null context set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("null context: FAIL (ok=%0b val=%0d)", ok, val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_config_null_ctx_test");
endmodule
