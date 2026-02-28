// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: config_db interceptor does not enforce type checking across parameterized specializations

// Test config_db type mismatch: set as int, get as string returns not-found.

// CHECK: [TEST] type mismatch get returns 0: PASS
// CHECK: [TEST] correct type still works: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_config_mismatch_test extends uvm_test;
    `uvm_component_utils(edge_config_mismatch_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int ival;
      string sval;
      bit ok;

      phase.raise_objection(this);

      // Set as int
      uvm_config_db#(int)::set(this, "", "typed_key", 42);

      // Try to get as string — should return 0 (not found)
      ok = uvm_config_db#(string)::get(this, "", "typed_key", sval);
      if (!ok)
        `uvm_info("TEST", "type mismatch get returns 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "type mismatch get returns 0: FAIL (should not find)")

      // Verify correct type still works
      ok = uvm_config_db#(int)::get(this, "", "typed_key", ival);
      if (ok && ival == 42)
        `uvm_info("TEST", "correct type still works: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "correct type still works: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_config_mismatch_test");
endmodule
