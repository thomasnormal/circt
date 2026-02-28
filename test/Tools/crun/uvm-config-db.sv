// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test basic UVM config_db set/get with int type.
// Verifies that config_db can store and retrieve integer values.
// NOTE: Wildcard matching (uvm_is_match) is known broken — only exact paths tested.

// CHECK: [TEST] config_db int set/get: PASS
// CHECK: [TEST] config_db string set/get: PASS
// CHECK: [TEST] config_db overwrite: PASS
// CHECK: [TEST] config_db miss: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class config_db_test extends uvm_test;
    `uvm_component_utils(config_db_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      string sval;
      bit ok;

      phase.raise_objection(this);

      // Test 1: set/get int
      uvm_config_db#(int)::set(this, "", "my_int", 42);
      ok = uvm_config_db#(int)::get(this, "", "my_int", val);
      if (ok && val == 42)
        `uvm_info("TEST", "config_db int set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("config_db int set/get: FAIL (ok=%0b val=%0d)", ok, val))

      // Test 2: set/get string
      uvm_config_db#(string)::set(this, "", "my_str", "hello");
      ok = uvm_config_db#(string)::get(this, "", "my_str", sval);
      if (ok && sval == "hello")
        `uvm_info("TEST", "config_db string set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config_db string set/get: FAIL")

      // Test 3: overwrite value
      uvm_config_db#(int)::set(this, "", "my_int", 99);
      ok = uvm_config_db#(int)::get(this, "", "my_int", val);
      if (ok && val == 99)
        `uvm_info("TEST", "config_db overwrite: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config_db overwrite: FAIL")

      // Test 4: get nonexistent key returns 0
      ok = uvm_config_db#(int)::get(this, "", "nonexistent_key", val);
      if (!ok)
        `uvm_info("TEST", "config_db miss: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "config_db miss: FAIL (should return 0)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("config_db_test");
endmodule
