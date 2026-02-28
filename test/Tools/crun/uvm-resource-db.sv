// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_resource_db set/read_by_name operations.
// Verifies resource database basic store and retrieve.

// CHECK: [TEST] resource_db int: PASS
// CHECK: [TEST] resource_db string: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class rdb_test extends uvm_test;
    `uvm_component_utils(rdb_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int ival;
      string sval;
      bit ok;

      phase.raise_objection(this);

      // Test 1: int resource
      uvm_resource_db #(int)::set("my_scope", "my_key", 42, null);
      ok = uvm_resource_db #(int)::read_by_name("my_scope", "my_key", ival, null);
      if (ok && ival == 42)
        `uvm_info("TEST", "resource_db int: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("resource_db int: FAIL (ok=%0b, val=%0d)", ok, ival))

      // Test 2: string resource
      uvm_resource_db #(string)::set("scope2", "skey", "hello", null);
      ok = uvm_resource_db #(string)::read_by_name("scope2", "skey", sval, null);
      if (ok && sval == "hello")
        `uvm_info("TEST", "resource_db string: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("resource_db string: FAIL (ok=%0b, val=%0s)", ok, sval))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("rdb_test");
endmodule
