// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_pool deeper operations: add, get, exists, delete, num, global pool.

// CHECK: [TEST] add/get/exists: PASS
// CHECK: [TEST] delete removes entry: PASS
// CHECK: [TEST] num after operations: PASS
// CHECK: [TEST] global pool shared: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class pool_ops_test extends uvm_test;
    `uvm_component_utils(pool_ops_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_pool #(string, int) p;
      uvm_pool #(string, int) gp1, gp2;

      phase.raise_objection(this);

      p = new("p");

      // Test 1: add, get, exists
      p.add("x", 10);
      p.add("y", 20);
      p.add("z", 30);
      if (p.exists("x") && p.get("x") == 10 &&
          p.exists("y") && p.get("y") == 20 &&
          p.exists("z") && p.get("z") == 30)
        `uvm_info("TEST", "add/get/exists: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "add/get/exists: FAIL")

      // Test 2: delete
      p.delete("y");
      if (!p.exists("y") && p.exists("x") && p.exists("z"))
        `uvm_info("TEST", "delete removes entry: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "delete removes entry: FAIL")

      // Test 3: num
      if (p.num() == 2)
        `uvm_info("TEST", "num after operations: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("num after operations: FAIL (got %0d)", p.num()))

      // Test 4: global pool returns same instance
      gp1 = uvm_pool#(string, int)::get_global_pool();
      gp2 = uvm_pool#(string, int)::get_global_pool();
      if (gp1 != null && gp1 == gp2)
        `uvm_info("TEST", "global pool shared: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "global pool shared: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("pool_ops_test");
endmodule
