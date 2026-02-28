// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_barrier_pool: global pool, get by name, threshold.

// CHECK: [TEST] global pool exists: PASS
// CHECK: [TEST] get by name returns barrier: PASS
// CHECK: [TEST] same name same barrier: PASS
// CHECK: [TEST] threshold set/get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class barrier_pool_test extends uvm_test;
    `uvm_component_utils(barrier_pool_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_barrier_pool pool;
      uvm_barrier b1, b2;

      phase.raise_objection(this);

      // Test 1: global pool exists
      pool = uvm_barrier_pool::get_global_pool();
      if (pool != null)
        `uvm_info("TEST", "global pool exists: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "global pool exists: FAIL")

      // Test 2: get barrier by name (auto-creates)
      b1 = pool.get("sync_point_1");
      if (b1 != null)
        `uvm_info("TEST", "get by name returns barrier: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get by name returns barrier: FAIL")

      // Test 3: same name returns same barrier
      b2 = pool.get("sync_point_1");
      if (b1 == b2)
        `uvm_info("TEST", "same name same barrier: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "same name same barrier: FAIL")

      // Test 4: set/get threshold
      b1.set_threshold(5);
      if (b1.get_threshold() == 5)
        `uvm_info("TEST", "threshold set/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("threshold set/get: FAIL (got %0d)", b1.get_threshold()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("barrier_pool_test");
endmodule
