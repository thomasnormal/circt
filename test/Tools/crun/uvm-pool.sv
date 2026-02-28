// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_pool (associative array wrapper) and uvm_queue.

// CHECK: [TEST] pool add/get: PASS
// CHECK: [TEST] pool exists/delete: PASS
// CHECK: [TEST] pool num: PASS
// CHECK: [TEST] queue push/get: PASS
// CHECK: [TEST] queue size: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class pool_test extends uvm_test;
    `uvm_component_utils(pool_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_pool #(string, int) pool;
      uvm_queue #(int) queue;
      int val;

      phase.raise_objection(this);

      // === Pool tests ===
      pool = new("pool");

      // Test 1: add and get
      pool.add("alpha", 10);
      pool.add("beta", 20);
      pool.add("gamma", 30);
      if (pool.get("alpha") == 10 && pool.get("beta") == 20 && pool.get("gamma") == 30)
        `uvm_info("TEST", "pool add/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "pool add/get: FAIL")

      // Test 2: exists and delete
      if (pool.exists("alpha")) begin
        pool.delete("alpha");
        if (!pool.exists("alpha"))
          `uvm_info("TEST", "pool exists/delete: PASS", UVM_LOW)
        else
          `uvm_error("TEST", "pool exists/delete: FAIL (still exists after delete)")
      end else begin
        `uvm_error("TEST", "pool exists/delete: FAIL (not found)")
      end

      // Test 3: num
      if (pool.num() == 2)
        `uvm_info("TEST", "pool num: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("pool num: FAIL (got %0d)", pool.num()))

      // === Queue tests ===
      queue = new("queue");

      // Test 4: push_back and get
      queue.push_back(100);
      queue.push_back(200);
      queue.push_back(300);
      if (queue.get(0) == 100 && queue.get(1) == 200 && queue.get(2) == 300)
        `uvm_info("TEST", "queue push/get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "queue push/get: FAIL")

      // Test 5: size
      if (queue.size() == 3)
        `uvm_info("TEST", "queue size: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("queue size: FAIL (got %0d)", queue.size()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("pool_test");
endmodule
