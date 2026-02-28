// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_queue push/pop/get/size operations.
// Verifies FIFO and LIFO ordering.

// CHECK: [TEST] push_back and size: PASS
// CHECK: [TEST] get by index: PASS
// CHECK: [TEST] pop_front FIFO: PASS
// CHECK: [TEST] pop_back LIFO: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class queue_test extends uvm_test;
    `uvm_component_utils(queue_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_queue#(int) q;
      int val;
      phase.raise_objection(this);

      q = new("q");
      q.push_back(10);
      q.push_back(20);
      q.push_back(30);

      // Test 1: size
      if (q.size() == 3)
        `uvm_info("TEST", "push_back and size: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "push_back and size: FAIL")

      // Test 2: get by index
      val = q.get(1);
      if (val == 20)
        `uvm_info("TEST", "get by index: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get by index: FAIL")

      // Test 3: pop_front (FIFO)
      val = q.pop_front();
      if (val == 10)
        `uvm_info("TEST", "pop_front FIFO: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("pop_front FIFO: FAIL got %0d", val))

      // Test 4: pop_back (LIFO)
      val = q.pop_back();
      if (val == 30)
        `uvm_info("TEST", "pop_back LIFO: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("pop_back LIFO: FAIL got %0d", val))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("queue_test");
endmodule
