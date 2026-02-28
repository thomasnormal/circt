// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_barrier and uvm_barrier_pool.
// Verifies wait_for with threshold, reset, and get_threshold.

// CHECK: [TEST] barrier releases all waiters: PASS
// CHECK: [TEST] get_threshold: PASS
// CHECK: [TEST] barrier reset: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class barrier_test extends uvm_test;
    `uvm_component_utils(barrier_test)

    uvm_barrier bar;
    int done_count;

    function new(string name, uvm_component parent);
      super.new(name, parent);
      done_count = 0;
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      bar = new("bar");
      bar.set_threshold(3);

      // Test 1: fork 3 processes, all wait_for the barrier
      fork
        begin bar.wait_for(); done_count++; end
        begin bar.wait_for(); done_count++; end
        begin bar.wait_for(); done_count++; end
      join

      if (done_count == 3)
        `uvm_info("TEST", "barrier releases all waiters: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "barrier releases all waiters: FAIL")

      // Test 2: get_threshold
      if (bar.get_threshold() == 3)
        `uvm_info("TEST", "get_threshold: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get_threshold: FAIL")

      // Test 3: reset and re-use
      bar.reset();
      bar.set_threshold(1);
      fork
        bar.wait_for();
      join
      `uvm_info("TEST", "barrier reset: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("barrier_test");
endmodule
