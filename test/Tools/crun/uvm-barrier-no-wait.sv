// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test barrier creation and query API without calling wait_for.
// Just exercise set_threshold, get_threshold, get_num_waiters.

// CHECK: [TEST] barrier create: PASS
// CHECK: [TEST] set/get threshold: PASS
// CHECK: [TEST] num waiters zero: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_barrier_api_test extends uvm_test;
    `uvm_component_utils(probe_barrier_api_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_barrier b;
      phase.raise_objection(this);

      b = new("my_barrier");
      if (b != null)
        `uvm_info("TEST", "barrier create: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "barrier create: FAIL")

      b.set_threshold(5);
      if (b.get_threshold() == 5)
        `uvm_info("TEST", "set/get threshold: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("set/get threshold: FAIL (got %0d)", b.get_threshold()))

      if (b.get_num_waiters() == 0)
        `uvm_info("TEST", "num waiters zero: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("num waiters zero: FAIL (got %0d)", b.get_num_waiters()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_barrier_api_test");
endmodule
