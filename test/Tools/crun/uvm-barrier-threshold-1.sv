// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: test barrier with threshold=1 (trivial case — single waiter).
// May work even if multi-process barrier sync is broken.

// CHECK: [TEST] barrier wait threshold 1: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class probe_barrier_test extends uvm_test;
    `uvm_component_utils(probe_barrier_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_barrier b;
      phase.raise_objection(this);

      b = new("b");
      b.set_threshold(1);

      // With threshold=1, a single wait_for should unblock immediately
      fork
        begin
          b.wait_for();
          `uvm_info("TEST", "barrier wait threshold 1: PASS", UVM_LOW)
        end
      join

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_barrier_test");
endmodule
