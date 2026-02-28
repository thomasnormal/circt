// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test TLM FIFO used() tracking and size() capacity.

// CHECK: [TEST] used increments on put: PASS
// CHECK: [TEST] used decrements on get: PASS
// CHECK: [TEST] size returns capacity: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_fifo_sz_test extends uvm_test;
    `uvm_component_utils(edge_fifo_sz_test)
    uvm_tlm_fifo #(int) fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("fifo", this, 8);
    endfunction

    task run_phase(uvm_phase phase);
      int val;
      bit put_ok, get_ok;
      phase.raise_objection(this);

      // Put 3 items, check used after each
      put_ok = 1;
      for (int i = 0; i < 3; i++) begin
        fifo.put(i * 10);
        if (fifo.used() != i + 1) put_ok = 0;
      end
      if (put_ok)
        `uvm_info("TEST", "used increments on put: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "used increments on put: FAIL")

      // Get 2 items, check used decrements
      get_ok = 1;
      for (int i = 0; i < 2; i++) begin
        fifo.get(val);
        if (fifo.used() != 2 - i) get_ok = 0;
      end
      if (get_ok)
        `uvm_info("TEST", "used decrements on get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "used decrements on get: FAIL")

      // Check size returns capacity
      if (fifo.size() == 8)
        `uvm_info("TEST", "size returns capacity: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("size returns capacity: FAIL (got %0d)", fifo.size()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_fifo_sz_test");
endmodule
