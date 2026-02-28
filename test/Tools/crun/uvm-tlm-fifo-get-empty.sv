// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: try_get() and try_peek() on empty TLM FIFO. Should return 0.

// CHECK: [TEST] try_get empty returns 0: PASS
// CHECK: [TEST] try_peek empty returns 0: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_tlm_item extends uvm_object;
    `uvm_object_utils(neg_tlm_item)
    int data;
    function new(string name = "neg_tlm_item");
      super.new(name);
    endfunction
  endclass

  class neg_tlm_fifo_empty_test extends uvm_test;
    `uvm_component_utils(neg_tlm_fifo_empty_test)
    uvm_tlm_fifo #(neg_tlm_item) fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("fifo", this);
    endfunction

    task run_phase(uvm_phase phase);
      neg_tlm_item item;
      bit ok;
      phase.raise_objection(this);

      // try_get on empty FIFO — should return 0
      ok = fifo.try_get(item);
      if (!ok)
        `uvm_info("TEST", "try_get empty returns 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "try_get empty: FAIL (returned 1)")

      // try_peek on empty FIFO — should return 0
      ok = fifo.try_peek(item);
      if (!ok)
        `uvm_info("TEST", "try_peek empty returns 0: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "try_peek empty: FAIL (returned 1)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_tlm_fifo_empty_test");
endmodule
