// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test TLM FIFO peek operations: peek() and try_peek() return item without removing.

// CHECK: [TEST] peek returns item: PASS
// CHECK: [TEST] try_peek returns item: PASS
// CHECK: [TEST] item still in FIFO after peeks: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class peek_item extends uvm_sequence_item;
    `uvm_object_utils(peek_item)
    int value;
    function new(string name = "peek_item");
      super.new(name);
    endfunction
  endclass

  class fifo_peek_test extends uvm_test;
    `uvm_component_utils(fifo_peek_test)
    uvm_tlm_fifo #(peek_item) fifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      fifo = new("fifo", this, 0);
    endfunction

    task run_phase(uvm_phase phase);
      peek_item txn, peeked, got;
      bit ok;

      phase.raise_objection(this);

      txn = peek_item::type_id::create("t1");
      txn.value = 123;
      fifo.put(txn);

      // Test 1: blocking peek — returns item without removing
      fork
        begin fifo.peek(peeked); end
        begin #100ns; end
      join_any
      if (peeked != null && peeked.value == 123)
        `uvm_info("TEST", "peek returns item: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "peek returns item: FAIL")

      // Test 2: try_peek — nonblocking peek
      ok = fifo.try_peek(peeked);
      if (ok && peeked.value == 123)
        `uvm_info("TEST", "try_peek returns item: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "try_peek returns item: FAIL")

      // Test 3: item still in FIFO — get should return it
      fifo.get(got);
      if (got.value == 123)
        `uvm_info("TEST", "item still in FIFO after peeks: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "item still in FIFO after peeks: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("fifo_peek_test");
endmodule
