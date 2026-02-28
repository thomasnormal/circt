// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test TLM FIFO flush, is_empty, and used() operations.

// CHECK: [TEST] FIFO used after 5 puts: PASS
// CHECK: [TEST] FIFO is_empty false: PASS
// CHECK: [TEST] flush empties FIFO: PASS
// CHECK: [TEST] FIFO is_empty after flush: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class flush_item extends uvm_sequence_item;
    `uvm_object_utils(flush_item)
    int value;
    function new(string name = "flush_item");
      super.new(name);
    endfunction
  endclass

  class fifo_flush_test extends uvm_test;
    `uvm_component_utils(fifo_flush_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_fifo #(flush_item) fifo;
      flush_item txn, got;
      bit ok;

      phase.raise_objection(this);

      fifo = new("fifo", this, 0);

      // Put 5 items
      for (int i = 0; i < 5; i++) begin
        txn = flush_item::type_id::create($sformatf("f_%0d", i));
        txn.value = i;
        fifo.put(txn);
      end

      // Test 1: used() returns 5
      if (fifo.used() == 5)
        `uvm_info("TEST", "FIFO used after 5 puts: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("FIFO used: FAIL (got %0d)", fifo.used()))

      // Test 2: is_empty returns 0
      if (!fifo.is_empty())
        `uvm_info("TEST", "FIFO is_empty false: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "FIFO is_empty false: FAIL")

      // Test 3: flush empties FIFO
      fifo.flush();
      ok = fifo.try_get(got);
      if (!ok && fifo.used() == 0)
        `uvm_info("TEST", "flush empties FIFO: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "flush empties FIFO: FAIL")

      // Test 4: is_empty after flush
      if (fifo.is_empty())
        `uvm_info("TEST", "FIFO is_empty after flush: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "FIFO is_empty after flush: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("fifo_flush_test");
endmodule
