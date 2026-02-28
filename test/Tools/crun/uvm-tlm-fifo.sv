// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test basic UVM TLM FIFO put/get operations.
// Verifies unbounded FIFO and try_put/try_get nonblocking operations.
// NOTE: Bounded FIFO capacity enforcement is known broken — not tested here.

// CHECK: [TEST] put/get basic: PASS
// CHECK: [TEST] try_put/try_get: PASS
// CHECK: [TEST] FIFO ordering: PASS
// CHECK: [TEST] empty try_get: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class simple_txn extends uvm_sequence_item;
    `uvm_object_utils(simple_txn)
    int data;
    function new(string name = "simple_txn");
      super.new(name);
    endfunction
  endclass

  class fifo_test extends uvm_test;
    `uvm_component_utils(fifo_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_fifo #(simple_txn) fifo;
      simple_txn txn, got;
      bit ok;

      phase.raise_objection(this);

      // Use unbounded FIFO (size 0) to avoid capacity enforcement bugs
      fifo = new("fifo", this, 0);

      // Test 1: basic put/get
      txn = simple_txn::type_id::create("t1");
      txn.data = 42;
      fifo.put(txn);
      fifo.get(got);
      if (got.data == 42)
        `uvm_info("TEST", "put/get basic: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("put/get basic: FAIL (got %0d)", got.data))

      // Test 2: try_put/try_get
      txn = simple_txn::type_id::create("t2");
      txn.data = 77;
      ok = fifo.try_put(txn);
      if (!ok) begin
        `uvm_error("TEST", "try_put failed on unbounded FIFO")
      end else begin
        ok = fifo.try_get(got);
        if (ok && got.data == 77)
          `uvm_info("TEST", "try_put/try_get: PASS", UVM_LOW)
        else
          `uvm_error("TEST", "try_put/try_get: FAIL")
      end

      // Test 3: ordering (FIFO semantics)
      for (int i = 0; i < 5; i++) begin
        txn = simple_txn::type_id::create($sformatf("ord_%0d", i));
        txn.data = i * 100;
        fifo.put(txn);
      end
      ok = 1;
      for (int i = 0; i < 5; i++) begin
        fifo.get(got);
        if (got.data != i * 100) ok = 0;
      end
      if (ok)
        `uvm_info("TEST", "FIFO ordering: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "FIFO ordering: FAIL")

      // Test 4: try_get on empty FIFO returns 0
      ok = fifo.try_get(got);
      if (!ok)
        `uvm_info("TEST", "empty try_get: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "empty try_get: FAIL (should return 0)")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("fifo_test");
endmodule
