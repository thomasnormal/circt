// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test bounded TLM FIFO capacity enforcement.
// Verifies try_put fails when FIFO is full and succeeds after get.
// KNOWN BROKEN: capacity enforcement may not work.

// CHECK: [TEST] bounded put 2: PASS
// CHECK: [TEST] full try_put rejected: PASS
// CHECK: [TEST] after get try_put: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class bnd_txn extends uvm_sequence_item;
    `uvm_object_utils(bnd_txn)
    int data;
    function new(string name = "bnd_txn");
      super.new(name);
    endfunction
  endclass

  class bounded_fifo_test extends uvm_test;
    `uvm_component_utils(bounded_fifo_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_tlm_fifo #(bnd_txn) fifo;
      bnd_txn txn, got;
      bit ok;

      phase.raise_objection(this);

      // Create bounded FIFO with capacity 2
      fifo = new("fifo", this, 2);

      // Put 2 items (should succeed)
      txn = bnd_txn::type_id::create("t1");
      txn.data = 10;
      ok = fifo.try_put(txn);
      txn = bnd_txn::type_id::create("t2");
      txn.data = 20;
      if (ok) ok = fifo.try_put(txn);
      if (ok)
        `uvm_info("TEST", "bounded put 2: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "bounded put 2: FAIL")

      // try_put a 3rd item — should fail (capacity full)
      txn = bnd_txn::type_id::create("t3");
      txn.data = 30;
      ok = fifo.try_put(txn);
      if (!ok)
        `uvm_info("TEST", "full try_put rejected: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "full try_put rejected: FAIL (was accepted)")

      // Get 1 item, then try_put should succeed
      fifo.get(got);
      txn = bnd_txn::type_id::create("t4");
      txn.data = 40;
      ok = fifo.try_put(txn);
      if (ok)
        `uvm_info("TEST", "after get try_put: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "after get try_put: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("bounded_fifo_test");
endmodule
