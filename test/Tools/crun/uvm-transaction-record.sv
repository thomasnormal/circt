// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test UVM transaction recording API.
// Verifies begin_tr/end_tr calls don't crash (recording may be no-op).

// CHECK: [TEST] transaction record: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_txn extends uvm_sequence_item;
    `uvm_object_utils(my_txn)
    int addr;
    int data;

    function new(string name = "my_txn");
      super.new(name);
    endfunction
  endclass

  class record_test extends uvm_test;
    `uvm_component_utils(record_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_txn txn;
      int tr_handle;

      phase.raise_objection(this);

      txn = my_txn::type_id::create("txn");
      txn.addr = 32'hA000;
      txn.data = 32'hDEAD;

      // begin_tr returns a transaction handle (or 0 if recording disabled)
      tr_handle = txn.begin_tr();
      #10;
      txn.end_tr();

      // If we got here without crashing, the API works
      `uvm_info("TEST", "transaction record: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("record_test");
endmodule
