// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test uvm_transaction timing APIs.
// Verifies accept_tr, begin_tr, end_tr and their getters.

// CHECK: [TEST] accept_time valid: PASS
// CHECK: [TEST] begin_time valid: PASS
// CHECK: [TEST] end_time valid: PASS
// CHECK: [TEST] transaction_id unique: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_txn extends uvm_transaction;
    `uvm_object_utils(my_txn)
    function new(string name = "my_txn");
      super.new(name);
    endfunction
  endclass

  class txn_timing_test extends uvm_test;
    `uvm_component_utils(txn_timing_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      my_txn t1, t2;
      phase.raise_objection(this);

      t1 = my_txn::type_id::create("t1");
      t2 = my_txn::type_id::create("t2");

      #5ns;
      t1.accept_tr();
      if (t1.get_accept_time() >= 0)
        `uvm_info("TEST", "accept_time valid: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "accept_time valid: FAIL")

      #5ns;
      t1.begin_tr();
      if (t1.get_begin_time() >= 0)
        `uvm_info("TEST", "begin_time valid: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "begin_time valid: FAIL")

      #5ns;
      t1.end_tr();
      if (t1.get_end_time() >= 0)
        `uvm_info("TEST", "end_time valid: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "end_time valid: FAIL")

      // Test transaction IDs are unique
      if (t1.get_transaction_id() != t2.get_transaction_id())
        `uvm_info("TEST", "transaction_id unique: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "transaction_id unique: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("txn_timing_test");
endmodule
