// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_queue iteration with get(i) and insert().

// CHECK: [TEST] iterate all items: PASS
// CHECK: [TEST] insert at position: PASS
// CHECK: [TEST] order after insert: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class queue_iter_test extends uvm_test;
    `uvm_component_utils(queue_iter_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_queue #(string) q;
      string s;
      bit ok;

      phase.raise_objection(this);

      q = new("q");
      q.push_back("alpha");
      q.push_back("beta");
      q.push_back("gamma");
      q.push_back("delta");

      // Test 1: iterate with get(i) for i in [0:size()-1]
      ok = 1;
      if (q.get(0) != "alpha") ok = 0;
      if (q.get(1) != "beta")  ok = 0;
      if (q.get(2) != "gamma") ok = 0;
      if (q.get(3) != "delta") ok = 0;
      if (q.size() != 4)       ok = 0;
      if (ok)
        `uvm_info("TEST", "iterate all items: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "iterate all items: FAIL")

      // Test 2: insert at specific position (index 2)
      q.insert(2, "inserted");
      if (q.size() == 5)
        `uvm_info("TEST", "insert at position: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("insert at position: FAIL (size=%0d)", q.size()))

      // Test 3: verify order after insert
      ok = 1;
      if (q.get(0) != "alpha")    ok = 0;
      if (q.get(1) != "beta")     ok = 0;
      if (q.get(2) != "inserted") ok = 0;
      if (q.get(3) != "gamma")    ok = 0;
      if (q.get(4) != "delta")    ok = 0;
      if (ok)
        `uvm_info("TEST", "order after insert: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "order after insert: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("queue_iter_test");
endmodule
