// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test analysis_port → uvm_tlm_analysis_fifo connection.
// Verifies items written to analysis_port appear in analysis_fifo.
// KNOWN BROKEN: analysis_fifo via analysis_port may not work.

// CHECK: [TEST] analysis fifo: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class afifo_txn extends uvm_sequence_item;
    `uvm_object_utils(afifo_txn)
    int data;
    function new(string name = "afifo_txn");
      super.new(name);
    endfunction
  endclass

  class afifo_test extends uvm_test;
    `uvm_component_utils(afifo_test)

    uvm_analysis_port #(afifo_txn) ap;
    uvm_tlm_analysis_fifo #(afifo_txn) afifo;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      ap = new("ap", this);
      afifo = new("afifo", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      ap.connect(afifo.analysis_export);
    endfunction

    task run_phase(uvm_phase phase);
      afifo_txn txn, got;
      bit ok;

      phase.raise_objection(this);

      txn = afifo_txn::type_id::create("txn");
      txn.data = 55;
      ap.write(txn);

      ok = afifo.try_get(got);
      if (ok && got.data == 55)
        `uvm_info("TEST", "analysis fifo: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("analysis fifo: FAIL (ok=%0b)", ok))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("afifo_test");
endmodule
