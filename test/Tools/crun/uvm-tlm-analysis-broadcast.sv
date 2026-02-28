// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test analysis_port broadcasting to multiple subscribers.
// Sends 5 items through analysis port, verifies all 3 subscribers receive all 5.

// CHECK: [TEST] subscriber counts correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_txn extends uvm_object;
    `uvm_object_utils(my_txn)
    int value;
    function new(string name = "my_txn");
      super.new(name);
    endfunction
  endclass

  class my_subscriber extends uvm_subscriber#(my_txn);
    `uvm_component_utils(my_subscriber)
    int rx_count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      rx_count = 0;
    endfunction
    function void write(my_txn t);
      rx_count++;
    endfunction
  endclass

  class analysis_bcast_test extends uvm_test;
    `uvm_component_utils(analysis_bcast_test)
    uvm_analysis_port#(my_txn) ap;
    my_subscriber sub0, sub1, sub2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      ap = new("ap", this);
      sub0 = my_subscriber::type_id::create("sub0", this);
      sub1 = my_subscriber::type_id::create("sub1", this);
      sub2 = my_subscriber::type_id::create("sub2", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      ap.connect(sub0.analysis_export);
      ap.connect(sub1.analysis_export);
      ap.connect(sub2.analysis_export);
    endfunction

    task run_phase(uvm_phase phase);
      my_txn t;
      phase.raise_objection(this);
      for (int i = 0; i < 5; i++) begin
        t = my_txn::type_id::create($sformatf("t%0d", i));
        t.value = i;
        ap.write(t);
      end
      if (sub0.rx_count == 5 && sub1.rx_count == 5 && sub2.rx_count == 5)
        `uvm_info("TEST", "subscriber counts correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("subscriber counts: FAIL (%0d,%0d,%0d)",
                   sub0.rx_count, sub1.rx_count, sub2.rx_count))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("analysis_bcast_test");
endmodule
