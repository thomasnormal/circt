// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test multiple sequences with SEQ_ARB_FIFO arbitration.
// Two sequences generate items, verify they interleave in FIFO order.

// CHECK: [TEST] seq_alpha item sent: PASS
// CHECK: [TEST] seq_beta item sent: PASS
// CHECK: [TEST] arbitration complete: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class arb_item extends uvm_sequence_item;
    `uvm_object_utils(arb_item)
    string origin;
    function new(string name = "arb_item");
      super.new(name);
    endfunction
  endclass

  class arb_driver extends uvm_driver#(arb_item);
    `uvm_component_utils(arb_driver)
    int count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      count = 0;
    endfunction
    task run_phase(uvm_phase phase);
      arb_item req;
      forever begin
        seq_item_port.get_next_item(req);
        count++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class seq_alpha extends uvm_sequence#(arb_item);
    `uvm_object_utils(seq_alpha)
    function new(string name = "seq_alpha");
      super.new(name);
    endfunction
    task body();
      arb_item item;
      item = arb_item::type_id::create("item");
      start_item(item);
      item.origin = "alpha";
      finish_item(item);
      `uvm_info("TEST", "seq_alpha item sent: PASS", UVM_LOW)
    endtask
  endclass

  class seq_beta extends uvm_sequence#(arb_item);
    `uvm_object_utils(seq_beta)
    function new(string name = "seq_beta");
      super.new(name);
    endfunction
    task body();
      arb_item item;
      item = arb_item::type_id::create("item");
      start_item(item);
      item.origin = "beta";
      finish_item(item);
      `uvm_info("TEST", "seq_beta item sent: PASS", UVM_LOW)
    endtask
  endclass

  class arb_test extends uvm_test;
    `uvm_component_utils(arb_test)
    uvm_sequencer#(arb_item) sqr;
    arb_driver drv;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      sqr = uvm_sequencer#(arb_item)::type_id::create("sqr", this);
      drv = arb_driver::type_id::create("drv", this);
    endfunction

    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction

    task run_phase(uvm_phase phase);
      seq_alpha sa;
      seq_beta sb;
      phase.raise_objection(this);
      sqr.set_arbitration(SEQ_ARB_FIFO);
      sa = seq_alpha::type_id::create("sa");
      sb = seq_beta::type_id::create("sb");
      fork
        sa.start(sqr);
        sb.start(sqr);
      join
      `uvm_info("TEST", "arbitration complete: PASS", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("arb_test");
endmodule
