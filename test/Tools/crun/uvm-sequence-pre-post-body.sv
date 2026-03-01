// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sequence pre_body/post_body callbacks.

// CHECK: [TEST] sequence order: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class ppb_item extends uvm_sequence_item;
    `uvm_object_utils(ppb_item)
    int data;
    function new(string name = "ppb_item");
      super.new(name);
    endfunction
  endclass

  class ppb_sequence extends uvm_sequence #(ppb_item);
    `uvm_object_utils(ppb_sequence)
    int order[$];
    function new(string name = "ppb_sequence");
      super.new(name);
    endfunction
    task pre_body();
      order.push_back(1);
      `uvm_info("TEST", "pre_body called", UVM_NONE)
    endtask
    task body();
      ppb_item item;
      order.push_back(2);
      `uvm_info("TEST", "body called", UVM_NONE)
      item = ppb_item::type_id::create("item");
      item.data = 42;
      start_item(item);
      finish_item(item);
    endtask
    task post_body();
      order.push_back(3);
      `uvm_info("TEST", "post_body called", UVM_NONE)
    endtask
  endclass

  class ppb_driver extends uvm_driver #(ppb_item);
    `uvm_component_utils(ppb_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      ppb_item item;
      forever begin
        seq_item_port.get_next_item(item);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class ppb_test extends uvm_test;
    `uvm_component_utils(ppb_test)
    ppb_driver driver;
    uvm_sequencer #(ppb_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = ppb_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(ppb_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      ppb_sequence seq;
      phase.raise_objection(this);
      seq = ppb_sequence::type_id::create("seq");
      seq.start(seqr);
      if (seq.order.size() == 3 && seq.order[0] == 1 &&
          seq.order[1] == 2 && seq.order[2] == 3)
        `uvm_info("TEST", "sequence order: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "sequence order: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("ppb_test");
endmodule
