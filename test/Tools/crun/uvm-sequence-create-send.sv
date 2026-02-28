// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_create and `uvm_send macros.

// CHECK: [TEST] driver received data: 55
// CHECK: [TEST] create/send: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class cs_item extends uvm_sequence_item;
    `uvm_object_utils(cs_item)
    int data;
    function new(string name = "cs_item");
      super.new(name);
    endfunction
  endclass

  class cs_sequence extends uvm_sequence #(cs_item);
    `uvm_object_utils(cs_sequence)
    function new(string name = "cs_sequence");
      super.new(name);
    endfunction
    task body();
      cs_item item;
      `uvm_create(item)
      item.data = 55;
      `uvm_send(item)
    endtask
  endclass

  class cs_driver extends uvm_driver #(cs_item);
    `uvm_component_utils(cs_driver)
    int received_data;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      received_data = -1;
    endfunction
    task run_phase(uvm_phase phase);
      cs_item item;
      forever begin
        seq_item_port.get_next_item(item);
        received_data = item.data;
        @(posedge clk);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class cs_test extends uvm_test;
    `uvm_component_utils(cs_test)
    cs_driver driver;
    uvm_sequencer #(cs_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = cs_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(cs_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      cs_sequence seq;
      phase.raise_objection(this);
      seq = cs_sequence::type_id::create("seq");
      seq.start(seqr);
      @(posedge clk);
      `uvm_info("TEST", $sformatf("driver received data: %0d", driver.received_data), UVM_LOW)
      if (driver.received_data == 55)
        `uvm_info("TEST", "create/send: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("create/send: FAIL (got %0d)", driver.received_data))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("cs_test");
endmodule
