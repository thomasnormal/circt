// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_driver with sequencer, get_next_item/item_done flow.

// CHECK: [TEST] driver received item 0: data=10
// CHECK: [TEST] driver received item 1: data=20
// CHECK: [TEST] driver received item 2: data=30
// CHECK: [TEST] driver got 3 items: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class drv_item extends uvm_sequence_item;
    `uvm_object_utils(drv_item)
    int data;
    function new(string name = "drv_item");
      super.new(name);
    endfunction
  endclass

  class drv_sequence extends uvm_sequence #(drv_item);
    `uvm_object_utils(drv_sequence)
    function new(string name = "drv_sequence");
      super.new(name);
    endfunction
    task body();
      drv_item item;
      for (int i = 0; i < 3; i++) begin
        item = drv_item::type_id::create($sformatf("item_%0d", i));
        item.data = (i + 1) * 10;
        start_item(item);
        finish_item(item);
      end
    endtask
  endclass

  class basic_driver extends uvm_driver #(drv_item);
    `uvm_component_utils(basic_driver)
    int count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      count = 0;
    endfunction
    task run_phase(uvm_phase phase);
      drv_item item;
      forever begin
        seq_item_port.get_next_item(item);
        @(posedge clk);
        `uvm_info("TEST", $sformatf("driver received item %0d: data=%0d", count, item.data), UVM_LOW)
        count++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class drv_test extends uvm_test;
    `uvm_component_utils(drv_test)
    basic_driver driver;
    uvm_sequencer #(drv_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = basic_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(drv_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      drv_sequence seq;
      phase.raise_objection(this);
      seq = drv_sequence::type_id::create("seq");
      seq.start(seqr);
      @(posedge clk);
      if (driver.count == 3)
        `uvm_info("TEST", "driver got 3 items: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("driver got %0d items", driver.count))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("drv_test");
endmodule
