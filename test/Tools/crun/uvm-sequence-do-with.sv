// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test `uvm_do_with macro for constrained sequence item generation.

// CHECK: [TEST] item data: 42
// CHECK: [TEST] uvm_do_with: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class dw_item extends uvm_sequence_item;
    `uvm_object_utils(dw_item)
    rand int data;
    function new(string name = "dw_item");
      super.new(name);
    endfunction
  endclass

  class dw_sequence extends uvm_sequence #(dw_item);
    `uvm_object_utils(dw_sequence)
    int captured_data;
    function new(string name = "dw_sequence");
      super.new(name);
    endfunction
    task body();
      dw_item item;
      `uvm_do_with(item, {data == 42;})
      captured_data = item.data;
    endtask
  endclass

  class dw_driver extends uvm_driver #(dw_item);
    `uvm_component_utils(dw_driver)
    int received_data;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      dw_item item;
      forever begin
        seq_item_port.get_next_item(item);
        received_data = item.data;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class dw_test extends uvm_test;
    `uvm_component_utils(dw_test)
    dw_driver driver;
    uvm_sequencer #(dw_item) seqr;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      driver = dw_driver::type_id::create("driver", this);
      seqr = uvm_sequencer#(dw_item)::type_id::create("seqr", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      driver.seq_item_port.connect(seqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      dw_sequence seq;
      phase.raise_objection(this);
      seq = dw_sequence::type_id::create("seq");
      seq.start(seqr);
      `uvm_info("TEST", $sformatf("item data: %0d", seq.captured_data), UVM_LOW)
      if (seq.captured_data == 42)
        `uvm_info("TEST", "uvm_do_with: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("uvm_do_with: FAIL (got %0d)", seq.captured_data))
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("dw_test");
endmodule
