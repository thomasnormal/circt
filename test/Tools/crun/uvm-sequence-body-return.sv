// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test sequence body() with early return.

// CHECK: [TEST] early return seq completed: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_ret_item extends uvm_sequence_item;
    `uvm_object_utils(edge_ret_item)
    int idx;
    function new(string name = "edge_ret_item");
      super.new(name);
    endfunction
  endclass

  class edge_early_ret_seq extends uvm_sequence #(edge_ret_item);
    `uvm_object_utils(edge_early_ret_seq)
    int items_sent;

    function new(string name = "edge_early_ret_seq");
      super.new(name);
      items_sent = 0;
    endfunction

    task body();
      edge_ret_item item;
      for (int i = 0; i < 10; i++) begin
        if (i >= 3) return; // Early return after 3 items
        item = edge_ret_item::type_id::create($sformatf("item_%0d", i));
        item.idx = i;
        start_item(item);
        finish_item(item);
        items_sent++;
      end
    endtask
  endclass

  class edge_ret_driver extends uvm_driver #(edge_ret_item);
    `uvm_component_utils(edge_ret_driver)
    int received;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      received = 0;
    endfunction
    task run_phase(uvm_phase phase);
      forever begin
        edge_ret_item item;
        seq_item_port.get_next_item(item);
        received++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class edge_ret_env extends uvm_env;
    `uvm_component_utils(edge_ret_env)
    uvm_sequencer #(edge_ret_item) sqr;
    edge_ret_driver drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sqr = uvm_sequencer#(edge_ret_item)::type_id::create("sqr", this);
      drv = edge_ret_driver::type_id::create("drv", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
  endclass

  class edge_body_ret_test extends uvm_test;
    `uvm_component_utils(edge_body_ret_test)
    edge_ret_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = edge_ret_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      edge_early_ret_seq seq;
      phase.raise_objection(this);
      seq = edge_early_ret_seq::type_id::create("seq");
      seq.start(env.sqr);
      `uvm_info("TEST", "early return seq completed: PASS", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_body_ret_test");
endmodule
