// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test 3-level nested sequences: A starts B starts C.

// CHECK: [TEST] seq_c body ran: PASS
// CHECK: [TEST] seq_b body ran: PASS
// CHECK: [TEST] seq_a body ran: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_item extends uvm_sequence_item;
    `uvm_object_utils(edge_item)
    int level;
    function new(string name = "edge_item");
      super.new(name);
    endfunction
  endclass

  class edge_seq_c extends uvm_sequence #(edge_item);
    `uvm_object_utils(edge_seq_c)
    bit done;
    function new(string name = "edge_seq_c");
      super.new(name);
      done = 0;
    endfunction
    task body();
      edge_item item;
      item = edge_item::type_id::create("item_c");
      item.level = 3;
      start_item(item);
      finish_item(item);
      done = 1;
    endtask
  endclass

  class edge_seq_b extends uvm_sequence #(edge_item);
    `uvm_object_utils(edge_seq_b)
    bit done;
    edge_seq_c seq_c;
    function new(string name = "edge_seq_b");
      super.new(name);
      done = 0;
    endfunction
    task body();
      seq_c = edge_seq_c::type_id::create("seq_c");
      seq_c.start(m_sequencer);
      done = 1;
    endtask
  endclass

  class edge_seq_a extends uvm_sequence #(edge_item);
    `uvm_object_utils(edge_seq_a)
    bit done;
    edge_seq_b seq_b;
    function new(string name = "edge_seq_a");
      super.new(name);
      done = 0;
    endfunction
    task body();
      seq_b = edge_seq_b::type_id::create("seq_b");
      seq_b.start(m_sequencer);
      done = 1;
    endtask
  endclass

  class edge_nest_driver extends uvm_driver #(edge_item);
    `uvm_component_utils(edge_nest_driver)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    task run_phase(uvm_phase phase);
      forever begin
        edge_item item;
        seq_item_port.get_next_item(item);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class edge_nest_env extends uvm_env;
    `uvm_component_utils(edge_nest_env)
    uvm_sequencer #(edge_item) sqr;
    edge_nest_driver drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sqr = uvm_sequencer#(edge_item)::type_id::create("sqr", this);
      drv = edge_nest_driver::type_id::create("drv", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
  endclass

  class edge_seq_nest_test extends uvm_test;
    `uvm_component_utils(edge_seq_nest_test)
    edge_nest_env env;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      env = edge_nest_env::type_id::create("env", this);
    endfunction
    task run_phase(uvm_phase phase);
      edge_seq_a seq;
      phase.raise_objection(this);
      seq = edge_seq_a::type_id::create("seq_a");
      seq.start(env.sqr);
      if (seq.seq_b.seq_c.done)
        `uvm_info("TEST", "seq_c body ran: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq_c body ran: FAIL")
      if (seq.seq_b.done)
        `uvm_info("TEST", "seq_b body ran: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq_b body ran: FAIL")
      if (seq.done)
        `uvm_info("TEST", "seq_a body ran: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq_a body ran: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_seq_nest_test");
endmodule
