// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Integration: two agents with independent sequencers running in parallel.

// CHECK: [TEST] agent_a produced 4 items
// CHECK: [TEST] agent_b produced 3 items
// CHECK: [TEST] multi-agent: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class integ_ma_item extends uvm_sequence_item;
    `uvm_object_utils(integ_ma_item)
    int tag;
    function new(string name = "integ_ma_item");
      super.new(name);
    endfunction
  endclass

  class integ_ma_seq extends uvm_sequence #(integ_ma_item);
    `uvm_object_utils(integ_ma_seq)
    int num_items;
    function new(string name = "integ_ma_seq");
      super.new(name);
      num_items = 1;
    endfunction
    task body();
      integ_ma_item item;
      for (int i = 0; i < num_items; i++) begin
        item = integ_ma_item::type_id::create($sformatf("item_%0d", i));
        item.tag = i;
        start_item(item);
        finish_item(item);
      end
    endtask
  endclass

  class integ_ma_driver extends uvm_driver #(integ_ma_item);
    `uvm_component_utils(integ_ma_driver)
    int driven_count;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      driven_count = 0;
    endfunction
    task run_phase(uvm_phase phase);
      integ_ma_item item;
      forever begin
        seq_item_port.get_next_item(item);
        @(posedge clk);
        driven_count++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class integ_ma_agent extends uvm_agent;
    `uvm_component_utils(integ_ma_agent)
    uvm_sequencer #(integ_ma_item) sqr;
    integ_ma_driver drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sqr = uvm_sequencer#(integ_ma_item)::type_id::create("sqr", this);
      drv = integ_ma_driver::type_id::create("drv", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
  endclass

  class integ_ma_test extends uvm_test;
    `uvm_component_utils(integ_ma_test)
    integ_ma_agent agent_a;
    integ_ma_agent agent_b;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      agent_a = integ_ma_agent::type_id::create("agent_a", this);
      agent_b = integ_ma_agent::type_id::create("agent_b", this);
    endfunction
    task run_phase(uvm_phase phase);
      integ_ma_seq seq_a, seq_b;
      phase.raise_objection(this);
      seq_a = integ_ma_seq::type_id::create("seq_a");
      seq_a.num_items = 4;
      seq_b = integ_ma_seq::type_id::create("seq_b");
      seq_b.num_items = 3;
      fork
        seq_a.start(agent_a.sqr);
        seq_b.start(agent_b.sqr);
      join
      #10;
      if (agent_a.drv.driven_count == 4)
        `uvm_info("TEST", "agent_a produced 4 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("agent_a count=%0d", agent_a.drv.driven_count))
      if (agent_b.drv.driven_count == 3)
        `uvm_info("TEST", "agent_b produced 3 items", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("agent_b count=%0d", agent_b.drv.driven_count))
      if (agent_a.drv.driven_count == 4 && agent_b.drv.driven_count == 3)
        `uvm_info("TEST", "multi-agent: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "multi-agent: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_ma_test");
endmodule
