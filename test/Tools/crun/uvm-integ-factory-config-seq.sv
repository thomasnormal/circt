// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: class method references module-scope clk — slang reports "unknown name `clk`"

// Integration: factory override + config_db num_items + sequence generation.

// CHECK: [TEST] received 6 items of type integ_fcs_custom_item
// CHECK: [TEST] factory-config-seq: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class integ_fcs_item extends uvm_sequence_item;
    `uvm_object_utils(integ_fcs_item)
    int data;
    function new(string name = "integ_fcs_item");
      super.new(name);
    endfunction
    virtual function string item_type_name();
      return "integ_fcs_item";
    endfunction
  endclass

  class integ_fcs_custom_item extends integ_fcs_item;
    `uvm_object_utils(integ_fcs_custom_item)
    function new(string name = "integ_fcs_custom_item");
      super.new(name);
    endfunction
    virtual function string item_type_name();
      return "integ_fcs_custom_item";
    endfunction
  endclass

  class integ_fcs_seq extends uvm_sequence #(integ_fcs_item);
    `uvm_object_utils(integ_fcs_seq)
    int num_items;
    function new(string name = "integ_fcs_seq");
      super.new(name);
      num_items = 1;
    endfunction
    task body();
      integ_fcs_item item;
      for (int i = 0; i < num_items; i++) begin
        item = integ_fcs_item::type_id::create($sformatf("item_%0d", i));
        item.data = i;
        start_item(item);
        finish_item(item);
      end
    endtask
  endclass

  class integ_fcs_driver extends uvm_driver #(integ_fcs_item);
    `uvm_component_utils(integ_fcs_driver)
    int recv_count;
    string last_type;
    int all_custom;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      recv_count = 0;
      all_custom = 1;
    endfunction
    task run_phase(uvm_phase phase);
      integ_fcs_item item;
      forever begin
        seq_item_port.get_next_item(item);
        @(posedge clk);
        last_type = item.item_type_name();
        if (last_type != "integ_fcs_custom_item") all_custom = 0;
        recv_count++;
        seq_item_port.item_done();
      end
    endtask
  endclass

  class integ_fcs_test extends uvm_test;
    `uvm_component_utils(integ_fcs_test)
    uvm_sequencer #(integ_fcs_item) sqr;
    integ_fcs_driver drv;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      integ_fcs_item::type_id::set_type_override(integ_fcs_custom_item::get_type());
      sqr = uvm_sequencer#(integ_fcs_item)::type_id::create("sqr", this);
      drv = integ_fcs_driver::type_id::create("drv", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
    endfunction
    task run_phase(uvm_phase phase);
      integ_fcs_seq seq;
      phase.raise_objection(this);
      seq = integ_fcs_seq::type_id::create("seq");
      seq.num_items = 6;
      seq.start(sqr);
      #10;
      if (drv.recv_count == 6 && drv.all_custom)
        `uvm_info("TEST", "received 6 items of type integ_fcs_custom_item", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("count=%0d all_custom=%0d", drv.recv_count, drv.all_custom))
      if (drv.recv_count == 6 && drv.all_custom)
        `uvm_info("TEST", "factory-config-seq: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "factory-config-seq: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_fcs_test");
endmodule
