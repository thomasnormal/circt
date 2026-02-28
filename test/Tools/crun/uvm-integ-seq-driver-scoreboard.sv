// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: class method references module-scope clk — slang reports "unknown name `clk`"

// Integration: sequence → driver → analysis_port → scoreboard data path.

// CHECK: [TEST] scoreboard received 5 items in order
// CHECK: [TEST] seq-driver-scoreboard: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  bit clk;
  always #5 clk = ~clk;

  class integ_sds_item extends uvm_sequence_item;
    `uvm_object_utils(integ_sds_item)
    int data;
    function new(string name = "integ_sds_item");
      super.new(name);
    endfunction
  endclass

  class integ_sds_seq extends uvm_sequence #(integ_sds_item);
    `uvm_object_utils(integ_sds_seq)
    function new(string name = "integ_sds_seq");
      super.new(name);
    endfunction
    task body();
      integ_sds_item item;
      for (int i = 0; i < 5; i++) begin
        item = integ_sds_item::type_id::create($sformatf("item_%0d", i));
        item.data = i * 10;
        start_item(item);
        finish_item(item);
      end
    endtask
  endclass

  class integ_sds_scoreboard extends uvm_subscriber #(integ_sds_item);
    `uvm_component_utils(integ_sds_scoreboard)
    int count;
    int in_order;
    function new(string name, uvm_component parent);
      super.new(name, parent);
      count = 0;
      in_order = 1;
    endfunction
    function void write(integ_sds_item t);
      if (t.data != count * 10) in_order = 0;
      count++;
    endfunction
  endclass

  class integ_sds_driver extends uvm_driver #(integ_sds_item);
    `uvm_component_utils(integ_sds_driver)
    uvm_analysis_port #(integ_sds_item) ap;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      ap = new("ap", this);
    endfunction
    task run_phase(uvm_phase phase);
      integ_sds_item item;
      forever begin
        seq_item_port.get_next_item(item);
        @(posedge clk);
        ap.write(item);
        seq_item_port.item_done();
      end
    endtask
  endclass

  class integ_sds_test extends uvm_test;
    `uvm_component_utils(integ_sds_test)
    uvm_sequencer #(integ_sds_item) sqr;
    integ_sds_driver drv;
    integ_sds_scoreboard sb;
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      sqr = uvm_sequencer#(integ_sds_item)::type_id::create("sqr", this);
      drv = integ_sds_driver::type_id::create("drv", this);
      sb = integ_sds_scoreboard::type_id::create("sb", this);
    endfunction
    function void connect_phase(uvm_phase phase);
      drv.seq_item_port.connect(sqr.seq_item_export);
      drv.ap.connect(sb.analysis_export);
    endfunction
    task run_phase(uvm_phase phase);
      integ_sds_seq seq;
      phase.raise_objection(this);
      seq = integ_sds_seq::type_id::create("seq");
      seq.start(sqr);
      #10;
      if (sb.count == 5 && sb.in_order)
        `uvm_info("TEST", "scoreboard received 5 items in order", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("scoreboard: count=%0d in_order=%0d", sb.count, sb.in_order))
      if (sb.count == 5 && sb.in_order)
        `uvm_info("TEST", "seq-driver-scoreboard: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "seq-driver-scoreboard: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("integ_sds_test");
endmodule
