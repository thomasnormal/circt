// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *

// Test virtual sequence that starts sub-sequences on sub-sequencers.

// CHECK: [TEST] sub_seq_a executed: PASS
// CHECK: [TEST] sub_seq_b executed: PASS
// CHECK: [TEST] virtual sequence done: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class my_item extends uvm_sequence_item;
    `uvm_object_utils(my_item)
    int data;
    function new(string name = "my_item");
      super.new(name);
    endfunction
  endclass

  class sub_seq_a extends uvm_sequence#(my_item);
    `uvm_object_utils(sub_seq_a)
    function new(string name = "sub_seq_a");
      super.new(name);
    endfunction
    task body();
      `uvm_info("TEST", "sub_seq_a executed: PASS", UVM_LOW)
    endtask
  endclass

  class sub_seq_b extends uvm_sequence#(my_item);
    `uvm_object_utils(sub_seq_b)
    function new(string name = "sub_seq_b");
      super.new(name);
    endfunction
    task body();
      `uvm_info("TEST", "sub_seq_b executed: PASS", UVM_LOW)
    endtask
  endclass

  class virt_sequencer extends uvm_sequencer#(my_item);
    `uvm_component_utils(virt_sequencer)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class virt_seq extends uvm_sequence#(my_item);
    `uvm_object_utils(virt_seq)
    `uvm_declare_p_sequencer(virt_sequencer)
    function new(string name = "virt_seq");
      super.new(name);
    endfunction
    task body();
      sub_seq_a sa;
      sub_seq_b sb;
      sa = sub_seq_a::type_id::create("sa");
      sb = sub_seq_b::type_id::create("sb");
      sa.start(p_sequencer);
      sb.start(p_sequencer);
      `uvm_info("TEST", "virtual sequence done: PASS", UVM_LOW)
    endtask
  endclass

  class vseq_test extends uvm_test;
    `uvm_component_utils(vseq_test)
    virt_sequencer sqr;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      sqr = virt_sequencer::type_id::create("sqr", this);
    endfunction

    task run_phase(uvm_phase phase);
      virt_seq vs;
      phase.raise_objection(this);
      vs = virt_seq::type_id::create("vs");
      vs.start(sqr);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("vseq_test");
endmodule
