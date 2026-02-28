// RUN: circt-verilog %s --ir-hw 2>/dev/null | circt-sim - --top top --max-time 200000000 2>&1 | FileCheck %s

// Regression: run_phase cleanup must complete when a driver blocks forever in
// get_next_item after the sequence is exhausted.
// CHECK: DROP_DONE
// CHECK: REPORT_DONE
// CHECK-NOT: Main loop exit: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class my_item extends uvm_sequence_item;
  rand bit [7:0] data;
  `uvm_object_utils(my_item)
  function new(string name = "my_item");
    super.new(name);
  endfunction
endclass

class my_seq extends uvm_sequence #(my_item);
  `uvm_object_utils(my_seq)
  function new(string name = "my_seq");
    super.new(name);
  endfunction

  task body();
    my_item req;
    repeat (2) begin
      req = my_item::type_id::create("req");
      start_item(req);
      assert(req.randomize());
      finish_item(req);
    end
  endtask
endclass

class my_driver extends uvm_driver #(my_item);
  `uvm_component_utils(my_driver)
  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      seq_item_port.get_next_item(req);
      seq_item_port.item_done();
    end
  endtask
endclass

class my_env extends uvm_env;
  `uvm_component_utils(my_env)
  uvm_sequencer #(my_item) seqr;
  my_driver drv;

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    seqr = uvm_sequencer#(my_item)::type_id::create("seqr", this);
    drv = my_driver::type_id::create("drv", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    drv.seq_item_port.connect(seqr.seq_item_export);
  endfunction
endclass

class my_test extends uvm_test;
  `uvm_component_utils(my_test)
  my_env env;

  function new(string name = "my_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = my_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    my_seq s;
    phase.raise_objection(this);
    s = my_seq::type_id::create("s");
    s.start(env.seqr);
    phase.drop_objection(this);
    $display("DROP_DONE");
  endtask

  function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    $display("REPORT_DONE");
  endfunction
endclass

module top;
  initial run_test("my_test");
endmodule
