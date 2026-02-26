// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=50000000000 2>&1 | FileCheck %s
//
// CHECK: DRV_GOT_REQ
// CHECK: SEQ_DONE
// CHECK-NOT: UVM_FATAL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class demo_item extends uvm_sequence_item;
  `uvm_object_utils(demo_item)

  function new(string name = "demo_item");
    super.new(name);
  endfunction
endclass

class demo_seq extends uvm_sequence #(demo_item);
  `uvm_object_utils(demo_seq)

  function new(string name = "demo_seq");
    super.new(name);
  endfunction

  task body();
    demo_item req;
    req = demo_item::type_id::create("req");
    wait_for_grant();
    send_request(req);
    wait_for_item_done();
    $display("SEQ_DONE");
  endtask
endclass

class demo_driver extends uvm_driver #(demo_item);
  `uvm_component_utils(demo_driver)

  function new(string name = "demo_driver", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    demo_item req;
    seq_item_port.get_next_item(req);
    $display("DRV_GOT_REQ");
    seq_item_port.item_done();
  endtask
endclass

class demo_env extends uvm_env;
  `uvm_component_utils(demo_env)

  uvm_sequencer #(demo_item) sqr;
  demo_driver drv;

  function new(string name = "demo_env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    sqr = uvm_sequencer#(demo_item)::type_id::create("sqr", this);
    drv = demo_driver::type_id::create("drv", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    drv.seq_item_port.connect(sqr.seq_item_export);
  endfunction
endclass

class demo_test extends uvm_test;
  `uvm_component_utils(demo_test)

  demo_env env;

  function new(string name = "demo_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    env = demo_env::type_id::create("env", this);
  endfunction

  task run_phase(uvm_phase phase);
    demo_seq seq;
    phase.raise_objection(this);
    seq = demo_seq::type_id::create("seq");
    seq.start(env.sqr);
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("demo_test");
endmodule
