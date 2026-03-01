// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=50000000000 2>&1 | FileCheck %s
//
// Regression: parallel sequences on different sequencers must each complete
// wait_for_grant -> send_request -> wait_for_item_done without deadlock.
//
// CHECK: DRV_GOT_REQ A
// CHECK: DRV_GOT_REQ B
// CHECK: SEQ_DONE seqA
// CHECK: SEQ_DONE seqB
// CHECK: TEST_DONE
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

class child_seq extends uvm_sequence #(demo_item);
  `uvm_object_utils(child_seq)

  function new(string name = "child_seq");
    super.new(name);
  endfunction

  task body();
    demo_item req;
    req = demo_item::type_id::create({get_name(), "_req"});
    wait_for_grant();
    send_request(req);
    wait_for_item_done();
    $display("SEQ_DONE %s", get_name());
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
    $display("DRV_GOT_REQ %s", get_name());
    seq_item_port.item_done();
  endtask
endclass

class demo_env extends uvm_env;
  `uvm_component_utils(demo_env)

  uvm_sequencer #(demo_item) sqrA;
  uvm_sequencer #(demo_item) sqrB;
  demo_driver drvA;
  demo_driver drvB;

  function new(string name = "demo_env", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    sqrA = uvm_sequencer#(demo_item)::type_id::create("sqrA", this);
    sqrB = uvm_sequencer#(demo_item)::type_id::create("sqrB", this);
    drvA = demo_driver::type_id::create("A", this);
    drvB = demo_driver::type_id::create("B", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    drvA.seq_item_port.connect(sqrA.seq_item_export);
    drvB.seq_item_port.connect(sqrB.seq_item_export);
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
    child_seq seqA;
    child_seq seqB;
    phase.raise_objection(this);
    seqA = child_seq::type_id::create("seqA");
    seqB = child_seq::type_id::create("seqB");
    fork
      seqA.start(env.sqrA);
      seqB.start(env.sqrB);
    join
    $display("TEST_DONE");
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("demo_test");
endmodule
