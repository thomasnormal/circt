// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: env CIRCT_UVM_ARGS='+ntb_random_seed=1' circt-sim %t.mlir --top top --max-time=3000000000 2>&1 | FileCheck %s
//
// Reproducer for sequencer cross-pop bug: after one valid dequeue from sqr_a,
// a second unresolved dequeue on the same port must not steal an item that was
// pushed only into sqr_b.
//
// CHECK: A_WAITING1
// CHECK: A_GOT1 type=item_a
// CHECK: A_WAITING2
// CHECK-NOT: A_GOT2 type=item_b
// CHECK-NOT: UVM_FATAL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class item_a extends uvm_sequence_item;
  `uvm_object_utils(item_a)
  function new(string name = "item_a");
    super.new(name);
  endfunction
endclass

class item_b extends uvm_sequence_item;
  `uvm_object_utils(item_b)
  function new(string name = "item_b");
    super.new(name);
  endfunction
endclass

class seq_a extends uvm_sequence #(item_a);
  `uvm_object_utils(seq_a)

  function new(string name = "seq_a");
    super.new(name);
  endfunction

  task body();
    item_a req;
    req = item_a::type_id::create("req_a");
    wait_for_grant();
    send_request(req);
    wait_for_item_done();
  endtask
endclass

class seq_b extends uvm_sequence #(item_b);
  `uvm_object_utils(seq_b)

  function new(string name = "seq_b");
    super.new(name);
  endfunction

  task body();
    item_b req;
    req = item_b::type_id::create("req_b");
    wait_for_grant();
    send_request(req);
    wait_for_item_done();
  endtask
endclass

class drv_a extends uvm_driver #(item_a);
  `uvm_component_utils(drv_a)

  function new(string name = "drv_a", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    item_a req1;
    item_a req2;

    $display("A_WAITING1");
    seq_item_port.get_next_item(req1);
    if (req1 == null)
      $display("A_GOT1 type=null");
    else
      $display("A_GOT1 type=%s", req1.get_type_name());
    seq_item_port.item_done();

    $display("A_WAITING2");
    seq_item_port.get_next_item(req2);
    if (req2 == null)
      $display("A_GOT2 type=null");
    else
      $display("A_GOT2 type=%s", req2.get_type_name());
    seq_item_port.item_done();
  endtask
endclass

class env2 extends uvm_env;
  `uvm_component_utils(env2)

  uvm_sequencer #(item_a) sqr_a;
  uvm_sequencer #(item_b) sqr_b;
  drv_a da;

  function new(string name = "env2", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    sqr_a = uvm_sequencer#(item_a)::type_id::create("sqr_a", this);
    sqr_b = uvm_sequencer#(item_b)::type_id::create("sqr_b", this);
    da = drv_a::type_id::create("da", this);
  endfunction

  function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    da.seq_item_port.connect(sqr_a.seq_item_export);
  endfunction
endclass

class test2 extends uvm_test;
  `uvm_component_utils(test2)

  env2 e;

  function new(string name = "test2", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    e = env2::type_id::create("e", this);
  endfunction

  task run_phase(uvm_phase phase);
    seq_a sa;
    seq_b sb;
    phase.raise_objection(this);
    sa = seq_a::type_id::create("sa");
    sb = seq_b::type_id::create("sb");
    sa.start(e.sqr_a);
    #1ns;
    sb.start(e.sqr_b);
    #1ns;
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("test2");
endmodule
