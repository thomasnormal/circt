// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=20000000000 2>&1 | FileCheck %s

// CHECK: DERIVED_SEQ_BODY
// CHECK-NOT: FCTTYP
// CHECK-NOT: Body definition undefined
// CHECK-NOT: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class demo_seq extends uvm_sequence #(uvm_sequence_item);
  `uvm_object_utils(demo_seq)

  function new(string name = "demo_seq");
    super.new(name);
  endfunction

  virtual task body();
    $display("DERIVED_SEQ_BODY");
  endtask
endclass

class demo_test extends uvm_test;
  `uvm_component_utils(demo_test)

  uvm_sequencer #(uvm_sequence_item) sqr;

  function new(string name = "demo_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    sqr = uvm_sequencer#(uvm_sequence_item)::type_id::create("sqr", this);
  endfunction

  task run_phase(uvm_phase phase);
    demo_seq seq;
    phase.raise_objection(this);
    seq = demo_seq::type_id::create("seq");
    seq.start(sqr);
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("demo_test");
endmodule
