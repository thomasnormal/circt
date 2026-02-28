// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time 300000000 2>&1 | FileCheck %s

// Regression: by-type override must still apply when create() happens in
// run_phase and the created object is randomized afterwards.
// CHECK: TYPE_OK
// CHECK-NOT: TYPE_FAIL

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class base_item extends uvm_sequence_item;
  `uvm_object_utils(base_item)
  rand logic [3:0] addr;

  function new(string name = "base_item");
    super.new(name);
  endfunction
endclass

class ovr_item extends base_item;
  `uvm_object_utils(ovr_item)
  constraint edge_c { addr inside {4'd0, 4'd15}; }

  function new(string name = "ovr_item");
    super.new(name);
  endfunction
endclass

class test_t extends uvm_test;
  `uvm_component_utils(test_t)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    base_item::type_id::set_type_override(ovr_item::get_type());
  endfunction

  task run_phase(uvm_phase phase);
    base_item it;
    phase.raise_objection(this);

    it = base_item::type_id::create("it");
    if (it.get_type_name() == "ovr_item")
      $display("TYPE_OK");
    else
      $display("TYPE_FAIL:%s", it.get_type_name());

    void'(it.randomize());
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("test_t");
endmodule
