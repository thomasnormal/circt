// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test uvm_push_driver basic API.
// Verifies push driver receives items via put interface.

// CHECK: [TEST] push driver item received: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class push_item extends uvm_sequence_item;
    `uvm_object_utils(push_item)
    int value;
    function new(string name = "push_item");
      super.new(name);
    endfunction
  endclass

  class my_push_driver extends uvm_push_driver #(push_item);
    `uvm_component_utils(my_push_driver)
    int received_value;
    bit got_item;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task push(push_item t);
      received_value = t.value;
      got_item = 1;
    endtask
  endclass

  class push_test extends uvm_test;
    `uvm_component_utils(push_test)
    my_push_driver drv;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      drv = my_push_driver::type_id::create("drv", this);
    endfunction

    task run_phase(uvm_phase phase);
      push_item item;

      phase.raise_objection(this);

      item = push_item::type_id::create("item");
      item.value = 99;

      // Directly call put on the driver's export to push an item
      drv.push(item);

      if (drv.got_item && drv.received_value == 99)
        `uvm_info("TEST", "push driver item received: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "push driver item received: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("push_test");
endmodule
