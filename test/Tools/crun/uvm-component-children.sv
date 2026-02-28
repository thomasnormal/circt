// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test component child iteration: get_first_child, get_next_child,
// get_num_children, get_child.

// CHECK: [TEST] get_num_children: PASS
// CHECK: [TEST] get_child by name: PASS
// CHECK: [TEST] iterate children count: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class child_comp extends uvm_component;
    `uvm_component_utils(child_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class children_test extends uvm_test;
    `uvm_component_utils(children_test)
    child_comp c0, c1, c2, c3;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      c0 = child_comp::type_id::create("child_0", this);
      c1 = child_comp::type_id::create("child_1", this);
      c2 = child_comp::type_id::create("child_2", this);
      c3 = child_comp::type_id::create("child_3", this);
    endfunction

    task run_phase(uvm_phase phase);
      string name;
      uvm_component child;
      int count;

      phase.raise_objection(this);

      // Test 1: get_num_children
      if (get_num_children() == 4)
        `uvm_info("TEST", "get_num_children: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("get_num_children: FAIL (got %0d)", get_num_children()))

      // Test 2: get_child by name
      child = get_child("child_2");
      if (child == c2)
        `uvm_info("TEST", "get_child by name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get_child by name: FAIL")

      // Test 3: iterate with get_first_child / get_next_child
      count = 0;
      if (get_first_child(name)) begin
        count++;
        while (get_next_child(name))
          count++;
      end
      if (count == 4)
        `uvm_info("TEST", "iterate children count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("iterate children count: FAIL (got %0d)", count))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("children_test");
endmodule
