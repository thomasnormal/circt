// RUN: crun %s --uvm-path=%S/../../../lib/Runtime/uvm-core --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test UVM component hierarchy navigation:
// get_parent, get_full_name, get_num_children, get_child, lookup.

// CHECK: [TEST] get_full_name: PASS
// CHECK: [TEST] get_parent: PASS
// CHECK: [TEST] get_num_children: PASS
// CHECK: [TEST] get_child: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class leaf_comp extends uvm_component;
    `uvm_component_utils(leaf_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class mid_comp extends uvm_component;
    `uvm_component_utils(mid_comp)
    leaf_comp leaf_a;
    leaf_comp leaf_b;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      leaf_a = leaf_comp::type_id::create("leaf_a", this);
      leaf_b = leaf_comp::type_id::create("leaf_b", this);
    endfunction
  endclass

  class hierarchy_test extends uvm_test;
    `uvm_component_utils(hierarchy_test)
    mid_comp mid;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      mid = mid_comp::type_id::create("mid", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_component child;

      phase.raise_objection(this);

      // Test 1: get_full_name
      if (mid.leaf_a.get_full_name() == "uvm_test_top.mid.leaf_a")
        `uvm_info("TEST", "get_full_name: PASS", UVM_LOW)
      else
        `uvm_error("TEST", {"get_full_name: FAIL — got: ", mid.leaf_a.get_full_name()})

      // Test 2: get_parent
      if (mid.leaf_a.get_parent() == mid)
        `uvm_info("TEST", "get_parent: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get_parent: FAIL")

      // Test 3: get_num_children
      if (mid.get_num_children() == 2)
        `uvm_info("TEST", "get_num_children: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("get_num_children: FAIL — got %0d", mid.get_num_children()))

      // Test 4: get_child
      child = mid.get_child("leaf_b");
      if (child == mid.leaf_b)
        `uvm_info("TEST", "get_child: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "get_child: FAIL")

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("hierarchy_test");
endmodule
