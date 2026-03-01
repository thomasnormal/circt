// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
// XFAIL: *
// Reason: find_all() is not available in our UVM library version

// Test find_all component lookup with patterns.

// CHECK: [TEST] find all children: PASS
// CHECK: [TEST] direct child count: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class edge_leaf_comp extends uvm_component;
    `uvm_component_utils(edge_leaf_comp)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class edge_lookup_test extends uvm_test;
    `uvm_component_utils(edge_lookup_test)
    edge_leaf_comp comp_a, comp_ab, comp_abc, comp_b, comp_bc;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      comp_a   = edge_leaf_comp::type_id::create("comp_a", this);
      comp_ab  = edge_leaf_comp::type_id::create("comp_ab", this);
      comp_abc = edge_leaf_comp::type_id::create("comp_abc", this);
      comp_b   = edge_leaf_comp::type_id::create("comp_b", this);
      comp_bc  = edge_leaf_comp::type_id::create("comp_bc", this);
    endfunction

    task run_phase(uvm_phase phase);
      uvm_component comps[$];
      phase.raise_objection(this);

      // Find all children using wildcard "*"
      find_all("*", comps);
      if (comps.size() == 5)
        `uvm_info("TEST", "find all children: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("find all children: FAIL (got %0d)", comps.size()))

      // get_num_children
      if (get_num_children() == 5)
        `uvm_info("TEST", "direct child count: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("direct child count: FAIL (got %0d)", get_num_children()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("edge_lookup_test");
endmodule
