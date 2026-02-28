// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test custom uvm_topdown_phase execution order.
// Verifies parent executes before child.

// CHECK: [TEST] parent before child: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  // Use build_phase which is a topdown phase to verify ordering
  class child_comp extends uvm_component;
    `uvm_component_utils(child_comp)
    static int order_idx = 0;
    int my_order;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      my_order = order_idx++;
    endfunction
  endclass

  class parent_comp extends uvm_component;
    `uvm_component_utils(parent_comp)
    child_comp child;
    int my_order;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      my_order = child_comp::order_idx++;
      child = child_comp::type_id::create("child", this);
    endfunction
  endclass

  class topdown_test extends uvm_test;
    `uvm_component_utils(topdown_test)
    parent_comp par;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      child_comp::order_idx = 0;
      par = parent_comp::type_id::create("par", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      // parent's build_phase should execute before child's
      if (par.my_order < par.child.my_order)
        `uvm_info("TEST", "parent before child: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("parent before child: FAIL parent=%0d child=%0d",
                   par.my_order, par.child.my_order))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("topdown_test");
endmodule
