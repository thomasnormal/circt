// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: two children with same name under same parent.
// UVM should report CLDEXT fatal.  We use a report catcher to demote
// the fatal to a warning so the simulation can continue (UVM 1.1d
// terminates on UVM_FATAL regardless of set_max_quit_count).

// CHECK: [CLDEXT] Cannot set 'same_name' as a child of 'uvm_test_top', which already has a child by that name.
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class cldext_catcher extends uvm_report_catcher;
    function new(string name = "cldext_catcher");
      super.new(name);
    endfunction
    function action_e catch();
      if (get_id() == "CLDEXT")
        set_severity(UVM_WARNING);
      return THROW;
    endfunction
  endclass

  class neg_dup_child extends uvm_component;
    `uvm_component_utils(neg_dup_child)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class neg_comp_dup_name_test extends uvm_test;
    `uvm_component_utils(neg_comp_dup_name_test)
    neg_dup_child child1;
    neg_dup_child child2;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      cldext_catcher catcher;
      super.build_phase(phase);

      // Install catcher BEFORE the duplicate create
      catcher = new();
      uvm_report_cb::add(null, catcher);

      // Create two children with the same name
      child1 = neg_dup_child::type_id::create("same_name", this);
      child2 = neg_dup_child::type_id::create("same_name", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      `uvm_info("TEST", "duplicate name survived: PASS", UVM_LOW)

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_comp_dup_name_test");
endmodule
