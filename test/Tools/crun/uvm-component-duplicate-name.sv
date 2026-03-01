// RUN: crun %s --uvm-path=%S/../../../lib/Runtime/uvm-core --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: two children with same name under same parent.
// UVM should report CLDEXT fatal.

// CHECK: [CLDEXT] Cannot set 'same_name' as a child of 'uvm_test_top', which already has a child by that name.
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

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
      uvm_report_server srv;
      super.build_phase(phase);
      srv = uvm_report_server::get_server();
      srv.set_max_quit_count(100);

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
