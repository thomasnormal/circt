// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test get_type_name() returns the registered type name from `uvm_component_utils.

// CHECK: [TEST] type name correct: PASS
// CHECK: [TEST] child type name correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class named_child extends uvm_component;
    `uvm_component_utils(named_child)
    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction
  endclass

  class type_name_test extends uvm_test;
    `uvm_component_utils(type_name_test)
    named_child child;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      child = named_child::type_id::create("child", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);

      if (get_type_name() == "type_name_test")
        `uvm_info("TEST", "type name correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("type name: FAIL (got %s)", get_type_name()))

      if (child.get_type_name() == "named_child")
        `uvm_info("TEST", "child type name correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("child type name: FAIL (got %s)", child.get_type_name()))

      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("type_name_test");
endmodule
