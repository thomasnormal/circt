// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Test nested UVM objections across hierarchy.
// Verifies that parent raise + child drop correctly propagates,
// and that the phase waits until all objections are resolved.

// CHECK: [TEST] child raised objection
// CHECK: [TEST] parent raised objection
// CHECK: [TEST] parent dropped objection
// CHECK: [TEST] child dropped objection
// CHECK: [TEST] objection test complete
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

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("TEST", "child raised objection", UVM_LOW)
      #20ns;
      `uvm_info("TEST", "child dropped objection", UVM_LOW)
      phase.drop_objection(this);
    endtask
  endclass

  class objection_test extends uvm_test;
    `uvm_component_utils(objection_test)

    child_comp child;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      child = child_comp::type_id::create("child", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      `uvm_info("TEST", "parent raised objection", UVM_LOW)
      #10ns;
      `uvm_info("TEST", "parent dropped objection", UVM_LOW)
      phase.drop_objection(this);
    endtask

    function void report_phase(uvm_phase phase);
      super.report_phase(phase);
      `uvm_info("TEST", "objection test complete", UVM_LOW)
    endfunction
  endclass

  initial run_test("objection_test");
endmodule
