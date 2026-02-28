// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Probe: verify build_phase executes top-down (parent before child).
// No custom phase creation — just verify built-in ordering.

// CHECK: [TEST] parent build first: PASS
// CHECK: [TEST] child build second: PASS
// CHECK: [TEST] top-down order: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  int build_order[$];

  class probe_child extends uvm_component;
    `uvm_component_utils(probe_child)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      build_order.push_back(2);
      `uvm_info("TEST", "child build second: PASS", UVM_LOW)
    endfunction
  endclass

  class probe_phase_test extends uvm_test;
    `uvm_component_utils(probe_phase_test)
    probe_child child;

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    function void build_phase(uvm_phase phase);
      super.build_phase(phase);
      build_order.push_back(1);
      `uvm_info("TEST", "parent build first: PASS", UVM_LOW)
      child = probe_child::type_id::create("child", this);
    endfunction

    task run_phase(uvm_phase phase);
      phase.raise_objection(this);
      if (build_order.size() == 2 && build_order[0] == 1 && build_order[1] == 2)
        `uvm_info("TEST", "top-down order: PASS", UVM_LOW)
      else
        `uvm_error("TEST", "top-down order: FAIL")
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("probe_phase_test");
endmodule
