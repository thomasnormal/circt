// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm

// Negative test: raise objection twice without dropping. Count should be 2.

// CHECK: [TEST] double raise count correct: PASS
// CHECK: [circt-sim] Simulation completed

`timescale 1ns/1ps
`include "uvm_macros.svh"

module tb_top;
  import uvm_pkg::*;

  class neg_phase_double_raise_test extends uvm_test;
    `uvm_component_utils(neg_phase_double_raise_test)

    function new(string name, uvm_component parent);
      super.new(name, parent);
    endfunction

    task run_phase(uvm_phase phase);
      int count;
      phase.raise_objection(this);
      phase.raise_objection(this);

      count = phase.get_objection().get_objection_count(this);
      if (count == 2)
        `uvm_info("TEST", "double raise count correct: PASS", UVM_LOW)
      else
        `uvm_error("TEST", $sformatf("double raise: FAIL (count=%0d, expected 2)", count))

      phase.drop_objection(this);
      phase.drop_objection(this);
    endtask
  endclass

  initial run_test("neg_phase_double_raise_test");
endmodule
