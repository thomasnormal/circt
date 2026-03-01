// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=10000000000 2>&1 | FileCheck %s
//
// Regression: uvm_test_done.set_drain_time(...) must delay run-phase
// completion after the final drop_objection. Without this, run_test ends
// immediately at drop time and kills active run_phase tasks too early.
//
// CHECK: DROP_OBJECTION t={{ *}}100 ns
// CHECK: REPORT_PHASE t={{ *}}3100 ns
// CHECK: [circt-sim] Simulation completed at time 3100000000 fs

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class drain_test extends uvm_test;
  `uvm_component_utils(drain_test)

  function new(string name = "drain_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void end_of_elaboration_phase(uvm_phase phase);
    super.end_of_elaboration_phase(phase);
    // 3000ns global drain on test_done objection.
    uvm_test_done.set_drain_time(this, 3000ns);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    #100ns;
    $display("DROP_OBJECTION t=%0t", $time);
    phase.drop_objection(this);
  endtask

  function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    $display("REPORT_PHASE t=%0t", $time);
  endfunction
endclass

module top;
  initial run_test("drain_test");
endmodule
