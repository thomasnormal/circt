// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=50000000000 2>&1 | FileCheck %s
//
// CHECK-DAG: RUN_PHASE_START
// CHECK-DAG: RUN_PHASE_END
// CHECK-DAG: POST_RUN_TEST
// CHECK-NOT: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class t extends uvm_test;
  `uvm_component_utils(t)

  function new(string name = "t", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  virtual task run_phase(uvm_phase phase);
    $display("RUN_PHASE_START t=%0t", $time);
    phase.raise_objection(this);
    #10;
    $display("RUN_PHASE_END t=%0t", $time);
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial begin
    // Default UVM behavior can call $finish inside run_test; disable that so
    // this regression specifically checks that run_test itself returns after
    // objections drop.
    uvm_top.finish_on_completion = 0;
    run_test("t");
    $display("POST_RUN_TEST t=%0t", $time);
  end
endmodule
