// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time=50000000000 2>&1 | FileCheck %s
//
// CHECK: RUN_PHASE_START
// CHECK: RUN_PHASE_END
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
    $display("RUN_PHASE_START");
    phase.raise_objection(this);
    #10;
    $display("RUN_PHASE_END");
    phase.drop_objection(this);
  endtask
endclass

module top;
  initial run_test("t");
endmodule
