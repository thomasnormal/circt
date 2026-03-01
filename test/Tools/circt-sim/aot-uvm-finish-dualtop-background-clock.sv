// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-compile %t.mlir -o %t.so
// RUN: circt-sim %t.mlir --compiled=%t.so --parallel=1 --mlir-disable-threading --max-time=200000000 2>&1 | FileCheck %s

// Regression: in AOT mode with dual tops and a free-running HDL clock,
// run_test must still progress through report/final phases and not strand
// wait_for_state on stale phase wrappers.
// CHECK: DROP_DONE
// CHECK: REPORT_DONE
// CHECK-NOT: Main loop exit: maxTime reached
// CHECK-NOT: advanceTime() returned false

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class finish_test extends uvm_test;
  `uvm_component_utils(finish_test)

  function new(string name = "finish_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    #20;
    phase.drop_objection(this);
    $display("DROP_DONE");
  endtask

  function void report_phase(uvm_phase phase);
    $display("REPORT_DONE");
  endfunction
endclass

module hvl_top;
  initial run_test("finish_test");
endmodule

module hdl_top;
  bit clk;
  always #1 clk = ~clk;
endmodule
