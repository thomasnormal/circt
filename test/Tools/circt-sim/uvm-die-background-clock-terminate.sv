// RUN: circt-verilog %s --ir-hw 2>/dev/null | circt-sim - --top top --max-time 50000000 2>&1 | FileCheck %s

// Regression: terminate triggered via uvm_root::die() must stop simulation
// even when an unrelated free-running HDL clock keeps scheduling events.
// CHECK-NOT: Main loop exit: maxTime reached
// CHECK: UVM_FATAL
// CHECK-NOT: Main loop exit: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class die_test extends uvm_test;
  `uvm_component_utils(die_test)

  function new(string name = "die_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    `uvm_fatal("DIE", "trigger die")
  endtask
endclass

module top;
  bit clk;
  always #1 clk = ~clk;

  initial run_test("die_test");
endmodule
