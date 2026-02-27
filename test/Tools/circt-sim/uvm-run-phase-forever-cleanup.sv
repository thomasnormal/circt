// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top top --max-time 200000000 2>&1 | FileCheck %s
// XFAIL: *
// REQUIRES-BUG: execute_phase cleanup currently hangs until max-time when
// run_phase contains forever-loop worker threads.

// Regression: run_test() must return even when a component run_phase has a
// forever loop. UVM should end run_phase after objections drop.
// CHECK: DROP_DONE
// CHECK: REPORT_DONE
// CHECK: AFTER_RUN_TEST
// CHECK-NOT: Main loop exit: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class forever_comp extends uvm_component;
  `uvm_component_utils(forever_comp)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      #1;
    end
  endtask
endclass

class my_test extends uvm_test;
  `uvm_component_utils(my_test)
  forever_comp c;

  function new(string name = "my_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    c = forever_comp::type_id::create("c", this);
  endfunction

  task run_phase(uvm_phase phase);
    phase.raise_objection(this);
    #10;
    phase.drop_objection(this);
    $display("DROP_DONE");
  endtask

  function void report_phase(uvm_phase phase);
    super.report_phase(phase);
    $display("REPORT_DONE");
  endfunction
endclass

module top;
  initial begin
    run_test("my_test");
    $display("AFTER_RUN_TEST");
  end
endmodule
