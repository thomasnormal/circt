// RUN: circt-verilog %s --ir-hw 2>/dev/null | circt-sim - --top top --max-time 200000000 2>&1 | FileCheck %s

// Regression: run_test cleanup must still terminate the simulation when there
// is an unrelated free-running HDL clock process. If terminate is absorbed and
// never re-fired, the run exits by max-time despite cleanup/report completing.
// CHECK-NOT: Main loop exit: maxTime reached
// CHECK: DROP_DONE
// CHECK: REPORT_DONE
// CHECK-NOT: Main loop exit: maxTime reached

`timescale 1ns/1ps
import uvm_pkg::*;
`include "uvm_macros.svh"

class forever_comp_bg extends uvm_component;
  `uvm_component_utils(forever_comp_bg)

  function new(string name, uvm_component parent);
    super.new(name, parent);
  endfunction

  task run_phase(uvm_phase phase);
    forever begin
      #1;
    end
  endtask
endclass

class my_bg_test extends uvm_test;
  `uvm_component_utils(my_bg_test)
  forever_comp_bg c;

  function new(string name = "my_bg_test", uvm_component parent = null);
    super.new(name, parent);
  endfunction

  function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    c = forever_comp_bg::type_id::create("c", this);
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
  bit clk;
  always #1 clk = ~clk;

  initial begin
    run_test("my_bg_test");
  end
endmodule
