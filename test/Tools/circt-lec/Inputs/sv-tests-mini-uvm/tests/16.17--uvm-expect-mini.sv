/*
:name: uvm_expect_mini
:description: minimal UVM expect statement for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_expect_smoke extends uvm_test;
  `uvm_component_utils(uvm_expect_smoke)

  function new(string name = "uvm_expect_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top(input logic clk, input logic a, input logic b);
  initial begin
    expect (@(posedge clk) a ##1 b);
  end
endmodule
