/*
:name: uvm_assume_mini
:description: minimal UVM assume property for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_assume_smoke extends uvm_test;
  `uvm_component_utils(uvm_assume_smoke)

  function new(string name = "uvm_assume_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top(input logic clk, input logic a, input logic b);
  assume property (@(posedge clk) a |-> b);
endmodule
