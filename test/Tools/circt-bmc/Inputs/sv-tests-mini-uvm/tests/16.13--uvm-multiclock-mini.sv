/*
:name: uvm_multiclock_mini
:description: minimal UVM multiclock assertions for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_multiclock_smoke extends uvm_test;
  `uvm_component_utils(uvm_multiclock_smoke)

  function new(string name = "uvm_multiclock_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top(
    input logic clk_a,
    input logic clk_b,
    input logic a,
    input logic b);
  property p_a;
    @(posedge clk_a) a |-> ##1 a;
  endproperty

  property p_b;
    @(posedge clk_b) b |-> ##1 b;
  endproperty

  assert property (p_a);
  assert property (p_b);
endmodule
