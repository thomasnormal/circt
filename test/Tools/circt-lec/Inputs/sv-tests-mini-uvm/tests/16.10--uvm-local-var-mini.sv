/*
:name: uvm_local_var_mini
:description: minimal UVM local-var property for harness coverage
:type: simulation elaboration
:tags: uvm uvm-assertions
:unsynthesizable: 1
*/

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_local_var_smoke extends uvm_test;
  `uvm_component_utils(uvm_local_var_smoke)

  function new(string name = "uvm_local_var_smoke",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module top(
    input logic clk,
    input logic valid,
    input logic [7:0] in,
    output logic [7:0] out);
  always_ff @(posedge clk) begin
    if (valid)
      out <= in + 8'd1;
  end

  property p;
    logic [7:0] x;
    @(posedge clk) (valid, x = in) |-> ##1 (out == x + 8'd1);
  endproperty

  assert property (p);
endmodule
