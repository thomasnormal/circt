// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_lec_smoke_test extends uvm_test;
  `uvm_component_utils(uvm_lec_smoke_test)

  function new(string name = "uvm_lec_smoke_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module modA(
    input logic clk,
    input logic valid,
    input logic [7:0] in,
    output logic [7:0] out);
  always_ff @(posedge clk) begin
    if (valid)
      out <= in + 8'd4;
  end

  property prop;
    logic [7:0] x;
    @(posedge clk) (valid, x = in) |-> ##1 (out == x + 8'd4);
  endproperty

  assert property (prop);
endmodule

module modB(
    input logic clk,
    input logic valid,
    input logic [7:0] in,
    output logic [7:0] out);
  always_ff @(posedge clk) begin
    if (valid)
      out <= in + 8'd4;
  end

  property prop;
    logic [7:0] x;
    @(posedge clk) (valid, x = in) |-> ##1 (out == x + 8'd4);
  endproperty

  assert property (prop);
endmodule

// CHECK: unsat
