// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_lec_multiclock_test extends uvm_test;
  `uvm_component_utils(uvm_lec_multiclock_test)

  function new(string name = "uvm_lec_multiclock_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

module modA(
    input logic clk0,
    input logic clk1,
    input logic a,
    input logic b);
  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  property p1;
    @(posedge clk1) b |-> a;
  endproperty

  assert property (p0);
  assert property (p1);
endmodule

module modB(
    input logic clk0,
    input logic clk1,
    input logic a,
    input logic b);
  property p0;
    @(posedge clk0) a |-> b;
  endproperty
  property p1;
    @(posedge clk1) b |-> a;
  endproperty

  assert property (p0);
  assert property (p1);
endmodule

// CHECK: unsat
