// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --uvm-path=%S/../../../lib/Runtime/uvm --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

`include "uvm_macros.svh"
import uvm_pkg::*;

class uvm_lec_interface_test extends uvm_test;
  `uvm_component_utils(uvm_lec_interface_test)

  function new(string name = "uvm_lec_interface_test",
               uvm_component parent = null);
    super.new(name, parent);
  endfunction
endclass

interface ifc(input logic clk);
  logic a;
  logic b;
  property p;
    @(posedge clk) a |-> b;
  endproperty
endinterface

module modA(
    input logic clk,
    input logic a,
    input logic b);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;

  assert property (i.p);
endmodule

module modB(
    input logic clk,
    input logic a,
    input logic b);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;

  assert property (i.p);
endmodule

// CHECK: unsat
