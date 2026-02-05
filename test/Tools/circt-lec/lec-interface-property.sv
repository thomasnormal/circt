// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | circt-opt \
// RUN:   --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

interface ifc(input logic clk);
  logic a;
  logic b;
  property p;
    @(posedge clk) a |-> b;
  endproperty
endinterface

module modA(input logic clk, input logic a, input logic b, output logic out);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;
  assign out = b;
  assert property (i.p);
endmodule

module modB(input logic clk, input logic a, input logic b, output logic out);
  ifc i(clk);
  assign i.a = a;
  assign i.b = b;
  assign out = b;
  assert property (i.p);
endmodule

// CHECK: unsat
