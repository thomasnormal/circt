// REQUIRES: slang
// REQUIRES: z3
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-opt --lower-clocked-assert-like --lower-ltl-to-core | \
// RUN:   circt-lec --emit-smtlib -c1=modA -c2=modB - | %z3 -in | FileCheck %s

module modA(input logic clk, input logic a, input logic b);
  property p;
    @(posedge clk) a ##1 b;
  endproperty
  expect property (p);
endmodule

module modB(input logic clk, input logic a, input logic b);
  property p;
    @(posedge clk) a ##1 b;
  endproperty
  expect property (p);
endmodule

// CHECK: unsat
