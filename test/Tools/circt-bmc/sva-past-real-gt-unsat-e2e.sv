// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-bmc --run-smtlib --ignore-asserts-until=1 -b 3 --module top %t.mlir | FileCheck %s

module top(input logic clk);
  real r;
  real s;
  initial r = 1.0;
  initial s = 0.0;

  always_ff @(posedge clk)
    assert property (@(posedge clk) $past(r) > s);
endmodule

// CHECK: BMC_RESULT=UNSAT
