// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-bmc --run-smtlib -b 3 --module top %t.mlir | FileCheck %s

// CHECK: BMC_RESULT=UNSAT

module top(input logic clk);
  real r;

  // r is never assigned, so it remains at 0.0. The sampled value should match.
  assert property (@(posedge clk) $past(r) == r);
endmodule
