// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 1 --module=sva_xprop_rose_sat - | FileCheck %s --check-prefix=ROSE
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 1 --module=sva_xprop_fell_sat - | FileCheck %s --check-prefix=FELL
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_rose_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // With unknown current samples, $rose does not produce a true edge.
  assert property (@(posedge clk) ($rose(in) == 1'b0));
endmodule

module sva_xprop_fell_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // With unknown current samples, $fell does not produce a true edge.
  assert property (@(posedge clk) ($fell(in) == 1'b0));
endmodule

// ROSE: BMC_RESULT=UNSAT
// FELL: BMC_RESULT=UNSAT
