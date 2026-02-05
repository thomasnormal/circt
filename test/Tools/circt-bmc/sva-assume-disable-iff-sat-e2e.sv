// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_assume_disable_iff_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_assume_disable_iff_sat(input logic clk);
  logic reset;
  logic a;
  logic b;
  assign reset = 1'b1;
  assign a = 1'b1;
  assign b = 1'b0;
  assume property (@(posedge clk) disable iff (reset) a |-> b);
  assert property (@(posedge clk) a |-> b);
endmodule

// CHECK: BMC_RESULT=SAT
