// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_concat_delay_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_concat_delay_sat(input logic clk);
  logic a;
  logic b;
  assign a = 1'b1;
  assign b = 1'b0;
  assert property (@(posedge clk) a ##1 b);
endmodule

// CHECK: BMC_RESULT=SAT
