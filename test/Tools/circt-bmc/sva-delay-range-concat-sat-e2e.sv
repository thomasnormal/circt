// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_delay_range_concat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_delay_range_concat_sat(input logic clk);
  logic a;
  logic b;
  logic c;
  assign a = 1'b1;
  assign b = 1'b0;
  assign c = 1'b0;
  assert property (@(posedge clk) a ##[1:2] b ##1 c);
endmodule

// CHECK: BMC_RESULT=SAT
