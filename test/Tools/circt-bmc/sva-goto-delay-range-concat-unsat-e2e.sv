// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_goto_delay_range_concat_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_goto_delay_range_concat_unsat(input logic clk);
  logic a;
  logic b;
  logic c;
  assign a = 1'b1;
  assign b = 1'b1;
  assign c = 1'b1;
  assert property (@(posedge clk) a [->1:3] ##[1:2] b ##1 c);
endmodule

// CHECK: BMC_RESULT=UNSAT
