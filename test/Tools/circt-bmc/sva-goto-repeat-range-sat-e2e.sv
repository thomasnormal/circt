// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_goto_repeat_range_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_goto_repeat_range_sat(input logic clk);
  logic a;
  assign a = 1'b0;
  assert property (@(posedge clk) a [->1:3]);
endmodule

// CHECK: BMC_RESULT=SAT
