// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_goto_repeat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_goto_repeat_sat(input logic clk);
  logic a;
  assign a = 1'b0;
  assert property (@(posedge clk) a [-> 1]);
endmodule

// CHECK: BMC_RESULT=SAT
