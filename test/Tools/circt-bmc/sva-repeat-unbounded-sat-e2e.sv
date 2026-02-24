// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 4 --module=sva_repeat_unbounded_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_repeat_unbounded_sat(input logic clk);
  logic a;
  assign a = 1'b0;
  assert property (@(posedge clk) a [*1:$]);
endmodule

// CHECK: BMC_RESULT=SAT
