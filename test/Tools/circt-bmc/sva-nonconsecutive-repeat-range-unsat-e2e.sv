// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_nonconsecutive_repeat_range_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_nonconsecutive_repeat_range_unsat(input logic clk);
  logic a;
  assign a = 1'b1;
  assert property (@(posedge clk) a [=1:3]);
endmodule

// CHECK: BMC_RESULT=UNSAT
