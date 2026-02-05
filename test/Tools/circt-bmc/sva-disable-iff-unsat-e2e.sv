// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_disable_iff_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_disable_iff_unsat(input logic clk);
  logic reset;
  logic a;
  logic b;
  assign reset = 1'b1;
  assign a = 1'b1;
  assign b = 1'b0;
  assert property (@(posedge clk) disable iff (reset) a |-> b);
endmodule

// CHECK: BMC_RESULT=UNSAT
