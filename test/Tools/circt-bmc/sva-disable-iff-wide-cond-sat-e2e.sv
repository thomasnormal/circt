// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_disable_iff_wide_cond_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_disable_iff_wide_cond_sat(input logic clk);
  logic [3:0] reset;
  logic a;
  logic b;

  assign reset = 4'b0000;
  assign a = 1'b1;
  assign b = 1'b0;

  // A multi-bit disable condition is legal SystemVerilog and should be
  // interpreted using integral truthiness.
  assert property (@(posedge clk) disable iff (reset) a |-> b);
endmodule

// CHECK: BMC_RESULT=SAT
