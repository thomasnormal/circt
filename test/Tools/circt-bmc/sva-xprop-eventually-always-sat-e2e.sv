// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_xprop_eventually_sat - | FileCheck %s --check-prefix=EV
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_xprop_always_sat - | FileCheck %s --check-prefix=AL
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_eventually_sat(input logic clk, input logic in);
  // Eventually with X operand can be X, so property can fail.
  assert property (@(posedge clk) s_eventually in);
endmodule

module sva_xprop_always_sat(input logic clk, input logic in);
  // Always with X operand can be X, so property can fail.
  assert property (@(posedge clk) always in);
endmodule

// EV: BMC_RESULT=SAT
// AL: BMC_RESULT=SAT
