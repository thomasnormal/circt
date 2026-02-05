// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_rose_sat - | FileCheck %s --check-prefix=ROSE
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_fell_sat - | FileCheck %s --check-prefix=FELL
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_rose_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // $rose should propagate X if either sample is unknown.
  assert property (@(posedge clk) ($rose(in) == 1'b0));
endmodule

module sva_xprop_fell_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // $fell should propagate X if either sample is unknown.
  assert property (@(posedge clk) ($fell(in) == 1'b0));
endmodule

// ROSE: BMC_RESULT=SAT
// FELL: BMC_RESULT=SAT
