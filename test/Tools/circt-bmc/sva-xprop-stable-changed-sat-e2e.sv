// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_stable_sat - | FileCheck %s --check-prefix=STABLE
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_changed_sat - | FileCheck %s --check-prefix=CHANGED
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_stable_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // $stable should propagate X if either sample is unknown.
  assert property (@(posedge clk) ($stable(in) == 1'b1));
endmodule

module sva_xprop_changed_sat(input logic clk);
  logic in;
  assign in = 1'bx;
  // $changed should propagate X if either sample is unknown.
  assert property (@(posedge clk) ($changed(in) == 1'b0));
endmodule

// STABLE: BMC_RESULT=SAT
// CHANGED: BMC_RESULT=SAT
