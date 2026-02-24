// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 2 --module=sva_xprop_stable_sat - | FileCheck %s --check-prefix=STABLE
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 2 --module=sva_xprop_changed_sat - | FileCheck %s --check-prefix=CHANGED
// REQUIRES: slang
// REQUIRES: z3

module sva_xprop_stable_sat(input logic clk);
  logic in = 1'b0;
  always @(posedge clk)
    in <= 1'bx;
  // If either sampled value is unknown, equality-to-1 can fail.
  assert property (@(posedge clk) ($stable(in) == 1'b1));
endmodule

module sva_xprop_changed_sat(input logic clk);
  logic in = 1'b0;
  always @(posedge clk)
    in <= 1'bx;
  // If either sampled value is unknown, equality-to-0 can fail.
  assert property (@(posedge clk) ($changed(in) == 1'b0));
endmodule

// STABLE: BMC_RESULT=SAT
// CHANGED: BMC_RESULT=SAT
