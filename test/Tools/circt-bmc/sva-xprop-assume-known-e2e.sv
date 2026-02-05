// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_assume_known - | FileCheck %s --check-prefix=NOASSUME
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --assume-known-inputs --module=sva_xprop_assume_known - | FileCheck %s --check-prefix=ASSUME
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_assume_known(input logic clk, input logic in);
  // For known inputs, (in == 0) || (in == 1) is always true.
  // For X inputs, each compare is X, so the OR can be X and the assert can fail.
  assert property (@(posedge clk) ((in == 1'b0) || (in == 1'b1)));
endmodule

// NOASSUME: BMC_RESULT=SAT
// ASSUME: BMC_RESULT=UNSAT
