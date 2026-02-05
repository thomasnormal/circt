// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_xprop_seq_and_sat - | FileCheck %s --check-prefix=AND
// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_xprop_seq_or_sat - | FileCheck %s --check-prefix=OR
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_seq_and_sat(input logic clk, input logic in);
  // Sequence AND with X operand can be X.
  assert property (@(posedge clk) (in and 1'b1));
endmodule

module sva_xprop_seq_or_sat(input logic clk, input logic in);
  // Sequence OR with X operand can be X.
  assert property (@(posedge clk) (in or 1'b0));
endmodule

// AND: BMC_RESULT=SAT
// OR: BMC_RESULT=SAT
