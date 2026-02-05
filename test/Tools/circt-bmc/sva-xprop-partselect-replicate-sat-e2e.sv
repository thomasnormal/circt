// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_partselect_replicate_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_partselect_replicate_sat(input logic clk,
                                          input logic [3:0] in);
  logic [1:0] slice;
  logic [1:0] rep;
  assign slice = in[1:0];
  assign rep = {2{in[0]}};
  // Part-select and replication should preserve X.
  assert property (@(posedge clk) (slice == 2'b00));
  assert property (@(posedge clk) (rep == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT
