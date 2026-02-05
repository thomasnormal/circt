// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_dyn_partselect_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_dyn_partselect_sat(input logic clk,
                                    input logic [3:0] in,
                                    input logic [1:0] idx_in);
  logic [1:0] slice;
  assign slice = in[idx_in +: 2];
  // Dynamic part-select should be X if idx_in is unknown.
  assert property (@(posedge clk) (slice == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT
