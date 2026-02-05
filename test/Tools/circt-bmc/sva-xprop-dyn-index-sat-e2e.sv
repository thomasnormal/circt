// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_dyn_index_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_dyn_index_sat(input logic clk, input logic idx_in);
  logic [0:0] arr [0:1];
  logic [0:0] out;
  logic [0:0] idx;
  assign idx = idx_in;
  always_comb begin
    arr[0] = 1'b0;
    arr[1] = 1'b1;
  end
  assign out = arr[idx];
  // Unknown index should make the read X, so equality can fail.
  assert property (@(posedge clk) (out == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT
