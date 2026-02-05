// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_array_inject_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_array_inject_sat(input logic clk, input logic [1:0] in);
  logic [1:0] arr [0:0];
  logic [1:0] out;
  assign arr[0] = 2'b00;
  always_comb begin
    arr[0] = in;
  end
  assign out = arr[0];
  // Injecting unknowns into the array should preserve X.
  assert property (@(posedge clk) (out == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT
