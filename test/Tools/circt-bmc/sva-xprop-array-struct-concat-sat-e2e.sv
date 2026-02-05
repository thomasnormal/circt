// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_array_struct_concat_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_array_struct_concat_sat(input logic clk,
                                         input logic [1:0] in);
  typedef struct packed { logic [1:0] a; logic b; } pair_t;
  pair_t arr [0:0];
  logic [2:0] out;
  assign arr[0] = '{a: in, b: 1'b0};
  assign out = {arr[0].a, arr[0].b};
  // Concat of array-of-struct fields should preserve X.
  assert property (@(posedge clk) (out == 3'b000));
endmodule

// CHECK: BMC_RESULT=SAT
