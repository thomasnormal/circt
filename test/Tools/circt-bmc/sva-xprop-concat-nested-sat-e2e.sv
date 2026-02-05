// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_concat_nested_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_concat_nested_sat(input logic clk,
                                   input logic [1:0] in);
  typedef struct packed { logic [1:0] a; logic b; } pair_t;
  pair_t p;
  logic [2:0] out;
  assign p = '{a: in, b: 1'b0};
  assign out = {p.a, p.b};
  // Concat of nested aggregate fields should preserve X.
  assert property (@(posedge clk) (out == 3'b000));
endmodule

// CHECK: BMC_RESULT=SAT
