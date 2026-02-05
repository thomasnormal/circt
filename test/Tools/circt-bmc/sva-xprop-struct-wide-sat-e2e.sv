// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_struct_wide_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_struct_wide_sat(input logic clk, input logic [1:0] in);
  typedef struct packed { logic [1:0] a; logic b; } pair_t;
  pair_t p;
  assign p = '{a: in, b: 1'b0};
  // Multi-bit struct field extraction should preserve X.
  assert property (@(posedge clk) (p.a == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT
