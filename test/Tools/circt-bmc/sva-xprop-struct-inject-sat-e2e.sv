// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_struct_inject_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_struct_inject_sat(input logic clk, input logic in);
  typedef struct packed { logic a; logic b; } pair_t;
  pair_t p;
  always_comb begin
    p = '{a: 1'b0, b: 1'b0};
    p.a = in;
  end
  // Struct field injection should preserve X.
  assert property (@(posedge clk) (p.a == 1'b0));
endmodule

// CHECK: BMC_RESULT=SAT
