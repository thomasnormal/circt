// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 1 --module=sva_xprop_nested_aggregate_inject_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_xprop_nested_aggregate_inject_sat(input logic clk,
                                             input logic [1:0] in);
  typedef struct packed { logic [1:0] a; logic b; } pair_t;
  pair_t arr [0:0];
  always_comb begin
    arr[0] = '{a: 2'b00, b: 1'b0};
    arr[0].a = in;
  end
  // Nested aggregate writes should preserve X.
  assert property (@(posedge clk) (arr[0].a == 2'b00));
endmodule

// CHECK: BMC_RESULT=SAT
