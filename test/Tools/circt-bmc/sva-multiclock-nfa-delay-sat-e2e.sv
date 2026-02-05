// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --allow-multi-clock --module=sva_multiclock_nfa_delay_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_multiclock_nfa_delay_sat(input logic clk_a, input logic clk_b);
  logic req;
  logic ack;
  assign req = 1'b1;
  assign ack = 1'b0;
  assert property (@(posedge clk_b) req |-> ##1 ack);
endmodule

// CHECK: BMC_RESULT=SAT
