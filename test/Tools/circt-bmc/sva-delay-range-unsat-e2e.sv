// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_delay_range_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_delay_range_unsat(input logic clk);
  logic req;
  logic ack;
  assign req = 1'b1;
  assign ack = 1'b1;
  assert property (@(posedge clk) req |-> ##[1:3] ack);
endmodule

// CHECK: BMC_RESULT=UNSAT
