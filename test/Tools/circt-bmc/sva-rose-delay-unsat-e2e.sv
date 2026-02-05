// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 2 --module=sva_rose_delay_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_rose_delay_unsat(input logic clk);
  logic req;
  logic ack;
  assign req = 1'b1;
  assign ack = 1'b1;
  assert property (@(posedge clk) $rose(req) |-> ##1 ack);
endmodule

// CHECK: BMC_RESULT=UNSAT
