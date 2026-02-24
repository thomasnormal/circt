// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc --run-smtlib -b 3 --module=sva_unbounded_delay_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: z3

module sva_unbounded_delay_unsat(input logic clk);
  logic req;
  logic ack;
  assign req = 1'b1;
  assign ack = 1'b1;
  assert property (@(posedge clk) req |-> ##[1:$] ack);
endmodule

// CHECK: BMC_RESULT=UNSAT
