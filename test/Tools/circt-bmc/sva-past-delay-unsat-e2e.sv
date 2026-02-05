// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_past_delay_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_past_delay_unsat(input logic clk);
  logic req;
  logic ack;
  initial begin
    req = 1'b1;
    ack = 1'b1;
  end
  assert property (@(posedge clk) $past(req) |-> ##1 ack);
endmodule

// CHECK: BMC_RESULT=UNSAT
