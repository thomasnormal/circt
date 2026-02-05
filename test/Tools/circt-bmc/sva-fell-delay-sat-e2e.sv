// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 3 --module=sva_fell_delay_sat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_fell_delay_sat(input logic clk);
  logic req;
  logic ack;
  initial begin
    req = 1'b1;
    ack = 1'b0;
  end
  always @(posedge clk) begin
    req <= 1'b0;
  end
  assert property (@(posedge clk) $fell(req) |-> ##1 ack);
endmodule

// CHECK: BMC_RESULT=SAT
