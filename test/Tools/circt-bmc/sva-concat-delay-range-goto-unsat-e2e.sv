// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_concat_delay_range_goto_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_concat_delay_range_goto_unsat(input logic clk);
  logic req;
  logic a;
  logic b;
  logic c;
  initial req = 1'b1;
  always @(posedge clk) begin
    req <= 1'b0;
  end
  assign a = 1'b1;
  assign b = 1'b1;
  assign c = 1'b1;
  assert property (@(posedge clk) req |-> (a ##[1:2] b) ##1 c [->1:2]);
endmodule

// CHECK: BMC_RESULT=UNSAT
