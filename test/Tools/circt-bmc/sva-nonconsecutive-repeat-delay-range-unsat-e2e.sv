// RUN: circt-verilog --ir-hw %s | \
// RUN:   circt-bmc -b 4 --module=sva_nonconsecutive_repeat_delay_range_unsat - | FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

module sva_nonconsecutive_repeat_delay_range_unsat(input logic clk);
  logic req;
  logic a;
  logic b;
  initial req = 1'b1;
  always @(posedge clk) begin
    req <= 1'b0;
  end
  assign a = 1'b1;
  assign b = 1'b1;
  assert property (@(posedge clk) req |-> a [=1:3] ##[1:2] b);
endmodule

// CHECK: BMC_RESULT=UNSAT
