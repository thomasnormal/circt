// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 6 --ignore-asserts-until=1 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

// In known-input mode, written state should not spuriously start as 4-state X
// if that causes arithmetic transition properties to fail vacuously.
module top(input logic clk, input logic up);
  logic [7:0] cnt;

  always @(posedge clk)
    if (up)
      cnt <= cnt + 8'd1;

  assert property (@(posedge clk) up |=> cnt == ($past(cnt) + 8'd1));
endmodule

// CHECK: BMC_RESULT=UNSAT
