// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc -b 6 --ignore-asserts-until=0 --module top --assume-known-inputs --rising-clocks-only - | \
// RUN:   FileCheck %s
// REQUIRES: slang
// REQUIRES: bmc-jit
// REQUIRES: z3

// First sampled-value observation uses an implicit unknown prior sample.
// With known-inputs enabled this should still hold for 4-state locals.
module top(input logic clk);
  logic [7:0] counter = 0;
  logic a = 0;
  logic b = 1;
  logic c;
  logic [2:0] wide_b = 'x;

  always @(posedge clk) begin
    if (counter == 0) begin
      assert property ($fell(a));
      assert property ($rose(b));
      assert property ($stable(c));
      assert property ($stable(wide_b));
      counter <= 1;
    end
  end
endmodule

// CHECK: BMC_RESULT=UNSAT
