// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | FileCheck %s
// REQUIRES: slang

module top(input logic clk, input logic a, input logic b);
  always @(posedge clk) begin
    if (a);
    else assume property (@(posedge clk) b);
  end

  assert property (@(posedge clk) b);
endmodule

// CHECK: verif.clocked_assume
// CHECK-SAME: if
// CHECK-SAME: posedge
// CHECK: verif.clocked_assert
// CHECK-SAME: posedge
