// RUN: circt-verilog %s --parse-only | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @PastEnableImplicit
// CHECK: moore.procedure always
// CHECK: moore.conditional
// CHECK: verif.clocked_assert
module PastEnableImplicit(input logic clk, input logic a, input logic b,
                          input logic enable);
  always @(posedge clk) begin
    assert property (a |-> $past(b, 1, enable));
  end
endmodule
