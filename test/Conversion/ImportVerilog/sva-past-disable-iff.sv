// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_past_disable_iff(
    input logic clk,
    input logic reset,
    input logic a);
  property p;
    @(posedge clk) disable iff (reset) $past(a);
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_past_disable_iff
// CHECK: moore.procedure always
// CHECK: moore.conditional
// CHECK: moore.blocking_assign
// CHECK: verif.clocked_assert
