// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module test_sampled_explicit(input logic clk, a);
  wire a_fell;
  assign a_fell = $fell(a, @(posedge clk));
endmodule

// CHECK-LABEL: moore.module @test_sampled_explicit
// CHECK: moore.procedure always
// CHECK: moore.wait_event
// CHECK: moore.blocking_assign
// CHECK: moore.assign
