// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module test_sampled_explicit_clock_default_disable(
    input logic clk, fast, reset, a);
  default disable iff (reset);

  property p;
    @(posedge clk) $rose(a, @(posedge fast));
  endproperty

  assert property (p);
endmodule

// CHECK-LABEL: moore.module @test_sampled_explicit_clock_default_disable
// CHECK-DAG: [[RESET:%[0-9]+]] = moore.read %reset
// CHECK-DAG: moore.wait_event
// CHECK: moore.conditional [[RESET]] : l1 -> l1
