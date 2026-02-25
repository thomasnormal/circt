// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledDynamicArrayExplicitClock(input logic clk_a, input logic clk_b);
  int s[];

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.locator.yield
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk_a) $stable(s, @(posedge clk_b)));

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk_a) $rose(s, @(posedge clk_b)));
endmodule
