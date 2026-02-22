// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledQueueExplicitClock(input logic clk_a, input logic clk_b);
  int q[$];

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.locator.yield
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk_a) $stable(q, @(posedge clk_b)));

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk_a) $rose(q, @(posedge clk_b)));
endmodule
