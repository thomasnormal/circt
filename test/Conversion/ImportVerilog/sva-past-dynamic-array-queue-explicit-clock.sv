// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaPastDynamicArrayQueueExplicitClock(input logic clk_a, input logic clk_b);
  int d[];
  int q[$];

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.wait_event
  // CHECK-DAG: moore.detect_event posedge
  // CHECK-DAG: moore.variable
  // CHECK-DAG: moore.blocking_assign
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk_a) ($past(d, 2, @(posedge clk_b)) == d));

  // CHECK-DAG: moore.procedure always
  // CHECK-DAG: moore.wait_event
  // CHECK-DAG: moore.detect_event posedge
  // CHECK-DAG: moore.variable
  // CHECK-DAG: moore.blocking_assign
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk_a) ($past(q, 1, @(posedge clk_b)) == q));
endmodule
