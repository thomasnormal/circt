// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledQueue(input logic clk);
  int q[$];

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.locator.yield
  // CHECK-DAG: moore.past
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk) $stable(q));

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.not
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk) $changed(q));

  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.past
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk) $rose(q));

  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.past
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.clocked_assert
  assert property (@(posedge clk) $fell(q));
endmodule
