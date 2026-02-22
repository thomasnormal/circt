// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSampledDynamicArray(input logic clk);
  int s[];

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.locator.yield
  // CHECK-DAG: moore.past
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) $stable(s));

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.not
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) $changed(s));

  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.past
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) $rose(s));

  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.ne
  // CHECK-DAG: moore.past
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) $fell(s));
endmodule
