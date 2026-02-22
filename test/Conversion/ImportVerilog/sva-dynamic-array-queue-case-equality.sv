// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaDynamicArrayQueueCaseEquality(input logic clk);
  int a[];
  int b[];
  int q0[$];
  int q1[$];

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.dyn_extract
  // CHECK-DAG: moore.case_eq
  // CHECK-DAG: moore.and
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) (a === b));

  // CHECK-DAG: moore.array.size
  // CHECK-DAG: moore.array.locator
  // CHECK-DAG: moore.not
  // CHECK-DAG: verif.assert
  assert property (@(posedge clk) (q0 !== q1));
endmodule
