// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #27: queue case-equality must lower and execute.

module tb;
  int q[$] = '{5, 3, 1, 4, 2};
  int expected[$];

  initial begin
    q.sort();
    expected = '{1, 2, 3, 4, 5};
    if (q !== expected)
      $display("FAIL");
    else
      $display("PASS");
    $finish;
  end

  // CHECK: PASS
endmodule
