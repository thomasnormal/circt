// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module expect_test(input logic clk, input logic a, input logic b);
  initial begin
    expect (@(posedge clk) a ##1 b);
  end

  // CHECK-LABEL: moore.module @expect_test
  // CHECK: verif.{{(clocked_)?}}assert
endmodule
