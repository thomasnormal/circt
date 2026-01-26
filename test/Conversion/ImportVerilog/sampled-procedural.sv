// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module SampledProc(input logic clk);
  logic val;

  always @(posedge clk) begin
    if (val != $sampled(val))
      $stop;

    val = ~val;

    if (val == $sampled(val))
      $stop;
  end

  // CHECK-LABEL: moore.module @SampledProc
  // CHECK: moore.procedure always
  // $sampled reads the value inline without creating a separate variable
  // CHECK: moore.read %val
  // CHECK: moore.read %val
  // CHECK: moore.ne
endmodule
