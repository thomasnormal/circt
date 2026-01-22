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
  // CHECK: [[SAMP:%.+]] = moore.variable
  // CHECK: moore.blocking_assign [[SAMP]]
  // CHECK: moore.read [[SAMP]]
endmodule
