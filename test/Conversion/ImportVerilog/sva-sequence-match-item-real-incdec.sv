// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVASequenceMatchItemRealIncDec(input logic clk, a);
  real r;
  sequence s_inc;
    real x;
    (a, x = r) ##1 (a, x++);
  endsequence

  sequence s_dec;
    real y;
    (a, y = r) ##1 (a, y--);
  endsequence

  // CHECK-LABEL: moore.module @SVASequenceMatchItemRealIncDec
  // CHECK: moore.constant_real 1.000000e+00 : f64
  // CHECK: moore.fadd
  assert property (@(posedge clk) s_inc);

  // CHECK: moore.constant_real 1.000000e+00 : f64
  // CHECK: moore.fsub
  assert property (@(posedge clk) s_dec);
endmodule
