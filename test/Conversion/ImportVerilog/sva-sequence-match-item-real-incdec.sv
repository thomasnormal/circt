// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module SVASequenceMatchItemRealIncDec(input logic clk, a);
  real r;
  sequence s_inc;
    real x;
    (a, x = r) ##1 (a, x++) ##1 (x == r + 1.0);
  endsequence

  sequence s_dec;
    real y;
    (a, y = r) ##1 (a, y--) ##1 (y == r - 1.0);
  endsequence

  // CHECK-LABEL: moore.module @SVASequenceMatchItemRealIncDec
  // CHECK: moore.constant_real 1.000000e+00 : f64
  // CHECK: moore.fadd
  // CHECK: moore.feq
  assert property (@(posedge clk) s_inc);

  // CHECK: moore.fsub
  // CHECK: moore.feq
  assert property (@(posedge clk) s_dec);
endmodule
