// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVASequenceMatchItemTimeIncDec(input logic clk, a);
  time t;
  sequence s_inc;
    time x;
    (a, x = t) ##1 (a, x++);
  endsequence

  sequence s_dec;
    time y;
    (a, y = t) ##1 (a, y--);
  endsequence

  // CHECK-LABEL: moore.module @SVASequenceMatchItemTimeIncDec
  // CHECK: moore.constant_real
  // CHECK: moore.fadd
  assert property (@(posedge clk) s_inc);

  // CHECK: moore.constant_real
  // CHECK: moore.fsub
  assert property (@(posedge clk) s_dec);
endmodule
