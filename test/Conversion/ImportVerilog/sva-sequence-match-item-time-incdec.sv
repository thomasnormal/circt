// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

module SVASequenceMatchItemTimeIncDec(input logic clk, a);
  time t;
  sequence s_inc;
    time x;
    (a, x = t) ##1 (a, x++) ##1 (x > t);
  endsequence

  sequence s_dec;
    time y;
    (a, y = t) ##1 (a, y--) ##1 (y < t);
  endsequence

  // CHECK-LABEL: moore.module @SVASequenceMatchItemTimeIncDec
  // CHECK: moore.constant_real
  // CHECK: moore.fadd
  // CHECK: moore.ugt
  assert property (@(posedge clk) s_inc);

  // CHECK: moore.fsub
  // CHECK: moore.ult
  assert property (@(posedge clk) s_dec);
endmodule
