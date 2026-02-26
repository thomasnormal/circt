// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module sva_concat_zero_delay_lowering(input logic clk, a, b, c);
  // CHECK-LABEL: moore.module @sva_concat_zero_delay_lowering
  // CHECK: [[CONV_A:%.+]] = moore.to_builtin_bool %a : l1
  // CHECK: [[DELAY_A:%.+]] = ltl.delay [[CONV_A]], 0, 0 : i1
  // CHECK: [[CONV_B:%.+]] = moore.to_builtin_bool %b : l1
  // CHECK: [[DELAY_B:%.+]] = ltl.delay [[CONV_B]], 0, 0 : i1
  // CHECK: [[OVERLAP:%.+]] = ltl.and [[DELAY_A]], [[DELAY_B]] : !ltl.sequence, !ltl.sequence
  // CHECK: verif.clocked_assert [[OVERLAP]], posedge [[CLK:%.+]] : !ltl.sequence
  assert property (@(posedge clk) a ##0 b);

  // CHECK: [[NONOVERLAP:%.+]] = ltl.concat [[DELAY_A]], [[DELAY_B]] : !ltl.sequence, !ltl.sequence
  // CHECK: verif.clocked_assert [[NONOVERLAP]], posedge [[CLK]] : !ltl.sequence
  assert property (@(posedge clk) a ##1 b);

  // CHECK: [[CONV_C:%.+]] = moore.to_builtin_bool %c : l1
  // CHECK: [[DELAY_C:%.+]] = ltl.delay [[CONV_C]], 0, 0 : i1
  // CHECK: [[OVERLAP_CHAIN:%.+]] = ltl.and [[OVERLAP]], [[DELAY_C]] : !ltl.sequence, !ltl.sequence
  // CHECK: verif.clocked_assert [[OVERLAP_CHAIN]], posedge [[CLK]] : !ltl.sequence
  assert property (@(posedge clk) a ##0 b ##0 c);

  // CHECK: [[MIXED:%.+]] = ltl.concat [[OVERLAP]], [[DELAY_C]] : !ltl.sequence, !ltl.sequence
  // CHECK: verif.clocked_assert [[MIXED]], posedge [[CLK]] : !ltl.sequence
  assert property (@(posedge clk) a ##0 b ##1 c);
endmodule
