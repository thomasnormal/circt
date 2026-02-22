// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemFatalSubroutine(input logic clk, a);
  sequence s;
    int x;
    (1, x = 0, $fatal(1, "boom")) ##1 (1, x += 1) ##1 (x == 1);
  endsequence

  // Match-item $fatal should preserve fatal side effects rather than being
  // dropped.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemFatalSubroutine
  // CHECK: moore.builtin.severity fatal
  // CHECK: moore.builtin.finish 1
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
