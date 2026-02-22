// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemDebugSubroutine(input logic clk, a, b);
  sequence s_showvars;
    (1, $showvars(a, b)) ##1 a;
  endsequence

  // Debug-oriented system subroutines in match-items should preserve effects.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemDebugSubroutine
  // CHECK: moore.fmt.literal " a = "
  // CHECK: moore.fmt.literal " b = "
  // CHECK: moore.builtin.display
  // CHECK: verif.assert
  assert property (@(posedge clk) s_showvars);
endmodule
