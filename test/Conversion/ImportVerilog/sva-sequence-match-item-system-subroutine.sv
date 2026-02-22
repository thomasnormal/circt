// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemSystemSubroutine(input logic clk, a);
  sequence s;
    int x;
    (1, x = 0, $display("x=%0d", x)) ##1 (1, x += 1) ##1 (x == 1);
  endsequence

  // System subroutine match-items should preserve side effects in lowering.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemSystemSubroutine
  // CHECK: moore.builtin.display
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
