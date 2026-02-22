// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemSeveritySubroutine(input logic clk, a);
  sequence s;
    int x;
    (1, x = 0, $info("i"), $warning("w"), $error("e")) ##1
        (1, x += 1) ##1 (x == 1);
  endsequence

  // Severity system subroutines in sequence match-items should preserve
  // side effects in lowering instead of being ignored.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemSeveritySubroutine
  // CHECK: moore.builtin.severity info
  // CHECK: moore.builtin.severity warning
  // CHECK: moore.builtin.severity error
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
