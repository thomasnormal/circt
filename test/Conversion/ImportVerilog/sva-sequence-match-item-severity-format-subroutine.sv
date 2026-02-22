// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchItemSeverityFormatSubroutine(input logic clk, a);
  sequence s;
    int x;
    (1, x = 3, $info("i=%0d", x), $warning("w=%0d", x),
     $error("e=%0d", x), $fatal(2, "f=%0d", x)) ##1 (x == 3);
  endsequence

  // Formatted severity system subroutines in sequence match-items should
  // preserve format arguments in lowering.
  // CHECK-LABEL: moore.module @SVASequenceMatchItemSeverityFormatSubroutine
  // CHECK: moore.fmt.int
  // CHECK: moore.builtin.severity info
  // CHECK: moore.builtin.severity warning
  // CHECK: moore.builtin.severity error
  // CHECK: moore.builtin.severity fatal
  // CHECK: moore.builtin.finish 1
  // CHECK: verif.assert
  assert property (@(posedge clk) a |-> s);
endmodule
