// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceMatchedMethod(input logic clk, a, b, c);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Sequence `.matched` should lower to `ltl.matched`.
  // CHECK-LABEL: moore.module @SVASequenceMatchedMethod
  // CHECK: ltl.matched
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) s.matched);
endmodule
