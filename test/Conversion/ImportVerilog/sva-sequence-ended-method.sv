// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASequenceEndedMethod(input logic clk, a, b);
  sequence s;
    @(posedge clk) a ##1 b;
  endsequence

  // Sequence `.ended` should lower to the sequence-endpoint bridge op.
  // CHECK-LABEL: moore.module @SVASequenceEndedMethod
  // CHECK: ltl.matched
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) s.ended);
endmodule
