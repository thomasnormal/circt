// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaSequenceEventListMultibitSignal(input logic [1:0] e, input logic a);
  sequence s;
    a;
  endsequence

  // Mixed sequence + signal event lists should accept multi-bit signal events.
  // CHECK-LABEL: moore.module @SvaSequenceEventListMultibitSignal
  // CHECK: moore.bool_cast
  // CHECK: ltl.matched
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (@(s or e) a);
endmodule
