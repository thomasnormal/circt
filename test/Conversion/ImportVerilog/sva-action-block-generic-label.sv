// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaActionBlockGenericLabel(input logic clk, a, b);
  logic shadow;

  // Non-message action blocks should still preserve that an action block
  // existed, via a deterministic fallback label.
  // CHECK-LABEL: moore.module @SvaActionBlockGenericLabel
  // CHECK: verif.assert {{.*}} label "action_block"
  assert property (@(posedge clk) a |-> b) else begin
    shadow = a;
  end
endmodule
