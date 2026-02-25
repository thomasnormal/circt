// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaActionBlockTaskFallbackLabel(input logic clk, a, b, input int x);
  // For dynamic-only display/warning/error payloads, preserve the originating
  // task kind as action label.
  // CHECK-LABEL: moore.module @SvaActionBlockTaskFallbackLabel
  // CHECK: verif.clocked_assert {{.*}} label "$display"
  assert property (@(posedge clk) a |-> b) else $display(x);
endmodule
