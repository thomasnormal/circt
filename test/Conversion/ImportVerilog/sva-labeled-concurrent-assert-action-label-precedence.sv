// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s | FileCheck %s
// REQUIRES: slang

module top(input logic clk, input logic a);
  // When both a statement label and action-block message label are present,
  // keep action-block-derived label precedence.
  // CHECK: verif.clocked_assert {{.*}} label "action_label"
  // CHECK-NOT: verif.clocked_assert {{.*}} label "stmt_label"
  stmt_label: assert property (@(posedge clk) a) else $error("action_label");
endmodule
