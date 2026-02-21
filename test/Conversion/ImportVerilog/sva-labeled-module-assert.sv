// RUN: circt-verilog --ir-moore --no-uvm-auto-include %s
// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module sva_labeled_module_assert(
  input logic clk,
  input logic rst,
  input logic antecedent,
  output logic consequent
);
  always @(posedge clk)
    consequent <= rst ? 1'b0 : antecedent;

  // CHECK-LABEL: moore.module @sva_labeled_module_assert
  // CHECK: verif.assert {{.*}} label "Failed with consequent = "
  // CHECK-NOT: cf.br
  test_assert: assert property (@(posedge clk) disable iff (rst)
                                antecedent |=> consequent)
                 else $error("Failed with consequent = ", $sampled(consequent));
endmodule
