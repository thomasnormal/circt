// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaConditionalPropertyMultibit(input logic clk, input logic [1:0] sel,
                                      input logic a, b);
  // Property-level conditional should accept integral truthy conditions.
  // CHECK-LABEL: moore.module @SvaConditionalPropertyMultibit
  // CHECK: moore.bool_cast
  // CHECK: ltl.not
  // CHECK: ltl.and
  // CHECK: ltl.or
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) if (sel) a else b);
endmodule
