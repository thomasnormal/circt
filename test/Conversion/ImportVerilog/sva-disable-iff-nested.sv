// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test nested disable iff handling.
module test_disable_iff_nested(
    input logic clk,
    input logic rst_outer,
    input logic rst_inner,
    input logic a,
    input logic b);
  property nested_disable;
    @(posedge clk) disable iff (rst_outer)
      (disable iff (rst_inner) a |-> b);
  endproperty

  assert property (nested_disable);
endmodule

// CHECK-LABEL: moore.module @test_disable_iff_nested
// CHECK-COUNT-2: ltl.or {{.*}} {sva.disable_iff}
