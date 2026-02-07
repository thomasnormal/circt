// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test nested disable iff handling.
// IEEE 1800-2017 §16.12: disable iff is only valid at property_spec level,
// so nested disable iff uses 'default disable iff' for the inner reset.
module test_disable_iff_nested(
    input logic clk,
    input logic rst_outer,
    input logic rst_inner,
    input logic a,
    input logic b);
  default disable iff (rst_inner);

  property nested_disable;
    @(posedge clk) disable iff (rst_outer) (a |-> b);
  endproperty

  assert property (nested_disable);
endmodule

// CHECK-LABEL: moore.module @test_disable_iff_nested
// CHECK-COUNT-2: ltl.or {{.*}} {sva.disable_iff}
