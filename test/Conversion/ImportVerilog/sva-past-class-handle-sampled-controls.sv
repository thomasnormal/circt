// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaPastClassHandleSampledControls(input logic clk, input logic a);
  class C;
    int x;
  endclass

  C c;
  initial c = new;

  property p_class_past;
    @(posedge clk) $past(c, 1, 1'b1, @(posedge clk)) == c;
  endproperty
  assert property (p_class_past);

  property p_class_past_disable;
    disable iff (!a) @(posedge clk) $past(c, 2, 1'b1, @(posedge clk)) != c;
  endproperty
  cover property (p_class_past_disable);
endmodule

// CHECK-LABEL: moore.module @SvaPastClassHandleSampledControls
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: moore.class_handle_cmp
// CHECK: verif.{{(clocked_)?}}assert
// CHECK: verif.{{(clocked_)?}}cover
