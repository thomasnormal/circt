// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// REQUIRES: slang

module SvaPastEventSamplingSupported(input logic clk, input logic a);
  event ev;
  always_ff @(posedge clk) if (a) -> ev;

  property p_event_past;
    @(posedge clk) $past(ev, 1, 1'b1, @(posedge clk)) == ev;
  endproperty
  assert property (p_event_past);

  property p_event_past_with_disable;
    disable iff (!a) @(posedge clk) $past(ev, 2, 1'b1, @(posedge clk)) == ev;
  endproperty
  assert property (p_event_past_with_disable);
endmodule

// CHECK-LABEL: moore.module @SvaPastEventSamplingSupported
// CHECK: moore.bool_cast
// CHECK: moore.wait_event
// CHECK: moore.detect_event posedge
// CHECK: verif.{{(clocked_)?}}assert
