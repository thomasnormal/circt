// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// Test $past with explicit clocking argument in position 3 (gating expr position).

module PastClocking(input logic clk, input logic a, input logic b,
                    input logic enable);
  // CHECK-LABEL: moore.module @PastClocking

  property past_with_clocking;
    @(posedge clk) a |-> $past(b, 2, @(posedge clk));
  endproperty
  // CHECK-COUNT-5: moore.variable
  // CHECK: moore.procedure
  // CHECK: moore.blocking_assign
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_with_clocking);

  property past_with_clocking_enable;
    @(posedge clk) a |-> $past(b, 1, enable, @(posedge clk));
  endproperty
  // CHECK: moore.conditional
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (past_with_clocking_enable);
endmodule
