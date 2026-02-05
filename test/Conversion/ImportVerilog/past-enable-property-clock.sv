// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module PastEnablePropertyClock(input logic clk, input logic a, input logic b,
                               input logic enable);
  // CHECK-LABEL: moore.module @PastEnablePropertyClock
  property p;
    @(posedge clk) a |-> $past(b, 1, enable);
  endproperty
  // CHECK: moore.conditional
  // CHECK: verif.{{(clocked_)?}}assert
  assert property (p);
endmodule
