// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaClockEventList(input logic clk, a, b);
  // Event-list property clocking should lower by clocking the property on each
  // listed event and OR'ing the resulting clocked properties.
  // CHECK-LABEL: moore.module @SvaClockEventList
  // CHECK-COUNT-2: ltl.clock
  // CHECK: ltl.or
  // CHECK: verif.assert
  assert property (@(posedge clk or negedge clk) (a |-> b));
endmodule
