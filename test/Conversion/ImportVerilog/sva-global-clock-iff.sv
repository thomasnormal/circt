// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SvaGlobalClockIff(input logic clk, en, a, b);
  global clocking @(posedge clk); endclocking

  // CHECK-LABEL: moore.module @SvaGlobalClockIff
  // Outer iff on $global_clock must gate property clocking.
  // CHECK: ltl.and {{.*}}, {{.*}} : i1, !ltl.property
  // CHECK: ltl.clock {{.*}} posedge
  assert property (@($global_clock iff en) (a |-> b));

  // Explicit sampled-value clocking with $global_clock iff must propagate iff
  // into the wait-event detect condition.
  // CHECK: moore.detect_event posedge {{.*}} if {{.*}} : l1
  // CHECK: verif.assert
  assert property ($rose(a, @($global_clock iff en)));
endmodule
