// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASampledStringStableChanged(input logic clk, clk2);
  string s;

  // Direct assertion-clock lowering should preserve string semantics for
  // sampled stability and change detection.
  // CHECK-LABEL: moore.module @SVASampledStringStableChanged
  // CHECK: moore.string_cmp eq
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $stable(s));

  // CHECK: moore.string_cmp eq
  // CHECK: moore.not
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $changed(s));

  // Explicit sampled clocking (different from assertion clock) forces helper
  // lowering; it should also preserve native string compare.
  // CHECK: moore.procedure always
  // CHECK: moore.string_cmp eq
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $stable(s, @(posedge clk2)));

  // CHECK: moore.procedure always
  // CHECK: moore.string_cmp eq
  // CHECK: moore.not
  // CHECK: verif.clocked_assert
  assert property (@(posedge clk) $changed(s, @(posedge clk2)));

  // CHECK-NOT: moore.string_to_int
endmodule
