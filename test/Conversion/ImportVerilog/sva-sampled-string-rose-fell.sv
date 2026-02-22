// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASampledStringRoseFell(input logic clk, clk2);
  string s;

  // Direct assertion-clock sampled edges should use native string bool-cast.
  // CHECK-LABEL: moore.module @SVASampledStringRoseFell
  // CHECK: moore.bool_cast {{.*}} : string -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(s));

  // CHECK: moore.bool_cast {{.*}} : string -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $fell(s));

  // Explicit sampled clocking helper path should also use string bool-cast.
  // CHECK: moore.procedure always
  // CHECK: moore.bool_cast {{.*}} : string -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(s, @(posedge clk2)));

  // CHECK: moore.procedure always
  // CHECK: moore.bool_cast {{.*}} : string -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $fell(s, @(posedge clk2)));

  // CHECK-NOT: moore.string_to_int
endmodule
