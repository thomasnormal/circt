// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s
// REQUIRES: slang

module SVASampledEvent(input logic clk, clk2);
  event e;

  // Event operands should be supported by sampled-value functions via native
  // event booleanization.
  // CHECK-LABEL: moore.module @SVASampledEvent
  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $stable(e));

  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: moore.not
  // CHECK: verif.assert
  assert property (@(posedge clk) $changed(e));

  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(e));

  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $fell(e));

  // Explicit sampled clock path should also support event operands.
  // CHECK: moore.procedure always
  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $stable(e, @(posedge clk2)));

  // CHECK: moore.procedure always
  // CHECK: moore.bool_cast {{.*}} : event -> i1
  // CHECK: verif.assert
  assert property (@(posedge clk) $rose(e, @(posedge clk2)));
endmodule
