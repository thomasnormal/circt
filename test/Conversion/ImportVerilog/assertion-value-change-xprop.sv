// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

module AssertionValueChangeStableChanged(input logic clk);
  logic in;
  assign in = 1'bx;
// CHECK-LABEL: moore.module @AssertionValueChangeStableChanged
// CHECK: moore.eq
// CHECK: moore.not
// CHECK-NOT: moore.case_eq
  assert property (@(posedge clk) $stable(in));
  assert property (@(posedge clk) $changed(in));
endmodule

module AssertionValueChangeRoseFell(input logic clk);
  logic in;
  assign in = 1'bx;
// CHECK-LABEL: moore.module @AssertionValueChangeRoseFell
// CHECK: moore.and
// CHECK: moore.not
// CHECK-NOT: moore.case_eq
  assert property (@(posedge clk) $rose(in));
  assert property (@(posedge clk) $fell(in));
endmodule
