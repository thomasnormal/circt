// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVAPastStringSampledControls(input logic clk, en, a);
  string s;

  // $past with sampled-value controls should preserve string storage and avoid
  // lossy string<->int round-trips.
  // CHECK-LABEL: moore.module @SVAPastStringSampledControls
  // CHECK: moore.variable : <string>
  // CHECK: moore.variable : <string>
  // CHECK: moore.blocking_assign %{{.*}} : string
  // CHECK: moore.string_cmp eq
  // CHECK-NOT: moore.string_to_int
  // CHECK-NOT: moore.int_to_string
  assert property (@(posedge clk) a |-> $past(s, 1, en, @(posedge clk)) == s);
endmodule
