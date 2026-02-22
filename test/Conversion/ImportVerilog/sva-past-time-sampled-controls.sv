// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVAPastTimeSampledControls(input logic clk, en, a);
  time t;

  // $past with sampled-value controls should preserve native time storage in
  // helper history (no bitvector/time conversion round-trips in helper state).
  // CHECK-LABEL: moore.module @SVAPastTimeSampledControls
  // CHECK: moore.variable : <time>
  // CHECK: moore.variable : <time>
  // CHECK: moore.blocking_assign %{{.*}} : time
  assert property (@(posedge clk) a |-> $past(t, 1, en, @(posedge clk)) == t);
endmodule
