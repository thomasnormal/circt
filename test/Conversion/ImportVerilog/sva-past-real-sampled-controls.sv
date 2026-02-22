// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVAPastRealSampledControls(input logic clk, en, a);
  real r;

  // Explicit clocking + enable control lowers through helper procedure and
  // keeps real-typed sampled history.
  // CHECK-LABEL: moore.module @SVAPastRealSampledControls
  // CHECK: moore.variable : <f64>
  // CHECK: moore.variable : <f64>
  // CHECK: moore.wait_event
  // CHECK: moore.read %{{.*}} : <f64>
  // CHECK: moore.blocking_assign %{{.*}}, %{{.*}} : f64
  // CHECK: moore.feq
  assert property (@(posedge clk) a |-> $past(r, 1, en, @(posedge clk)) == r);
endmodule
