// RUN: circt-translate --import-verilog %s | FileCheck %s

module SVASampledRealExplicitAndImplicitClock(input logic clk);
  real r;

  // Explicit clocking arguments use helper-procedure sampled state.
  // CHECK: moore.procedure always
  // CHECK: moore.feq
  assert property ($stable(r, @(posedge clk)));

  // CHECK: moore.procedure always
  // CHECK: moore.feq
  // CHECK: moore.not
  assert property ($changed(r, @(posedge clk)));

  // CHECK: moore.procedure always
  // CHECK: moore.constant_real 0.000000e+00 : f64
  // CHECK: moore.fne
  assert property ($rose(r, @(posedge clk)));

  // CHECK: moore.procedure always
  // CHECK: moore.constant_real 0.000000e+00 : f64
  // CHECK: moore.fne
  assert property ($fell(r, @(posedge clk)));

  // Implicit clocking inside clocked assertions uses direct sampled lowering.
  // CHECK: moore.past
  // CHECK: moore.feq
  // CHECK: ltl.clock
  assert property (@(posedge clk) $stable(r));

  // CHECK: moore.past
  // CHECK: moore.feq
  // CHECK: moore.not
  // CHECK: ltl.clock
  assert property (@(posedge clk) $changed(r));

  // CHECK: moore.constant_real 0.000000e+00 : f64
  // CHECK: moore.fne
  // CHECK: moore.past
  // CHECK: ltl.clock
  assert property (@(posedge clk) $rose(r));

  // CHECK: moore.constant_real 0.000000e+00 : f64
  // CHECK: moore.fne
  // CHECK: moore.past
  // CHECK: ltl.clock
  assert property (@(posedge clk) $fell(r));
endmodule
