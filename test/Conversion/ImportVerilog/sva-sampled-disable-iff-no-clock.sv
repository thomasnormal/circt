// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaSampledDisableIffNoClock(input logic rst, a, b);
  // In no-clock assertion contexts, sampled-value helper state should still
  // honor disable iff controls.
  assert property (disable iff (rst) ($rose(a) |-> b));
endmodule

// CHECK-LABEL: moore.module @SvaSampledDisableIffNoClock
// CHECK: %[[RST_VAR:.*]] = moore.variable name "rst" : <l1>
// CHECK: moore.procedure always
// CHECK: %[[RST_READ:.*]] = moore.read %[[RST_VAR]] : <l1>
// CHECK: moore.conditional %[[RST_READ]] : l1 ->
// CHECK: verif.assert {{.*}} if {{.*}} : !ltl.property
