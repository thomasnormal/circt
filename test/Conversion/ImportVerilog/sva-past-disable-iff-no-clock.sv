// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module SvaPastDisableIffNoClock(input logic rst, en, a);
  // Without an explicit or inferred assertion clock, disable iff should still
  // participate in sampled-value helper state updates for $past.
  assert property (disable iff (rst) ($past(a, 1, en) |-> a));
endmodule

// CHECK-LABEL: moore.module @SvaPastDisableIffNoClock
// CHECK: %[[RST_VAR:.*]] = moore.variable name "rst" : <l1>
// CHECK: moore.procedure always
// CHECK: %[[RST_READ:.*]] = moore.read %[[RST_VAR]] : <l1>
// CHECK: moore.conditional %[[RST_READ]] : l1 -> l1
// CHECK: verif.assert {{.*}} if {{.*}} : !ltl.property
