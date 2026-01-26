// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// The clock input is a four-state value represented as a struct with 'value' and 'unknown' fields.
// The effective clock is derived as: and(value, xor(unknown, true)) -- i.e., value when unknown=0.

// CHECK-LABEL: hw.module @ClockPosEdgeAlwaysFF(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
module ClockPosEdgeAlwaysFF(input logic clock, input int d, output int q);
  // CHECK: [[VALUE:%.+]] = hw.struct_extract %clock["value"]
  // CHECK: [[UNKNOWN:%.+]] = hw.struct_extract %clock["unknown"]
  // CHECK: [[NOT_UNKNOWN:%.+]] = comb.xor [[UNKNOWN]], %true
  // CHECK: [[EFFECTIVE:%.+]] = comb.and bin [[VALUE]], [[NOT_UNKNOWN]]
  // CHECK: [[CLK:%.+]] = seq.to_clock [[EFFECTIVE]]
  // CHECK: %q = seq.firreg {{%.+}} clock [[CLK]]{{.*}} : i32
  // CHECK: hw.output %q
  always_ff @(posedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ClockPosEdge(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
module ClockPosEdge(input logic clock, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: %q = seq.firreg {{%.+}} clock [[CLK]]{{.*}} : i32
  // CHECK: hw.output %q
  always @(posedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ClockNegEdge(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
module ClockNegEdge(input logic clock, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: [[CLK_INV:%.+]] = seq.clock_inv [[CLK]]
  // CHECK: %q = seq.firreg {{%.+}} clock [[CLK_INV]]{{.*}} : i32
  // CHECK: hw.output %q
  always @(negedge clock) q <= d;
endmodule

// CHECK-LABEL: hw.module @ActiveHighReset(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %reset : !hw.struct<value: i1, unknown: i1>
module ActiveHighReset(input logic clock, input logic reset, input int d1, input int d2, output int q1, output int q2);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: %q1 = seq.firreg {{%.+}} clock [[CLK]] reset async {{%.+}}, %c42_i32{{.*}} : i32
  // CHECK: hw.output %q1
  always @(posedge clock, posedge reset) if (reset) q1 <= 42; else q1 <= d1;
  always @(posedge clock, posedge reset) q2 <= reset ? 42 : d2;
endmodule

// CHECK-LABEL: hw.module @ActiveLowReset(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %reset : !hw.struct<value: i1, unknown: i1>
module ActiveLowReset(input logic clock, input logic reset, input int d1, input int d2, output int q1, output int q2);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: %q1 = seq.firreg {{%.+}} clock [[CLK]] reset async {{%.+}}, %c42_i32{{.*}} : i32
  // CHECK: hw.output %q1
  always @(posedge clock, negedge reset) if (!reset) q1 <= 42; else q1 <= d1;
  always @(posedge clock, negedge reset) q2 <= !reset ? 42 : d2;
endmodule

// CHECK-LABEL: hw.module @Enable(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %enable : !hw.struct<value: i1, unknown: i1>
module Enable(input logic clock, input logic enable, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: %q = seq.firreg {{%.+}} clock [[CLK]]{{.*}} : i32
  // CHECK: hw.output %q
  always @(posedge clock) if (enable) q <= d;
endmodule

// CHECK-LABEL: hw.module @ResetAndEnable(
// CHECK-SAME: in %clock : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %reset : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %enable : !hw.struct<value: i1, unknown: i1>
module ResetAndEnable(input logic clock, input logic reset, input logic enable, input int d, output int q);
  // CHECK: [[CLK:%.+]] = seq.to_clock
  // CHECK: %q = seq.firreg {{%.+}} clock [[CLK]] reset async {{%.+}}, %c42_i32{{.*}} : i32
  // CHECK: hw.output %q
  always @(posedge clock, posedge reset) if (reset) q <= 42; else if (enable) q <= d;
endmodule
