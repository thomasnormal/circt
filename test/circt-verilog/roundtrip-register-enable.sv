// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

// Verify that two different flavors of enables on a register produce the
// expected IR structure with four-state logic support.

// CHECK-LABEL: hw.module @Foo
// CHECK-SAME: in %clk : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %en : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: in %d : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: out qa : !hw.struct<value: i1, unknown: i1>
// CHECK-SAME: out qb : !hw.struct<value: i1, unknown: i1>
module Foo (input logic clk, en, d, output logic qa, qb);
  // Verify clock derivation from four-state value
  // CHECK: hw.struct_extract %clk["value"]
  // CHECK: hw.struct_extract %clk["unknown"]
  // CHECK: seq.to_clock

  // Verify register outputs
  // CHECK: seq.firreg
  // CHECK: hw.output
  always @(posedge clk) begin
    if (en)
      qa <= d;

    qb <= en ? d : qb;
  end
endmodule
