// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s
// REQUIRES: slang

// Test procedural assign/deassign statements (IEEE 1800-2017 Section 10.6)
// These are deprecated simulation constructs that we convert to blocking
// assignments with simplified semantics.

// CHECK-LABEL: moore.module @ProceduralAssignTest
module ProceduralAssignTest(input clk, input d, input clr, input set, output logic q);

  // CHECK: moore.procedure always
  always @(clr or set)
    if (clr)
      // Procedural assign - converted to blocking_assign
      // CHECK: moore.blocking_assign
      assign q = 0;
    else if (set)
      // Procedural assign - converted to blocking_assign
      // CHECK: moore.blocking_assign
      assign q = 1;
    else
      // Deassign - becomes no-op
      deassign q;

  // CHECK: moore.procedure always
  always @(posedge clk)
    q <= d;

endmodule

// Test force/release in a simple module-local context
// CHECK-LABEL: moore.module @ForceReleaseTest
module ForceReleaseTest(input d, input f, output logic q);

  // CHECK: moore.procedure always
  always @(f or d)
    if (f)
      // CHECK: moore.force_assign
      force q = d;
    else
      // CHECK: moore.release_assign
      release q;

endmodule
