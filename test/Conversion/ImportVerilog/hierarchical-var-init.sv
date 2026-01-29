// RUN: circt-verilog %s --ir-moore 2>&1 | FileCheck %s

// Test that variables with hierarchical initializers are properly handled.
// These must be processed after instances are created.

module Sub;
  logic [7:0] data = 8'hAB;
endmodule

// CHECK-LABEL: moore.module @Top
module Top(output logic [7:0] result);
  // CHECK: moore.instance "u_sub" @Sub
  Sub u_sub();

  // Variable with hierarchical initializer - processed after instance
  // CHECK: [[READ:%.+]] = moore.read %u_sub.data
  // CHECK: %captured = moore.variable [[READ]]
  logic [7:0] captured = u_sub.data;

  // Use the variable so it's not DCE'd
  // CHECK: moore.read %captured
  assign result = captured;
endmodule
