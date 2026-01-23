// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test for case inside set membership pattern matching (SystemVerilog 12.5.4)

// CHECK-LABEL: moore.module @CaseInside
module CaseInside(
  input logic [3:0] a,
  output logic [3:0] b
);
  // Test case inside with:
  // - Multiple values in same item (1, 3)
  // - Wildcard pattern (4'b01??)
  // - Range expression ([5:6])
  //
  // CHECK: moore.procedure always_comb
  // CHECK: moore.wildcard_eq
  // CHECK: moore.wildcard_eq
  // CHECK: moore.wildcard_eq
  // CHECK: moore.uge
  // CHECK: moore.ule
  // CHECK: moore.and
  always_comb begin
    case (a) inside
      1, 3: b = 1;
      4'b01??, [5:6]: b = 2;
      default: b = 3;
    endcase
  end
endmodule

// CHECK-LABEL: moore.module @CaseInsideSignedRange
module CaseInsideSignedRange(
  input logic signed [7:0] x,
  output logic [1:0] y
);
  // Test case inside with signed range comparison
  // CHECK: moore.procedure always_comb
  // CHECK: moore.sge
  // CHECK: moore.sle
  always_comb begin
    case (x) inside
      [-10:-1]: y = 1;
      [0:10]: y = 2;
      default: y = 0;
    endcase
  end
endmodule
