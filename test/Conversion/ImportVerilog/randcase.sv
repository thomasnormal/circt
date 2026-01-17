// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test randcase statement handling (IEEE 1800-2017 Section 18.16).
// RandCase provides weighted random case selection for standalone statements.

module RandCaseTest;
  int x;
  int result;

  // CHECK-LABEL: moore.module @RandCaseTest

  // Basic randcase with constant weights
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      3: x = 1;  // weight 3
      1: x = 2;  // weight 1
      4: x = 3;  // weight 4
    endcase
  end

  // Randcase with variable assignment in each branch
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      10: result = 100;
      20: result = 200;
      30: result = 300;
    endcase
  end

  // Randcase with begin-end blocks
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      1: begin
        x = 10;
        result = x + 5;
      end
      2: begin
        x = 20;
        result = x * 2;
      end
    endcase
  end

  // Randcase with equal weights (uniform distribution)
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      1: x = 0;
      1: x = 1;
      1: x = 2;
      1: x = 3;
    endcase
  end

  // Randcase with only two branches
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      50: x = -1;
      50: x = 1;
    endcase
  end

  // Randcase with different weight ratios
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randcase
      1: x = 1;   // 10%
      9: x = 2;   // 90%
    endcase
  end

endmodule
