// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test randsequence rand join(N) statement handling.
// IEEE 1800-2017 Section 18.17 specifies that rand join(N) should select
// N distinct productions from the alternatives to execute.

module RandJoinTest;
  int result;
  int count;

  // CHECK-LABEL: moore.module @RandJoinTest

  // Test rand join(1) - select exactly one production randomly
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    result = 0;
    randsequence(test1)
      test1: rand join (1) p1 p2 p3;
      p1: { result = result + 1; };
      p2: { result = result + 10; };
      p3: { result = result + 100; };
    endsequence
    // result should be 1, 10, or 100
  end

  // Test rand join(2) - select exactly two distinct productions
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  // CHECK: moore.builtin.urandom_range
  initial begin
    result = 0;
    randsequence(test2)
      test2: rand join (2) a b c d;
      a: { result = result + 1; };
      b: { result = result + 2; };
      c: { result = result + 4; };
      d: { result = result + 8; };
    endsequence
    // result should be sum of exactly 2 distinct values from {1,2,4,8}
    // Possible: 3, 5, 6, 9, 10, 12
  end

  // Test rand join(3) - select three distinct productions
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    count = 0;
    randsequence(test3)
      test3: rand join (3) x1 x2 x3 x4 x5;
      x1: { count = count + 1; };
      x2: { count = count + 1; };
      x3: { count = count + 1; };
      x4: { count = count + 1; };
      x5: { count = count + 1; };
    endsequence
    // count should always be exactly 3
  end

  // Test rand join(N) where N >= number of alternatives - execute all
  // CHECK: moore.procedure initial
  initial begin
    result = 0;
    randsequence(test_all)
      test_all: rand join (5) m1 m2 m3;
      m1: { result = result + 1; };
      m2: { result = result + 2; };
      m3: { result = result + 4; };
    endsequence
    // N=5 >= 3 alternatives, so all are executed
    // result should be 7 (1+2+4)
  end

  // Test rand join(N) where N equals number of alternatives
  // CHECK: moore.procedure initial
  initial begin
    result = 0;
    randsequence(test_exact)
      test_exact: rand join (3) e1 e2 e3;
      e1: { result = result + 100; };
      e2: { result = result + 200; };
      e3: { result = result + 400; };
    endsequence
    // N=3 == 3 alternatives, so all are executed
    // result should be 700 (100+200+400)
  end

  // Test rand join with production arguments
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    result = 0;
    randsequence(test_args)
      test_args: rand join (2) arg_prod(10) arg_prod(20) arg_prod(30);
      arg_prod(int val): { result = result + val; };
    endsequence
    // Should execute 2 of {10, 20, 30}: possible sums 30, 40, 50
  end

endmodule
