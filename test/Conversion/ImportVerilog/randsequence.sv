// RUN: circt-verilog --ir-moore %s | FileCheck %s
// REQUIRES: slang

// Test randsequence statement handling.
// RandSequence is a complex verification feature (IEEE 1800-2017 Section 18.17)
// that generates random sequences of productions.

module RandSequenceTest;
  int data;

  // CHECK-LABEL: moore.module @RandSequenceTest

  // Basic randsequence with simple productions
  // CHECK: moore.procedure initial
  initial begin
    randsequence(main)
      main: first second third;
      first: { data = 1; };
      second: { data = 2; };
      third: { data = 4; };
    endsequence
  end

  // Randsequence with weighted alternatives (uses $urandom_range)
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randsequence(start)
      start: a b;
      a: { data = 10; } := 3 | { data = 20; } := 7;
      b: { data = 30; };
    endsequence
  end

  // Randsequence with if-else production
  // CHECK: moore.procedure initial
  initial begin
    static bit cond = 1;
    randsequence(top)
      top: if (cond) branch_a else branch_b;
      branch_a: { data = 100; };
      branch_b: { data = 200; };
    endsequence
  end

  // Randsequence with repeat
  // CHECK: moore.procedure initial
  initial begin
    randsequence(loop_test)
      loop_test: repeat(3) item;
      item: { data = data + 1; };
    endsequence
  end

  // Randsequence with case production
  // CHECK: moore.procedure initial
  initial begin
    static int sel = 1;
    randsequence(case_test)
      case_test: case (sel)
        0: opt_a;
        1: opt_b;
        default: opt_c;
      endcase;
      opt_a: { data = 1000; };
      opt_b: { data = 2000; };
      opt_c: { data = 3000; };
    endsequence
  end

  // Randsequence with production arguments and defaults
  // CHECK: moore.procedure initial
  initial begin
    randsequence(arg_test)
      arg_test: prod(5) prod_default();
      prod(int x): { data = x; };
      prod_default(int y = 7): { data = y; };
    endsequence
  end

  // Randsequence with rand join(1) production
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randsequence(join_test)
      join_test: rand join (1) j1 j2 j3;
      j1: { data = 11; };
      j2: { data = 22; };
      j3: { data = 33; };
    endsequence
  end

  // Randsequence with rand join(2) production - select 2 distinct
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  // CHECK: moore.builtin.urandom_range
  initial begin
    randsequence(join_two)
      join_two: rand join (2) k1 k2 k3;
      k1: { data = 101; };
      k2: { data = 202; };
      k3: { data = 303; };
    endsequence
  end

  // Randsequence with rand join(all) production - executes all sequentially
  // CHECK: moore.procedure initial
  // CHECK: moore.builtin.urandom_range
  initial begin
    randsequence(join_all)
      join_all: rand join (3) m1 m2 m3;
      m1: { data = 111; };
      m2: { data = 222; };
      m3: { data = 333; };
    endsequence
  end

  // Randsequence break/return handling in productions
  // CHECK-LABEL: func.func @break_return_test
  // CHECK: cf.br
  // CHECK: func.return
  function int break_return_test();
    int x;
    randsequence(main)
      main: first second third;
      first: { x = 10; };
      second: { if (x) break; };
      third: { x = 20; };
    endsequence
    return x;
  endfunction

  // CHECK-LABEL: func.func @return_in_production
  // CHECK: cf.br
  // CHECK: func.return
  function int return_in_production();
    int x;
    randsequence(main)
      main: first second third;
      first: { x = 20; };
      second: { if (x) return; x = 5; };
      third: { x = x + 5; };
    endsequence
    return x;
  endfunction

  // Randsequence break in forked randjoin production
  // CHECK-LABEL: func.func @randjoin_break
  // CHECK: moore.fork
  // CHECK: cf.br
  function int randjoin_break();
    int x;
    randsequence(main)
      main: rand join (2) a b c;
      a: { x = 1; };
      b: { if (x) break; x = 2; };
      c: { x = 3; };
    endsequence
    return x;
  endfunction

endmodule
