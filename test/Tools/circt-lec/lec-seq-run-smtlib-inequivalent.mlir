// REQUIRES: z3
// RUN: circt-lec --run-smtlib --bound 3 -c1=counterA -c2=counterBad %s | FileCheck %s

// counterA increments by 1, counterBad increments by 2 — not equivalent.
hw.module @counterA(in %clk: !seq.clock,
                    in %r_state: i8, out out: i8, out r_next: i8)
    attributes {num_regs = 1 : i32, initial_values = [0 : i8]} {
  %one = hw.constant 1 : i8
  %sum = comb.add %r_state, %one : i8
  hw.output %sum, %sum : i8, i8
}

hw.module @counterBad(in %clk: !seq.clock,
                      in %r_state: i8, out out: i8, out r_next: i8)
    attributes {num_regs = 1 : i32, initial_values = [0 : i8]} {
  %two = hw.constant 2 : i8
  %sum = comb.add %r_state, %two : i8
  hw.output %sum, %sum : i8, i8
}

// CHECK: c1 != c2
// CHECK: LEC_RESULT=NEQ
