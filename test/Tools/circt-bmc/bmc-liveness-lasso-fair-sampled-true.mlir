// RUN: circt-bmc --emit-smtlib -b 1 --module top --liveness --liveness-lasso %s | FileCheck %s

// A sampled final assert that is always true must not be accepted as a
// liveness counterexample loop. Lasso fairness now requires the sampled final
// assertion to remain unsatisfied along the chosen loop segment.
// CHECK: (assert false)

hw.module @top(in %clk: !seq.clock, in %a: i1) {
  %true = hw.constant true
  verif.assert %true {bmc.final, bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge posedge>} : i1
  hw.output
}
