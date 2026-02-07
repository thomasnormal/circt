// RUN: circt-bmc --emit-smtlib -b 1 --module top --liveness --liveness-lasso %s | FileCheck %s

// With a negedge-only final check and bound=1 (single posedge transition),
// lasso fairness rejects the vacuous loop because the final check is never
// sampled on the loop segment.
// CHECK: (assert false)

hw.module @top(in %clk: !seq.clock, in %a: i1) {
  %true = hw.constant true
  verif.assert %true {bmc.final, bmc.clock = "clk", bmc.clock_edge = #ltl<clock_edge negedge>} : i1
  hw.output
}
