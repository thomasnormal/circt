// RUN: circt-bmc -b 1 --k-induction --run-smtlib --z3-path=%S/Inputs/fake-z3-unsat.sh --module top %s | FileCheck %s

// Exercise liveness-style source property through the end-to-end induction
// pipeline (SVA/LTL lowering + final-check handling).
hw.module @top(in %clk: !seq.clock, in %a: i1, out out: i1) {
  %ev = ltl.eventually %a : i1
  %clock = seq.from_clock %clk
  %clocked = ltl.clock %ev, posedge %clock : !ltl.property
  verif.clocked_assert %clocked, posedge %clock : !ltl.property
  hw.output %a : i1
}

// CHECK: BMC_BASE=UNSAT
// CHECK: BMC_STEP=UNSAT
// CHECK: BMC_RESULT=UNSAT
