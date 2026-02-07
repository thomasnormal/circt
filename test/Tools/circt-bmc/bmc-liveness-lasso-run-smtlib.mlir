// RUN: circt-bmc --run-smtlib --z3-path=%S/Inputs/fake-z3-unsat.sh -b 2 --module top --liveness --liveness-lasso %s | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  %r = seq.compreg %in, %clk : i1
  verif.assert %r {bmc.final} : i1
  hw.output
}

// CHECK: BMC_RESULT=UNSAT
