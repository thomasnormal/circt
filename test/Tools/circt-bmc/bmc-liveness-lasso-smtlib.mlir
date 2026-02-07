// RUN: circt-bmc --emit-smtlib -b 2 --module top --liveness %s | FileCheck %s --check-prefix=NO-LASSO
// RUN: circt-bmc --emit-smtlib -b 2 --module top --liveness --liveness-lasso %s | FileCheck %s --check-prefix=LASSO

// Liveness-lasso mode should emit an additional loop-closure assertion over
// BMC state snapshots.
// NO-LASSO: (declare-const r_state (_ BitVec 1))
// NO-LASSO-NOT: (= r_state
// NO-LASSO: (check-sat)
// LASSO: (declare-const r_state (_ BitVec 1))
// LASSO: (= r_state
// LASSO: (check-sat)

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  %r = seq.compreg %in, %clk : i1
  verif.assert %r {bmc.final} : i1
  hw.output
}
