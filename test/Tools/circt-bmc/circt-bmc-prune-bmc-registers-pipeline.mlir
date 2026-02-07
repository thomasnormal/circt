// RUN: circt-bmc --emit-mlir --prune-bmc-registers=true -b 1 --module top %s | FileCheck %s --check-prefix=PRUNE
// RUN: circt-bmc --emit-mlir --prune-bmc-registers=false -b 1 --module top %s | FileCheck %s --check-prefix=NOPRUNE

// When pruning is enabled, dead register/input logic is removed from the BMC
// state space. Here, %b/%r_dead do not feed any property and should disappear.
// PRUNE-NOT: smt.declare_fun "b"
// PRUNE: func.func @bmc_circuit(%{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>)

// With pruning disabled, the dead `%b` input remains in the SMT problem.
// NOPRUNE: smt.declare_fun "b"
// NOPRUNE: func.func @bmc_circuit(%{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<1>, %{{.*}}: !smt.bv<1>) -> (!smt.bv<1>, !smt.bv<1>)

hw.module @top(in %clk : !seq.clock, in %a : i1, in %b : i1) {
  %r_live = seq.compreg %a, %clk : i1
  %r_dead = seq.compreg %b, %clk : i1
  verif.assert %r_live : i1
  hw.output
}
