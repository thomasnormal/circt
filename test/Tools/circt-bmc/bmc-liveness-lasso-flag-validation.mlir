// RUN: not circt-bmc --emit-smtlib -b 1 --module top --liveness-lasso %s 2>&1 | FileCheck %s --check-prefix=REQUIRES-LIVENESS
// RUN: not circt-bmc --emit-mlir -b 1 --module top --liveness --liveness-lasso %s 2>&1 | FileCheck %s --check-prefix=REQUIRES-SMTLIB

// REQUIRES-LIVENESS: --liveness-lasso requires --liveness
// REQUIRES-SMTLIB: --liveness-lasso requires --emit-smtlib or --run-smtlib

hw.module @top(in %a: i1) {
  verif.assert %a {bmc.final} : i1
  hw.output
}
