// RUN: not circt-bmc --run-smtlib -b 1 --module top --k-induction --liveness %s 2>&1 | FileCheck %s
// RUN: not circt-bmc --run-smtlib -b 1 --module top --induction --liveness %s 2>&1 | FileCheck %s

// CHECK: --liveness is incompatible with --induction/--k-induction
hw.module @top(in %clk: !seq.clock, in %a: i1) {
  verif.assert %a : i1
  hw.output
}
