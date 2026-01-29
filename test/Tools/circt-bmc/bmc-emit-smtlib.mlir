// XFAIL: *
// The BMC pipeline now generates solver scopes with results, which the SMTLIB
// emitter doesn't support. This test is disabled until SMTLIB emission is
// updated to handle structured control flow.
// RUN: circt-bmc --emit-smtlib -b 1 --module top %s | FileCheck %s

// CHECK: (declare-const in
// CHECK: (check-sat)

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  verif.assert %in : i1
  hw.output
}
