// RUN: circt-bmc --emit-smtlib -b 1 --module top %s | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  verif.assert %in : i1
  hw.output
}

// CHECK: (declare-const in
// CHECK: (check-sat)
