// RUN: not circt-bmc --emit-smtlib -b 1 --module top %s 2>&1 | FileCheck %s

hw.module @top(in %clk: !seq.clock, in %in: i1) {
  verif.assert %in : i1
  hw.output
}

// CHECK: Printing SMT-LIB not yet supported!
