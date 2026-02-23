// RUN: circt-bmc -b 1 --run --z3-path=%S/Inputs/fake-z3-unsat.sh --module top %s | FileCheck %s

// CHECK: BMC_RESULT=UNSAT
// CHECK: Bound reached with no violations!

module {
  hw.module @top(in %clk : i1) {
    %c1 = hw.constant true
    verif.assert %c1 : i1
    hw.output
  }
}
