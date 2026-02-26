// REQUIRES: z3
// RUN: circt-bmc --run-smtlib -b 2 --module m %s | FileCheck %s
// CHECK: BMC_RESULT=UNSAT

module {
  hw.module @m(in %clk : i1) {
    %false = hw.constant false
    verif.clocked_cover %false, posedge %clk : i1
    hw.output
  }
}
