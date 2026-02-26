// RUN: circt-bmc -b 5 --module top --run-smtlib %s | FileCheck %s
// CHECK: BMC_RESULT=UNSAT

// A clocked assertion with a constant-false clock is never sampled.
// It must not produce a violation even if the asserted expression is false.

module {
  hw.module @top() {
    %false = hw.constant false
    verif.clocked_assert %false, posedge %false : i1
    hw.output
  }
}
