// RUN: circt-bmc -b 1 --module top --run-smtlib %s | FileCheck %s
// CHECK: BMC_RESULT=UNSAT
module {
  hw.module @top() {
    hw.output
  }
}

