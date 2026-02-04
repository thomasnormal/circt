// RUN: circt-bmc -b 1 --module top --emit-smtlib %s | FileCheck %s
// CHECK: (assert false)
// CHECK: (check-sat)
module {
  hw.module @top() {
    hw.output
  }
}

