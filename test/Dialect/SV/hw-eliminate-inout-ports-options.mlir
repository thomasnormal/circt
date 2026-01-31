// RUN: circt-opt --hw-eliminate-inout-ports=allow-multiple-writers-same-value %s | FileCheck %s

// CHECK: hw.module @top
module {
  hw.module @top() {
    hw.output
  }
}
