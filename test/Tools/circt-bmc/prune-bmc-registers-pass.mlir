// RUN: circt-opt --prune-bmc-registers %s | FileCheck %s

// CHECK: hw.module @top
module {
  hw.module @top() {
    hw.output
  }
}
