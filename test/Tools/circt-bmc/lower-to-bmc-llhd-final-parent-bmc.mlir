// RUN: circt-opt --externalize-registers --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

hw.module @top() {
  llhd.final {
    llhd.halt
  }
  hw.output
}

// CHECK: verif.bmc bound
// CHECK: llhd.final
