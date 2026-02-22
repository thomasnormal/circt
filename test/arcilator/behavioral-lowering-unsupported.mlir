// RUN: not arcilator %s 2>&1 | FileCheck %s

module {
  hw.module @Top(out out: i1) {
    %false = hw.constant false
    %proc:1 = llhd.process -> i1 {
      llhd.wait yield (%false : i1), ^bb1
    ^bb1:
      llhd.halt %false : i1
    }
    hw.output %proc#0 : i1
  }
}

// CHECK: failed to legalize operation 'llhd.process'
