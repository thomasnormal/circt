// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // Residual LLHD in helper funcs should be ignored by require-no-llhd checks.
  func.func private @helper(%v: i1) -> i1 {
    %sig = llhd.sig %v : i1
    %p = llhd.prb %sig : i1
    return %p : i1
  }

  hw.module @m(in %a : i1, out out_o : i1) {
    hw.output %a : i1
  }
}

// CHECK: func.func private @helper
// CHECK: llhd.sig
// CHECK: hw.module @m
