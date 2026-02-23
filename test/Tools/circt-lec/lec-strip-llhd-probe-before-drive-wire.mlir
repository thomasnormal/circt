// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  hw.module @top(in %in : i1, out out : i1) {
    %false = hw.constant false
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %src = llhd.sig name "in" %in : i1
    %sig = llhd.sig %false : i1
    %probe = llhd.prb %sig : i1
    %srcv = llhd.prb %src : i1
    llhd.drv %sig, %srcv after %t0 : i1
    hw.output %probe : i1
  }
}

// CHECK-LABEL: hw.module @top(
// CHECK-NOT: llhd.
// CHECK: hw.output %in : i1
