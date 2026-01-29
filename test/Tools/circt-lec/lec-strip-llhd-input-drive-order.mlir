// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @top(in %in : i1, out out : i1) {
    %false = hw.constant false
    %t = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %false : i1
    %0 = llhd.prb %sig : i1
    llhd.drv %sig, %in after %t : i1
    hw.output %0 : i1
  }
}

// CHECK: smt.declare_fun "in"
// CHECK-NOT: llhd.
