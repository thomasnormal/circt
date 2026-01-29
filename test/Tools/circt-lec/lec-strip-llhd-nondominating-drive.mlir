// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @top(out o : i1) {
    %c0 = hw.constant 0 : i1
    %sig = llhd.sig %c0 : i1
    %probe = llhd.prb %sig : i1
    %c1 = hw.constant 1 : i1
    %late = comb.xor %c1, %c1 : i1
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %sig, %late after %t0 : i1
    hw.output %probe : i1
  }
}

// CHECK-NOT: llhd.
