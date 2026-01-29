// RUN: not circt-lec --emit-mlir --lec-strict -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// CHECK: LLHD signal requires abstraction; rerun without --strict-llhd

module {
  hw.module @top(in %a : i1, in %b : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : i1
    llhd.drv %sig, %a after %t0 : i1
    llhd.drv %sig, %b after %t0 : i1
    %p = llhd.prb %sig : i1
    hw.output %p : i1
  }
}
