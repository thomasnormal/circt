// RUN: circt-lec --emit-mlir --strict-llhd -c1=top -c2=top %s %s 2>&1 | FileCheck %s

// Strict LEC should allow multiple identical unconditional drives.

// CHECK: smt.solver
// CHECK: smt.check

module {
  hw.module @top(in %a : i1, out o : i1) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %a : i1
    llhd.drv %sig, %a after %t0 : i1
    llhd.drv %sig, %a after %t0 : i1
    %val = llhd.prb %sig : i1
    hw.output %val : i1
  }
}
