// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @child(out r : !llhd.ref<i1>, out o : i1) {
    %c0 = hw.constant 0 : i1
    %c1 = hw.constant 1 : i1
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %r = llhd.sig %c0 : i1
    llhd.drv %r, %c1 after %t0 : i1
    %v = llhd.prb %r : i1
    hw.output %r, %v : !llhd.ref<i1>, i1
  }

  hw.module @top(in %i : i1, out o : i1) {
    %r, %v0 = hw.instance "u" @child() -> (r: !llhd.ref<i1>, o: i1)
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %r, %i after %t0 : i1
    hw.output %v0 : i1
  }
}

// CHECK: smt.solver
// CHECK-NOT: llhd.
