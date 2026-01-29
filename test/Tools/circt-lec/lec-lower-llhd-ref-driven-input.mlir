// RUN: circt-lec --emit-mlir -c1=top -c2=top %s %s | FileCheck %s

module {
  hw.module @driver(in %sink : !llhd.ref<i1>) {
    %c0 = hw.constant 0 : i1
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %sink, %c0 after %t0 : i1
    hw.output
  }

  hw.module @reader(out r : !llhd.ref<i1>, out v : i1) {
    %c0 = hw.constant 0 : i1
    %r = llhd.sig %c0 : i1
    %v = llhd.prb %r : i1
    hw.output %r, %v : !llhd.ref<i1>, i1
  }

  hw.module @top(out v : i1) {
    %r, %v0 = hw.instance "u_read" @reader() -> (r: !llhd.ref<i1>, v: i1)
    hw.instance "u_drv" @driver(sink: %r: !llhd.ref<i1>) -> ()
    hw.output %v0 : i1
  }
}

// CHECK-NOT: llhd.
