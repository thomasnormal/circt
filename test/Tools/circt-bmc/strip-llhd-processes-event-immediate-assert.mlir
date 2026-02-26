// RUN: circt-opt --strip-llhd-processes %s | FileCheck %s

module {
  // Event-only wait process should preserve immediate assert-like checks.
  // CHECK-LABEL: hw.module @event_only_assert_proc
  // CHECK: verif.assert
  // CHECK-NOT: llhd.process
  hw.module @event_only_assert_proc() {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %init = hw.constant false
    %sig = llhd.sig %init : i1
    llhd.process {
      %cur = llhd.prb %sig : i1
      verif.assert %cur : i1
      llhd.wait (%cur : i1), ^bb1
    ^bb1:
      %next = llhd.prb %sig : i1
      verif.assert %next : i1
      llhd.wait (%next : i1), ^bb1
    }
    llhd.drv %sig, %init after %t0 : i1
    hw.output
  }
}
