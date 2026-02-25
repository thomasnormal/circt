// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // Multiple element drives into an LLHD array signal should preserve the
  // array-typed root probe/output after stripping.
  hw.module @top(in %a : i1, in %b : i1, out out : !hw.array<2xi1>) {
    %false = hw.constant false
    %zero = hw.constant 0 : i1
    %one = hw.constant 1 : i1
    %init = hw.array_create %false, %false : i1
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %init : !hw.array<2xi1>
    %r0 = llhd.sig.array_get %sig[%zero] : <!hw.array<2xi1>>
    %r1 = llhd.sig.array_get %sig[%one] : <!hw.array<2xi1>>
    llhd.drv %r0, %a after %t0 : i1
    llhd.drv %r1, %b after %t0 : i1
    %p = llhd.prb %sig : !hw.array<2xi1>
    hw.output %p : !hw.array<2xi1>
  }
}

// CHECK-LABEL: hw.module @top(
// CHECK-NOT: llhd.
// CHECK: hw.output {{.*}} : !hw.array<2xi1>
