// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  hw.module @m(in %a : i8, out out_o : i8) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c0 = hw.constant 0 : i8
    %arr0 = hw.array_create %c0, %c0, %c0, %c0 : i8
    %idx = hw.constant 2 : i2
    %sig = llhd.sig %arr0 : !hw.array<4xi8>
    %comb = llhd.combinational -> i8 {
      %elem = llhd.sig.array_get %sig[%idx] : <!hw.array<4xi8>>
      llhd.drv %elem, %a after %t0 : i8
      %p = llhd.prb %sig : !hw.array<4xi8>
      %pe = hw.array_get %p[%idx] : !hw.array<4xi8>, i2
      llhd.yield %pe : i8
    }
    hw.output %comb : i8
  }
}

// CHECK: hw.array_inject
// CHECK-NOT: llhd.
