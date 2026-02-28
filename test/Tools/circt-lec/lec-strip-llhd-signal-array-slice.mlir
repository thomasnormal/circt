// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  // `llhd.sig.array_slice` paths should be stripped like other LLHD ref paths.
  hw.module @m(in %a : i8, out out_o : i8) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %c0 = hw.constant 0 : i8
    %arr0 = hw.array_create %c0, %c0, %c0, %c0 : i8
    %idx1 = hw.constant 1 : i2
    %idx0 = hw.constant 0 : i2
    %sig = llhd.sig %arr0 : !hw.array<4xi8>
    %slice = llhd.sig.array_slice %sig at %idx1 : <!hw.array<4xi8>> -> <!hw.array<3xi8>>
    %elem = llhd.sig.array_get %slice[%idx0] : <!hw.array<3xi8>>
    llhd.drv %elem, %a after %t0 : i8
    %p = llhd.prb %slice : !hw.array<3xi8>
    %pe = hw.array_get %p[%idx0] : !hw.array<3xi8>, i2
    hw.output %pe : i8
  }
}

// CHECK-LABEL: hw.module @m(
// CHECK-NOT: llhd.
// CHECK: hw.output
