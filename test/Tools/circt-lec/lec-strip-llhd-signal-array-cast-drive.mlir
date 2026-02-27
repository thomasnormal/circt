// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

module {
  hw.module @m(out out_o : !hw.array<2xstruct<value: i4, unknown: i4>>) {
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %initBits = hw.constant 0 : i16
    %init = hw.bitcast %initBits : (i16) -> !hw.array<2xstruct<value: i4, unknown: i4>>
    %sig = llhd.sig %init : !hw.array<2xstruct<value: i4, unknown: i4>>

    // Packed integer source is wrapped in an unrealized cast to the signal type.
    // Stripping must not unwrap this to i16 when replacing probes.
    %bits = hw.constant 51966 : i16
    %cast = builtin.unrealized_conversion_cast %bits : i16 to !hw.array<2xstruct<value: i4, unknown: i4>>
    llhd.drv %sig, %cast after %t0 : !hw.array<2xstruct<value: i4, unknown: i4>>

    %p = llhd.prb %sig : !hw.array<2xstruct<value: i4, unknown: i4>>
    hw.output %p : !hw.array<2xstruct<value: i4, unknown: i4>>
  }
}

// CHECK-NOT: llhd.
// CHECK: builtin.unrealized_conversion_cast
// CHECK: hw.output
