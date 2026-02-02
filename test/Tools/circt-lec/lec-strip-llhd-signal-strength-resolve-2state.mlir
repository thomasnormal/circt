// RUN: circt-opt --strip-llhd-interface-signals %s | FileCheck %s

hw.module @sig_strength_2state(out o: i1) {
  %t0 = llhd.constant_time <0ns, 0d, 1e>
  %zero = hw.constant 0 : i1
  %one = hw.constant 1 : i1
  %sig = llhd.sig %zero : i1
  llhd.drv %sig, %zero after %t0 strength(strong, strong) : i1
  llhd.drv %sig, %one after %t0 strength(strong, strong) : i1
  %p0 = llhd.prb %sig : i1
  hw.output %p0 : i1
}

// CHECK-LABEL: hw.module @sig_strength_2state
// CHECK: in %[[UNK:[^ ]+]]{{ *}}: i1
// CHECK: comb.and bin %[[UNK]], %{{.*}} : i1
// CHECK: hw.output %{{.*}} : i1
// CHECK-NOT: llhd.drv
// CHECK-NOT: llhd.prb
