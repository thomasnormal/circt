// RUN: circt-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @delay_fold
// CHECK-NOT: llhd.delay %in by <0ns, 0d, 0e>
// CHECK: %[[NONZERO:.*]] = llhd.delay %in by <1ns, 0d, 0e> : i1
// CHECK: hw.output %in, %[[NONZERO]] : i1, i1
hw.module @delay_fold(in %in : i1, out out0 : i1, out out1 : i1) {
  %0 = llhd.delay %in by <0ns, 0d, 0e> : i1
  %1 = llhd.delay %in by <1ns, 0d, 0e> : i1
  hw.output %0, %1 : i1, i1
}
