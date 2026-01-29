// RUN: circt-opt --hw-eliminate-inout-ports=allow-multiple-writers-same-value %s | FileCheck %s

// CHECK-LABEL:   hw.module @multiSame(out a_wr : i42) {
// CHECK:           hw.constant 0 : i42
// CHECK:           %[[VAL_0:.*]] = hw.constant 0 : i42
// CHECK:           hw.output %[[VAL_0]] : i42
// CHECK:         }
hw.module @multiSame(inout %a: i42) {
  %0 = hw.constant 0 : i42
  %1 = hw.constant 0 : i42
  sv.assign %a, %0 : i42
  sv.assign %a, %1 : i42
}
