// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test that reading from ref function parameters uses llhd.prb.
// Signal references are correctly tracked after function inlining.

// CHECK-LABEL: func.func private @read_ref_param
// CHECK-SAME: (%[[ARG:.*]]: !llhd.ref<i32>)
// CHECK: %[[PROBED:.*]] = llhd.prb %[[ARG]] : i32
// CHECK: return %[[PROBED]] : i32
func.func private @read_ref_param(%arg0: !moore.ref<i32>) -> !moore.i32 {
  %0 = moore.read %arg0 : <i32>
  return %0 : !moore.i32
}
