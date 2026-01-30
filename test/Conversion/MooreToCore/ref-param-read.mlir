// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test that reading from ref function parameters uses llvm.load.
// Function ref parameters are memory pointers, not signals, so we use
// llvm.load instead of llhd.prb. The simulator cannot track signal
// references through function call boundaries for ref parameters.

// CHECK-LABEL: func.func private @read_ref_param
// CHECK-SAME: (%[[ARG:.*]]: !llhd.ref<i32>)
// CHECK: %[[PTR:.*]] = builtin.unrealized_conversion_cast %[[ARG]] : !llhd.ref<i32> to !llvm.ptr
// CHECK: %[[LOADED:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
// CHECK: return %[[LOADED]] : i32
func.func private @read_ref_param(%arg0: !moore.ref<i32>) -> !moore.i32 {
  %0 = moore.read %arg0 : <i32>
  return %0 : !moore.i32
}
