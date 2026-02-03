// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test integer to queue conversion (bit unpacking)

// CHECK-LABEL: hw.module @int8_to_queue_i1
// Output result to prevent DCE
moore.module @int8_to_queue_i1(out result : !moore.queue<i1, 0>) {
  %0 = moore.constant 0 : i8
  // 8-bit integer unpacked to queue of 8 1-bit elements
  // CHECK-DAG: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
  // CHECK-DAG: llvm.mlir.zero
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: hw.output
  %1 = moore.conversion %0 : !moore.i8 -> !moore.queue<i1, 0>
  moore.output %1 : !moore.queue<i1, 0>
}

// CHECK-LABEL: hw.module @int1_to_queue_i1
moore.module @int1_to_queue_i1(out result : !moore.queue<i1, 0>) {
  %0 = moore.constant 1 : i1
  // Single bit to single-element queue
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: hw.output
  %1 = moore.conversion %0 : !moore.i1 -> !moore.queue<i1, 0>
  moore.output %1 : !moore.queue<i1, 0>
}

// CHECK-LABEL: hw.module @int16_to_queue_i8
moore.module @int16_to_queue_i8(out result : !moore.queue<i8, 0>) {
  %0 = moore.constant 0 : i16
  // 16-bit integer to queue of 8-bit elements (2 elements)
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: llvm.call @__moore_queue_push_back
  // CHECK: hw.output
  %1 = moore.conversion %0 : !moore.i16 -> !moore.queue<i8, 0>
  moore.output %1 : !moore.queue<i8, 0>
}
