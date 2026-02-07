// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that moore.stream_unpack correctly handles 4-state (logic) source types.
// The 4-state type !moore.l8 is converted to a hw::StructType {value: i8, unknown: i8}.
// The conversion must extract the value field before extending to i64 for the
// runtime call.

// CHECK-LABEL: func.func @StreamUnpackFourState
func.func @StreamUnpackFourState(%dst: !moore.ref<open_uarray<i1>>, %src: !moore.l8) {
  // CHECK: hw.struct_extract {{.*}}["value"]
  // CHECK: arith.extui {{.*}} : i8 to i64
  // CHECK: llvm.call @__moore_stream_unpack_bits
  moore.stream_unpack %dst, %src {isRightToLeft = true} : !moore.ref<open_uarray<i1>>, !moore.l8
  return
}

// Also test with a 2-state source type (should still work, no struct_extract needed)
// CHECK-LABEL: func.func @StreamUnpackTwoState
func.func @StreamUnpackTwoState(%dst: !moore.ref<open_uarray<i1>>, %src: !moore.i8) {
  // CHECK-NOT: hw.struct_extract
  // CHECK: arith.extui {{.*}} : i8 to i64
  // CHECK: llvm.call @__moore_stream_unpack_bits
  moore.stream_unpack %dst, %src : !moore.ref<open_uarray<i1>>, !moore.i8
  return
}

// Test with wider 4-state type
// CHECK-LABEL: func.func @StreamUnpackFourStateWide
func.func @StreamUnpackFourStateWide(%dst: !moore.ref<open_uarray<i1>>, %src: !moore.l32) {
  // CHECK: hw.struct_extract {{.*}}["value"]
  // CHECK: arith.extui {{.*}} : i32 to i64
  // CHECK: llvm.call @__moore_stream_unpack_bits
  moore.stream_unpack %dst, %src {isRightToLeft = true} : !moore.ref<open_uarray<i1>>, !moore.l32
  return
}

// Test with queue destination and 4-state source
// CHECK-LABEL: func.func @StreamUnpackFourStateQueue
func.func @StreamUnpackFourStateQueue(%dst: !moore.ref<queue<i8, 0>>, %src: !moore.l64) {
  // CHECK: hw.struct_extract {{.*}}["value"]
  // CHECK: llvm.call @__moore_stream_unpack_bits
  moore.stream_unpack %dst, %src : !moore.ref<queue<i8, 0>>, !moore.l64
  return
}
