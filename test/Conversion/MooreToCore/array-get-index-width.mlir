// RUN: circt-opt --convert-moore-to-core %s | FileCheck %s

// Test that array get with wide index is correctly truncated to the minimum
// required width. hw::ArrayGetOp requires the index to have exactly
// ceil(log2(array_size)) bits.

// CHECK-LABEL: hw.module @TestArrayGetIndexWidth4
moore.module @TestArrayGetIndexWidth4(in %arr : !moore.uarray<4 x i32>, in %idx : !moore.i32, out result : !moore.i32) {
  // Array size 4 requires ceil(log2(4)) = 2 bits for index
  // CHECK: comb.extract %idx from 0 : (i32) -> i2
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<4xi32>, i2
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<4 x i32>, !moore.i32 -> !moore.i32
  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @TestArrayGetIndexWidth32
moore.module @TestArrayGetIndexWidth32(in %arr : !moore.uarray<32 x i8>, in %idx : !moore.i32, out result : !moore.i8) {
  // Array size 32 requires ceil(log2(32)) = 5 bits for index
  // CHECK: comb.extract %idx from 0 : (i32) -> i5
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<32xi8>, i5
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<32 x i8>, !moore.i32 -> !moore.i8
  moore.output %0 : !moore.i8
}

// CHECK-LABEL: hw.module @TestArrayGetIndexWidth2
moore.module @TestArrayGetIndexWidth2(in %arr : !moore.uarray<2 x i16>, in %idx : !moore.i32, out result : !moore.i16) {
  // Array size 2 requires ceil(log2(2)) = 1 bit for index
  // CHECK: comb.extract %idx from 0 : (i32) -> i1
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<2xi16>, i1
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<2 x i16>, !moore.i32 -> !moore.i16
  moore.output %0 : !moore.i16
}

// CHECK-LABEL: hw.module @TestArrayGetIndexWidth16
moore.module @TestArrayGetIndexWidth16(in %arr : !moore.uarray<16 x i8>, in %idx : !moore.i8, out result : !moore.i8) {
  // Array size 16 requires ceil(log2(16)) = 4 bits for index
  // Index is already i8 (wider than 4), should truncate to i4
  // CHECK: comb.extract %idx from 0 : (i8) -> i4
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<16xi8>, i4
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<16 x i8>, !moore.i8 -> !moore.i8
  moore.output %0 : !moore.i8
}

// CHECK-LABEL: hw.module @TestArrayGetNarrowIndex
moore.module @TestArrayGetNarrowIndex(in %arr : !moore.uarray<256 x i8>, in %idx : !moore.i4, out result : !moore.i8) {
  // Array size 256 requires ceil(log2(256)) = 8 bits for index
  // Index is i4 (narrower than 8), should zero-extend to i8
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i4
  // CHECK: comb.concat [[ZERO]], %idx : i4, i4
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<256xi8>, i8
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<256 x i8>, !moore.i4 -> !moore.i8
  moore.output %0 : !moore.i8
}

// CHECK-LABEL: hw.module @TestArrayGetExactWidth
moore.module @TestArrayGetExactWidth(in %arr : !moore.uarray<8 x i32>, in %idx : !moore.i3, out result : !moore.i32) {
  // Array size 8 requires ceil(log2(8)) = 3 bits for index
  // Index is already i3, no conversion needed
  // CHECK: hw.array_get %arr[%idx] : !hw.array<8xi32>, i3
  %0 = moore.dyn_extract %arr from %idx : !moore.uarray<8 x i32>, !moore.i3 -> !moore.i32
  moore.output %0 : !moore.i32
}

// CHECK-LABEL: hw.module @TestPackedArrayGetIndexWidth
moore.module @TestPackedArrayGetIndexWidth(in %arr : !moore.array<4 x i32>, in %idx : !moore.i32, out result : !moore.i32) {
  // Packed array with size 4 also requires ceil(log2(4)) = 2 bits
  // CHECK: comb.extract %idx from 0 : (i32) -> i2
  // CHECK: hw.array_get %arr[%{{.*}}] : !hw.array<4xi32>, i2
  %0 = moore.dyn_extract %arr from %idx : !moore.array<4 x i32>, !moore.i32 -> !moore.i32
  moore.output %0 : !moore.i32
}
