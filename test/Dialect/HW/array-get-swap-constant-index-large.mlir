// RUN: circt-opt -canonicalize='top-down=true region-simplify=aggressive' %s | FileCheck %s

// This is a regression test for `hw.array_get` canonicalization. The
// SwapConstantIndex transform is useful for small arrays, but can blow up IR
// size when the array is large. We cap it at a conservative threshold.

// CHECK-LABEL: hw.module @SwapConstantIndexLarge(
hw.module @SwapConstantIndexLarge(
    in %a_0: !hw.array<4xi1>, in %a_1: !hw.array<4xi1>, in %a_2: !hw.array<4xi1>, in %a_3: !hw.array<4xi1>,
    in %a_4: !hw.array<4xi1>, in %a_5: !hw.array<4xi1>, in %a_6: !hw.array<4xi1>, in %a_7: !hw.array<4xi1>,
    in %a_8: !hw.array<4xi1>, in %a_9: !hw.array<4xi1>, in %a_10: !hw.array<4xi1>, in %a_11: !hw.array<4xi1>,
    in %a_12: !hw.array<4xi1>, in %a_13: !hw.array<4xi1>, in %a_14: !hw.array<4xi1>, in %a_15: !hw.array<4xi1>,
    in %a_16: !hw.array<4xi1>, in %a_17: !hw.array<4xi1>, in %a_18: !hw.array<4xi1>, in %a_19: !hw.array<4xi1>,
    in %a_20: !hw.array<4xi1>, in %a_21: !hw.array<4xi1>, in %a_22: !hw.array<4xi1>, in %a_23: !hw.array<4xi1>,
    in %a_24: !hw.array<4xi1>, in %a_25: !hw.array<4xi1>, in %a_26: !hw.array<4xi1>, in %a_27: !hw.array<4xi1>,
    in %a_28: !hw.array<4xi1>, in %a_29: !hw.array<4xi1>, in %a_30: !hw.array<4xi1>, in %a_31: !hw.array<4xi1>,
    in %a_32: !hw.array<4xi1>, in %sel: i6, out b: i1) {
  %c0_i2 = hw.constant 0 : i2
  %0 = hw.array_create %a_32, %a_31, %a_30, %a_29, %a_28, %a_27, %a_26, %a_25, %a_24, %a_23, %a_22, %a_21, %a_20, %a_19, %a_18, %a_17, %a_16, %a_15, %a_14, %a_13, %a_12, %a_11, %a_10, %a_9, %a_8, %a_7, %a_6, %a_5, %a_4, %a_3, %a_2, %a_1, %a_0 : !hw.array<4xi1>
  %1 = hw.array_get %0[%sel] : !hw.array<33xarray<4xi1>>, i6
  %2 = hw.array_get %1[%c0_i2] : !hw.array<4xi1>, i2
  hw.output %2 : i1

  // CHECK-NOT: hw.array_get %a_
  // CHECK: %[[ARR:.*]] = hw.array_create {{.*}} : !hw.array<4xi1>
  // CHECK: %[[INNER:.*]] = hw.array_get %[[ARR]][%sel] : !hw.array<33xarray<4xi1>>, i6
  // CHECK: %[[OUT:.*]] = hw.array_get %[[INNER]][%c0_i2] : !hw.array<4xi1>, i2
  // CHECK: hw.output %[[OUT]] : i1
}
