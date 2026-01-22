// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that moore::TimeType converts to i64 and that sbv_to_packed works
// with time types. This test was added to fix the error:
// 'hw.bitcast' op result #0 must be Type wherein the bitwidth in hardware is
// known, but got '!llhd.time'

// CHECK-LABEL: func @TimeTypeArgAndReturn
// CHECK-SAME: (%arg0: i64) -> i64
func.func @TimeTypeArgAndReturn(%arg0: !moore.time) -> !moore.time {
  // CHECK-NEXT: return %arg0 : i64
  return %arg0 : !moore.time
}

// CHECK-LABEL: func @SBVToPackedTime
// CHECK-SAME: (%arg0: !hw.struct<value: i64, unknown: i64>) -> i64
func.func @SBVToPackedTime(%arg0: !moore.l64) -> !moore.time {
  // 4-state l64 (struct) to time (i64): extract value component
  // CHECK: [[VALUE:%.+]] = hw.struct_extract %arg0["value"] : !hw.struct<value: i64, unknown: i64>
  // CHECK-NEXT: return [[VALUE]] : i64
  %0 = moore.sbv_to_packed %arg0 : time
  return %0 : !moore.time
}

// CHECK-LABEL: func @PackedToSBVTime
// CHECK-SAME: (%arg0: i64) -> !hw.struct<value: i64, unknown: i64>
func.func @PackedToSBVTime(%arg0: !moore.time) -> !moore.l64 {
  // time (i64) to 4-state l64 (struct): wrap in struct with unknown=0
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i64
  // CHECK: [[RESULT:%.+]] = hw.struct_create (%arg0, [[ZERO]]) : !hw.struct<value: i64, unknown: i64>
  // CHECK-NEXT: return [[RESULT]] : !hw.struct<value: i64, unknown: i64>
  %0 = moore.packed_to_sbv %arg0 : time
  return %0 : !moore.l64
}

// CHECK-LABEL: func @TimeToLogic
// CHECK-SAME: (%arg0: i64) -> !hw.struct<value: i64, unknown: i64>
func.func @TimeToLogic(%arg0: !moore.time) -> !moore.l64 {
  // Since TimeType converts to i64, time_to_logic wraps it in 4-state struct
  // CHECK: [[ZERO:%.+]] = hw.constant 0 : i64
  // CHECK: [[RESULT:%.+]] = hw.struct_create (%arg0, [[ZERO]]) : !hw.struct<value: i64, unknown: i64>
  // CHECK-NEXT: return [[RESULT]] : !hw.struct<value: i64, unknown: i64>
  %0 = moore.time_to_logic %arg0
  return %0 : !moore.l64
}

// CHECK-LABEL: func @LogicToTime
// CHECK-SAME: (%arg0: !hw.struct<value: i64, unknown: i64>) -> i64
func.func @LogicToTime(%arg0: !moore.l64) -> !moore.time {
  // Since TimeType converts to i64, logic_to_time extracts value from 4-state
  // CHECK: [[VALUE:%.+]] = hw.struct_extract %arg0["value"] : !hw.struct<value: i64, unknown: i64>
  // CHECK-NEXT: return [[VALUE]] : i64
  %0 = moore.logic_to_time %arg0
  return %0 : !moore.time
}
