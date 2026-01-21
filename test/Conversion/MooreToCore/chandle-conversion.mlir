// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that chandle to integer conversions are correctly lowered to llvm.ptrtoint
// This is used in UVM for DPI-C handle conversions

// CHECK-LABEL: func.func @test_chandle_to_int
// CHECK-SAME: (%arg0: !llvm.ptr)
// CHECK: %[[RESULT:.*]] = llvm.ptrtoint %arg0 : !llvm.ptr to i64
// CHECK: return %[[RESULT]] : i64
func.func @test_chandle_to_int(%handle: !moore.chandle) -> !moore.i64 {
  %result = moore.conversion %handle : !moore.chandle -> !moore.i64
  return %result : !moore.i64
}

// CHECK-LABEL: func.func @test_int_to_chandle
// CHECK-SAME: (%arg0: i64)
// CHECK: %[[RESULT:.*]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
// CHECK: return %[[RESULT]] : !llvm.ptr
func.func @test_int_to_chandle(%value: !moore.i64) -> !moore.chandle {
  %result = moore.conversion %value : !moore.i64 -> !moore.chandle
  return %result : !moore.chandle
}

// CHECK-LABEL: func.func @test_chandle_to_int_width
// CHECK-SAME: (%arg0: !llvm.ptr)
// CHECK: %[[RESULT:.*]] = llvm.ptrtoint %arg0 : !llvm.ptr to i32
// CHECK: return %[[RESULT]] : i32
func.func @test_chandle_to_int_width(%handle: !moore.chandle) -> !moore.i32 {
  %result = moore.conversion %handle : !moore.chandle -> !moore.i32
  return %result : !moore.i32
}
