// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test lowering of covergroups with illegal_bins and ignore_bins.
// This test verifies that the illegal/ignore bins are registered with the runtime.

// CHECK-DAG: llvm.mlir.global internal @__cg_handle_TestCGWithBins
// CHECK-DAG: llvm.func @__moore_covergroup_create(!llvm.ptr, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @__moore_coverpoint_init(!llvm.ptr, i32, !llvm.ptr)
// CHECK-DAG: llvm.func @__moore_coverpoint_add_illegal_bin(!llvm.ptr, i32, !llvm.ptr, i64, i64)
// CHECK-DAG: llvm.func @__moore_coverpoint_add_ignore_bin(!llvm.ptr, i32, !llvm.ptr, i64, i64)

// Covergroup with illegal_bins and ignore_bins
moore.covergroup.decl @TestCGWithBins {
  moore.coverpoint.decl @data_cp : !moore.l4 {
    moore.coverbin.decl @valid kind<bins> values [1, 2, 3, 4]
    moore.coverbin.decl @reserved kind<illegal_bins> values [15]
    moore.coverbin.decl @zero kind<ignore_bins> values [0]
  }
}

// CHECK-LABEL: llvm.func @__cg_init_TestCGWithBins()
// CHECK: llvm.call @__moore_covergroup_create
// CHECK: llvm.call @__moore_coverpoint_init
// CHECK: llvm.call @__moore_coverpoint_add_illegal_bin
// CHECK-SAME: i64 15, i64 15
// CHECK: llvm.call @__moore_coverpoint_add_ignore_bin
// CHECK-SAME: i64 0, i64 0

// Test with range values in illegal_bins
moore.covergroup.decl @TestCGWithRanges {
  moore.coverpoint.decl @addr_cp : !moore.l8 {
    moore.coverbin.decl @bad_range kind<illegal_bins> values [[200, 255]]
    moore.coverbin.decl @skip_low kind<ignore_bins> values [[0, 15]]
  }
}

// CHECK-LABEL: llvm.func @__cg_init_TestCGWithRanges()
// CHECK: llvm.call @__moore_coverpoint_add_illegal_bin
// CHECK-SAME: i64 200, i64 255
// CHECK: llvm.call @__moore_coverpoint_add_ignore_bin
// CHECK-SAME: i64 0, i64 15
