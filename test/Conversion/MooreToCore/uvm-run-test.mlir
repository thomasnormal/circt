// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that calls to uvm_pkg::run_test are converted to __uvm_run_test runtime calls.

// CHECK: llvm.func @__uvm_run_test(!llvm.ptr, i64)

// CHECK-LABEL: func.func @test_run_test_with_name
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, i64)>)
func.func @test_run_test_with_name(%test_name: !moore.string) {
  // CHECK: %[[PTR:.*]] = llvm.extractvalue %[[ARG]][0] : !llvm.struct<(ptr, i64)>
  // CHECK: %[[LEN:.*]] = llvm.extractvalue %[[ARG]][1] : !llvm.struct<(ptr, i64)>
  // CHECK: llvm.call @__uvm_run_test(%[[PTR]], %[[LEN]]) : (!llvm.ptr, i64) -> ()
  func.call @"uvm_pkg::run_test"(%test_name) : (!moore.string) -> ()
  return
}

// Declare the original run_test function that we intercept
func.func private @"uvm_pkg::run_test"(!moore.string)
