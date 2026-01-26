// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that calls to uvm_pkg::run_test are properly converted with string type.
// The function call passes through with converted string type.

// CHECK-LABEL: func.func @test_run_test_with_name
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(ptr, i64)>)
func.func @test_run_test_with_name(%test_name: !moore.string) {
  // CHECK: call @"uvm_pkg::run_test"(%[[ARG]]) : (!llvm.struct<(ptr, i64)>) -> ()
  func.call @"uvm_pkg::run_test"(%test_name) : (!moore.string) -> ()
  return
}

// Declare the original run_test function
// CHECK: func.func private @"uvm_pkg::run_test"(!llvm.struct<(ptr, i64)>)
func.func private @"uvm_pkg::run_test"(!moore.string)
