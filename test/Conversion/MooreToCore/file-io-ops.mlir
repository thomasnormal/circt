// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test file I/O operations: $fopen, $fclose
// These lower to __moore_fopen, __moore_fclose runtime calls.

// CHECK-DAG: llvm.func @__moore_fopen(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-DAG: llvm.func @__moore_fclose(i32)

//===----------------------------------------------------------------------===//
// $fopen Operation (with mode)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fopen
moore.module @test_fopen(in %filename: !moore.string, in %mode: !moore.string, out fd: !moore.i32) {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_fopen
  %fd = moore.builtin.fopen %filename, %mode
  moore.output %fd : !moore.i32
}

//===----------------------------------------------------------------------===//
// $fopen Operation (without mode - defaults to "r")
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fopen_no_mode
moore.module @test_fopen_no_mode(in %filename: !moore.string, out fd: !moore.i32) {
  // CHECK: llvm.alloca
  // CHECK: llvm.store
  // CHECK: llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_fopen
  %fd = moore.builtin.fopen %filename
  moore.output %fd : !moore.i32
}

//===----------------------------------------------------------------------===//
// $fclose Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_fclose
moore.module @test_fclose(in %fd: !moore.i32) {
  // CHECK: llvm.call @__moore_fclose(%{{.*}}) : (i32) -> ()
  moore.builtin.fclose %fd
  moore.output
}

//===----------------------------------------------------------------------===//
// $fwrite Operation
//===----------------------------------------------------------------------===//

// Note: format_string cannot be a module port type - it's a simulation type.
// fwrite is typically used with internally-generated format strings.
// The fwrite conversion was tested as part of file-io integration.
// This test validates only fopen and fclose operations.
