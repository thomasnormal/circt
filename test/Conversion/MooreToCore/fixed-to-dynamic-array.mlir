// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @malloc(i64) -> !llvm.ptr

//===----------------------------------------------------------------------===//
// Fixed Array to Open Array Conversion
// Tests conversion from fixed-size unpacked array (uarray<N x T>) to
// dynamic array (open_uarray<T>). This is a legal SystemVerilog conversion
// used in I2S AVIP and similar UVM verification environments.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_fixed_to_dyn_array_i8
// CHECK-SAME: (%[[ARG0:.*]]: !hw.array<4xi8>)
// CHECK: llvm.call @malloc
// Extract elements from fixed array using hw.array_get
// CHECK: hw.array_get %[[ARG0]]
// CHECK: hw.array_get %[[ARG0]]
// CHECK: hw.array_get %[[ARG0]]
// CHECK: hw.array_get %[[ARG0]]
// Build result struct with pointer and size
// CHECK: llvm.insertvalue {{.*}}[0]
// CHECK: llvm.insertvalue {{.*}}[1]
// CHECK: return
func.func @test_fixed_to_dyn_array_i8(%arr: !moore.uarray<4 x i8>) -> !moore.open_uarray<!moore.i8> {
  %dyn = moore.conversion %arr : !moore.uarray<4 x i8> -> !moore.open_uarray<!moore.i8>
  return %dyn : !moore.open_uarray<!moore.i8>
}

// CHECK-LABEL: func @test_fixed_to_dyn_array_i32
// CHECK-SAME: (%[[ARG1:.*]]: !hw.array<4xi32>)
// CHECK: llvm.call @malloc
// CHECK: hw.array_get %[[ARG1]]
// CHECK: hw.array_get %[[ARG1]]
// CHECK: llvm.insertvalue {{.*}}[0]
// CHECK: llvm.insertvalue {{.*}}[1]
func.func @test_fixed_to_dyn_array_i32(%arr: !moore.uarray<4 x i32>) -> !moore.open_uarray<!moore.i32> {
  %dyn = moore.conversion %arr : !moore.uarray<4 x i32> -> !moore.open_uarray<!moore.i32>
  return %dyn : !moore.open_uarray<!moore.i32>
}

// Test with different array sizes
// CHECK-LABEL: func @test_fixed_to_dyn_array_size_8
// CHECK-SAME: (%[[ARG2:.*]]: !hw.array<8xi8>)
// CHECK: llvm.call @malloc
// CHECK-COUNT-8: hw.array_get %[[ARG2]]
// CHECK: llvm.insertvalue {{.*}}[1]
func.func @test_fixed_to_dyn_array_size_8(%arr: !moore.uarray<8 x i8>) -> !moore.open_uarray<!moore.i8> {
  %dyn = moore.conversion %arr : !moore.uarray<8 x i8> -> !moore.open_uarray<!moore.i8>
  return %dyn : !moore.open_uarray<!moore.i8>
}

// Test with single element array
// CHECK-LABEL: func @test_fixed_to_dyn_array_size_1
// CHECK-SAME: (%[[ARG3:.*]]: !hw.array<1xi8>)
// CHECK: llvm.call @malloc
// CHECK: hw.array_get %[[ARG3]]
// CHECK: llvm.insertvalue {{.*}}[1]
func.func @test_fixed_to_dyn_array_size_1(%arr: !moore.uarray<1 x i8>) -> !moore.open_uarray<!moore.i8> {
  %dyn = moore.conversion %arr : !moore.uarray<1 x i8> -> !moore.open_uarray<!moore.i8>
  return %dyn : !moore.open_uarray<!moore.i8>
}
