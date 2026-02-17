// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_assoc_create(i32, i32) -> !llvm.ptr
// CHECK-DAG: llvm.func @__moore_assoc_first(!llvm.ptr, !llvm.ptr) -> i1
// CHECK-DAG: llvm.func @__moore_assoc_next(!llvm.ptr, !llvm.ptr) -> i1
// CHECK-DAG: llvm.func @__moore_assoc_get_ref(!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr

//===----------------------------------------------------------------------===//
// String-keyed Associative Array Creation and Iteration
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_string_assoc
hw.module @test_string_assoc() {
  // CHECK: [[KEY_SIZE:%.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[VAL_SIZE:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: [[ASSOC:%.+]] = llvm.call @__moore_assoc_create([[KEY_SIZE]], [[VAL_SIZE]]) : (i32, i32) -> !llvm.ptr
  %values = moore.variable : <assoc_array<i32, string>>

  // CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.store
  %s = moore.variable : <string>

  moore.procedure initial {
    // Test first() method for string-keyed assoc array
    // CHECK: llvm.call @__moore_assoc_first({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i1
    %first_result = moore.assoc.first %values, %s : <assoc_array<i32, string>>, <string>

    // Test next() method for string-keyed assoc array
    // CHECK: llvm.call @__moore_assoc_next({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i1
    %next_result = moore.assoc.next %values, %s : <assoc_array<i32, string>>, <string>

    moore.return
  }
  hw.output
}

//===----------------------------------------------------------------------===//
// String-keyed Associative Array Element Access
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_string_assoc_access
hw.module @test_string_assoc_access() {
  %values = moore.variable : <assoc_array<i32, string>>
  %c1 = moore.constant 1 : i32
  %key_int = moore.constant 104 : i64  // "h" as integer
  %key_str = moore.int_to_string %key_int : i64

  moore.procedure initial {
    // Test element access via dyn_extract_ref
    // CHECK: llvm.call @__moore_assoc_get_ref({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> !llvm.ptr
    // CHECK: llvm.store {{.*}} : i32, !llvm.ptr
    %elem_ref = moore.dyn_extract_ref %values from %key_str : <assoc_array<i32, string>>, string -> <i32>
    moore.blocking_assign %elem_ref, %c1 : i32
    moore.return
  }
  hw.output
}

//===----------------------------------------------------------------------===//
// Integer-keyed Associative Array (for comparison)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_int_assoc
hw.module @test_int_assoc() {
  // CHECK: [[KSIZE:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: [[VSIZE:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: llvm.call @__moore_assoc_create([[KSIZE]], [[VSIZE]]) : (i32, i32) -> !llvm.ptr
  %values = moore.variable : <assoc_array<i32, i32>>

  moore.procedure initial {
    %key = moore.variable : <i32>
    // CHECK: llvm.call @__moore_assoc_first({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i1
    %first_result = moore.assoc.first %values, %key : <assoc_array<i32, i32>>, <i32>
    // CHECK: llvm.call @__moore_assoc_next({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i1
    %next_result = moore.assoc.next %values, %key : <assoc_array<i32, i32>>, <i32>
    moore.return
  }
  hw.output
}
