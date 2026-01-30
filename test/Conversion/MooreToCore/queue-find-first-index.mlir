// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test queue find_first_index() method lowering
// IEEE 1800-2017 Section 7.12.2 "Array locator methods"
// This tests the find_first_index() method which returns a queue of indices
// where the predicate evaluates to true, stopping after the first match.

// CHECK-DAG: llvm.func @__moore_array_find_eq(!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_array_find_cmp(!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)

// Global test queue
moore.global_variable @testQueue : !moore.queue<!moore.i32, 0>

//===----------------------------------------------------------------------===//
// Queue find_first_index with equality predicate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_first_index_eq
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_first_index_eq() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_first_index with equality comparison: x == 42
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %target = moore.constant 42 : i32
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_first_index with comparison predicate (greater than)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_first_index_sgt
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_first_index_sgt() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_first_index with greater-than comparison: x > 100
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %threshold = moore.constant 100 : i32
    %cond = moore.sgt %item, %threshold : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_first_index with not-equal predicate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_first_index_ne
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_first_index_ne() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_first_index with not-equal comparison: x != 0
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %cond = moore.ne %item, %zero : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_first_index with less-than-or-equal predicate
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_first_index_sle
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_first_index_sle() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_first_index with less-than-or-equal comparison: x <= 50
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %fifty = moore.constant 50 : i32
    %cond = moore.sle %item, %fifty : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_first_index with complex predicate (inline loop path)
//===----------------------------------------------------------------------===//

// Complex predicates (AND/OR) use the inline loop lowering approach
// CHECK-LABEL: func @test_queue_find_first_index_complex
// CHECK: scf.for
// CHECK: comb.and
// CHECK: scf.if
// CHECK: llvm.call @__moore_queue_push_back
func.func @test_queue_find_first_index_complex() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_first_index with compound predicate: x > 10 && x < 100
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %hundred = moore.constant 100 : i32
    %gt10 = moore.sgt %item, %ten : i32 -> i1
    %lt100 = moore.slt %item, %hundred : i32 -> i1
    %cond = moore.and %gt10, %lt100 : i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_index (all matching indices) - contrast with find_first_index
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_index_all
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_index_all() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_index returns ALL matching indices (mode = all)
  %result = moore.array.locator all, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %target = moore.constant 5 : i32
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Queue find_last_index - contrast with find_first_index
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_find_last_index
// CHECK-DAG: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: [[MODE:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_find_last_index() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>

  // find_last_index returns the last matching index (mode = last)
  %result = moore.array.locator last, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %target = moore.constant 99 : i32
    %cond = moore.eq %item, %target : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}
