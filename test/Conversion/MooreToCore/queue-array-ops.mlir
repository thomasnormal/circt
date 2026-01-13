// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test global variables for container types
moore.global_variable @testQueue : !moore.queue<!moore.i32, 0>
moore.global_variable @testDynArray : !moore.open_uarray<!moore.i32>
moore.global_variable @testAssoc : !moore.assoc_array<!moore.i32, !moore.i8>

//===----------------------------------------------------------------------===//
// Queue Max/Min Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_max
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: llvm.call @__moore_queue_max({{.*}}) : (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_max() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %max = moore.queue.max %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0>
  return
}

// CHECK-LABEL: func @test_queue_min
// CHECK: llvm.call @__moore_queue_min({{.*}}) : (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_min() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %min = moore.queue.min %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0>
  return
}

// CHECK-LABEL: func @test_queue_unique
// CHECK: llvm.call @__moore_queue_unique({{.*}}) : (!llvm.ptr) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_unique() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %unique = moore.queue.unique %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0>
  return
}

// CHECK-LABEL: func @test_queue_sort
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: llvm.call @__moore_queue_sort({{.*}}) : (!llvm.ptr) -> ()
func.func @test_queue_sort() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort %queue_ref : !moore.ref<queue<!moore.i32, 0>>
  return
}

//===----------------------------------------------------------------------===//
// Dynamic Array New Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_dyn_array_new
// CHECK: [[SIZE:%.+]] = hw.constant 10 : i32
// CHECK: llvm.call @__moore_dyn_array_new([[SIZE]]) : (i32) -> !llvm.struct<(ptr, i64)>
func.func @test_dyn_array_new() {
  %c10 = moore.constant 10 : i32
  %arr = moore.dyn_array.new %c10 : !moore.i32 -> !moore.open_uarray<!moore.i32>
  return
}

//===----------------------------------------------------------------------===//
// Associative Array Delete Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_assoc_delete
// CHECK: [[ADDR:%.+]] = llvm.mlir.addressof @testAssoc : !llvm.ptr
// CHECK: llvm.call @__moore_assoc_delete([[ADDR]]) : (!llvm.ptr) -> ()
func.func @test_assoc_delete() {
  %assoc_ref = moore.get_global_variable @testAssoc : !moore.ref<assoc_array<!moore.i32, !moore.i8>>
  moore.assoc.delete %assoc_ref : !moore.ref<assoc_array<!moore.i32, !moore.i8>>
  return
}

// CHECK-LABEL: func @test_assoc_delete_key
// CHECK: [[ADDR:%.+]] = llvm.mlir.addressof @testAssoc : !llvm.ptr
// CHECK: [[KEY:%.+]] = hw.constant 42 : i8
// CHECK: llvm.alloca {{.*}} x i8
// CHECK: llvm.store [[KEY]], {{.*}} : i8, !llvm.ptr
// CHECK: llvm.call @__moore_assoc_delete_key([[ADDR]], {{.*}}) : (!llvm.ptr, !llvm.ptr) -> ()
func.func @test_assoc_delete_key() {
  %assoc_ref = moore.get_global_variable @testAssoc : !moore.ref<assoc_array<!moore.i32, !moore.i8>>
  %key = moore.constant 42 : i8
  moore.assoc.delete_key %assoc_ref, %key : !moore.ref<assoc_array<!moore.i32, !moore.i8>>, !moore.i8
  return
}

//===----------------------------------------------------------------------===//
// Stream Concatenation Operations
//===----------------------------------------------------------------------===//

moore.global_variable @testStringQueue : !moore.queue<!moore.string, 0>

// CHECK-LABEL: func @test_stream_concat_string_queue
// CHECK: llvm.mlir.addressof @testStringQueue : !llvm.ptr
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: llvm.call @__moore_stream_concat_strings({{.*}}, [[FALSE]]) : (!llvm.ptr, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_stream_concat_string_queue() -> !moore.string {
  %queue_ref = moore.get_global_variable @testStringQueue : !moore.ref<queue<!moore.string, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.string, 0>>
  %result = moore.stream_concat %queue : !moore.queue<!moore.string, 0> -> !moore.string
  return %result : !moore.string
}

// CHECK-LABEL: func @test_stream_concat_string_queue_rtl
// CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1
// CHECK: llvm.call @__moore_stream_concat_strings({{.*}}, [[TRUE]]) : (!llvm.ptr, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_stream_concat_string_queue_rtl() -> !moore.string {
  %queue_ref = moore.get_global_variable @testStringQueue : !moore.ref<queue<!moore.string, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.string, 0>>
  %result = moore.stream_concat %queue {isRightToLeft = true} : !moore.queue<!moore.string, 0> -> !moore.string
  return %result : !moore.string
}

// CHECK-LABEL: func @test_stream_concat_int_queue
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: [[WIDTH:%.+]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_stream_concat_bits({{.*}}, [[WIDTH]], [[FALSE]]) : (!llvm.ptr, i32, i1) -> i64
// CHECK: arith.trunci [[RESULT]] : i64 to i32
func.func @test_stream_concat_int_queue() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.stream_concat %queue : !moore.queue<!moore.i32, 0> -> !moore.i32
  return %result : !moore.i32
}

// CHECK-LABEL: func @test_stream_concat_int_queue_rtl
// CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1
// CHECK: [[WIDTH:%.+]] = llvm.mlir.constant(32 : i32) : i32
// CHECK: llvm.call @__moore_stream_concat_bits({{.*}}, [[WIDTH]], [[TRUE]]) : (!llvm.ptr, i32, i1) -> i64
func.func @test_stream_concat_int_queue_rtl() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.stream_concat %queue {isRightToLeft = true} : !moore.queue<!moore.i32, 0> -> !moore.i32
  return %result : !moore.i32
}

//===----------------------------------------------------------------------===//
// Queue Push/Pop Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_push_back
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: llvm.alloca {{.*}} x i32
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @__moore_queue_push_back({{.*}}, {{.*}}, [[SIZE]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
func.func @test_queue_push_back() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.constant 42 : i32
  moore.queue.push_back %queue_ref, %elem : !moore.ref<queue<!moore.i32, 0>>, !moore.i32
  return
}

// CHECK-LABEL: func @test_queue_push_front
// CHECK: llvm.call @__moore_queue_push_front({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
func.func @test_queue_push_front() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.constant 42 : i32
  moore.queue.push_front %queue_ref, %elem : !moore.ref<queue<!moore.i32, 0>>, !moore.i32
  return
}

// CHECK-LABEL: func @test_queue_pop_back
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_queue_pop_back({{.*}}, [[SIZE]]) : (!llvm.ptr, i64) -> i64
// CHECK: arith.trunci [[RESULT]] : i64 to i32
func.func @test_queue_pop_back() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.queue.pop_back %queue_ref : !moore.ref<queue<!moore.i32, 0>> -> !moore.i32
  return %elem : !moore.i32
}

// CHECK-LABEL: func @test_queue_pop_front
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_queue_pop_front({{.*}}, {{.*}}) : (!llvm.ptr, i64) -> i64
// CHECK: arith.trunci [[RESULT]] : i64 to i32
func.func @test_queue_pop_front() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.queue.pop_front %queue_ref : !moore.ref<queue<!moore.i32, 0>> -> !moore.i32
  return %elem : !moore.i32
}

//===----------------------------------------------------------------------===//
// Array Locator Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_array_locator_find_all_eq
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK: llvm.alloca {{.*}} x i32
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: [[MODE:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[INDICES:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, [[SIZE]], {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_all_eq() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %five = moore.constant 5 : i32
    %cond = moore.eq %item, %five : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_first_index_eq
// CHECK: [[MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_first_index_eq() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator first, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %cond = moore.eq %item, %zero : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_last_eq
// CHECK: [[MODE:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[INDICES:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: llvm.call @__moore_array_find_eq({{.*}}, {{.*}}, {{.*}}, [[MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_last_eq() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator last, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %cond = moore.eq %item, %ten : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

//===----------------------------------------------------------------------===//
// Array Locator Operations with Comparison Predicates
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_array_locator_find_ne
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x i32
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: [[LOC_MODE:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[INDICES:%.+]] = llvm.mlir.constant(false) : i1
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, [[SIZE]], {{.*}}, [[CMP_MODE]], [[LOC_MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_ne() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %cond = moore.ne %item, %zero : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_sgt
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: [[LOC_MODE:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, [[CMP_MODE]], [[LOC_MODE]], {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_sgt() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %five = moore.constant 5 : i32
    %cond = moore.sgt %item, %five : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_sge
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: [[LOC_MODE:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, [[CMP_MODE]], [[LOC_MODE]], {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_sge() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator first, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %cond = moore.sge %item, %ten : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_slt
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: [[LOC_MODE:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, [[CMP_MODE]], [[LOC_MODE]], {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_slt() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator last, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %cond = moore.slt %item, %zero : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-LABEL: func @test_array_locator_find_sle
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: [[LOC_MODE:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: [[INDICES:%.+]] = llvm.mlir.constant(true) : i1
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, [[CMP_MODE]], [[LOC_MODE]], [[INDICES]]) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_sle() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, indices %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %hundred = moore.constant 100 : i32
    %cond = moore.sle %item, %hundred : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// Test operand swapping: when constant is on LHS, comparison direction should be swapped
// CHECK-LABEL: func @test_array_locator_find_sgt_swapped
// CHECK: [[CMP_MODE:%.+]] = llvm.mlir.constant(4 : i32) : i32
// Swapped: sgt with const on LHS becomes slt (mode 4)
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}, {{.*}}, {{.*}}, [[CMP_MODE]], {{.*}}, {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
func.func @test_array_locator_find_sgt_swapped() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %five = moore.constant 5 : i32
    // const > item  is equivalent to  item < const
    %cond = moore.sgt %five, %item : i32 -> i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// CHECK-DAG: llvm.func @__moore_queue_max(!llvm.ptr) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_min(!llvm.ptr) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_unique(!llvm.ptr) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_sort(!llvm.ptr)
// CHECK-DAG: llvm.func @__moore_queue_push_back(!llvm.ptr, !llvm.ptr, i64)
// CHECK-DAG: llvm.func @__moore_queue_push_front(!llvm.ptr, !llvm.ptr, i64)
// CHECK-DAG: llvm.func @__moore_queue_pop_back(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_queue_pop_front(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_stream_concat_strings(!llvm.ptr, i1) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_stream_concat_bits(!llvm.ptr, i32, i1) -> i64
// CHECK-DAG: llvm.func @__moore_dyn_array_new(i32) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_assoc_delete(!llvm.ptr)
// CHECK-DAG: llvm.func @__moore_assoc_delete_key(!llvm.ptr, !llvm.ptr)
// CHECK-DAG: llvm.func @__moore_array_find_eq(!llvm.ptr, i64, !llvm.ptr, i32, i1) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_array_find_cmp(!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
