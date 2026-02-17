// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_array_max(!llvm.ptr, i64, i32) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_array_min(!llvm.ptr, i64, i32) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_unique(!llvm.ptr) -> !llvm.struct<(ptr, i64)>
// CHECK-DAG: llvm.func @__moore_queue_sort(!llvm.ptr, i64)
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
// CHECK-DAG: llvm.func @__moore_assoc_size(!llvm.ptr) -> i64

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
// CHECK: llvm.call @__moore_array_max({{.*}}) : (!llvm.ptr, i64, i32) -> !llvm.struct<(ptr, i64)>
func.func @test_queue_max() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %max = moore.queue.max %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0>
  return
}

// CHECK-LABEL: func @test_queue_min
// CHECK: llvm.call @__moore_array_min({{.*}}) : (!llvm.ptr, i64, i32) -> !llvm.struct<(ptr, i64)>
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
// CHECK: llvm.call @__moore_queue_sort({{.*}}) : (!llvm.ptr, i64) -> ()
func.func @test_queue_sort() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort %queue_ref : !moore.ref<queue<!moore.i32, 0>>
  return
}

//===----------------------------------------------------------------------===//
// Queue Sort With Operations (custom key expression)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_sort_with
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x i32
// CHECK: llvm.alloca {{.*}} x i64
// CHECK: scf.for
// CHECK:   llvm.store {{.*}} : i64, !llvm.ptr
// CHECK:   llvm.load {{.*}} : !llvm.ptr -> i32
// CHECK:   comb.mods
// CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.cmpi sgt
// CHECK:     scf.if
// CHECK: scf.for
// CHECK: scf.for
func.func @test_queue_sort_with() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %key = moore.mods %item, %ten : i32
    moore.queue.sort.key.yield %key : i32
  }
  return
}

// CHECK-LABEL: func @test_queue_rsort_with
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca {{.*}} x i32
// CHECK: llvm.alloca {{.*}} x i64
// CHECK: scf.for
// CHECK:   comb.mods
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.cmpi slt
// CHECK:     scf.if
// CHECK: scf.for
// CHECK: scf.for
func.func @test_queue_rsort_with() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.rsort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %key = moore.mods %item, %ten : i32
    moore.queue.sort.key.yield %key : i32
  }
  return
}

// Test queue.sort.with with absolute value key (negative to positive sort)
// CHECK-LABEL: func @test_queue_sort_with_abs
// CHECK: scf.for
// CHECK:   comb.icmp slt
// CHECK:   comb.sub
// CHECK:   comb.mux
// CHECK:   llvm.store
// CHECK: scf.for
func.func @test_queue_sort_with_abs() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %neg = moore.slt %item, %zero : i32 -> i1
    %negval = moore.neg %item : i32
    %absval = moore.conditional %neg : i1 -> i32 {
      moore.yield %negval : i32
    } {
      moore.yield %item : i32
    }
    moore.queue.sort.key.yield %absval : i32
  }
  return
}

//===----------------------------------------------------------------------===//
// Dynamic Array New Operation
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_dyn_array_new
// CHECK: [[SIZE:%.+]] = hw.constant 10 : i32
// CHECK: [[ELEMSIZE:%.+]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: [[TOTAL:%.+]] = llvm.mul [[SIZE]], [[ELEMSIZE]] : i32
// CHECK: llvm.call @__moore_dyn_array_new([[TOTAL]]) : (i32) -> !llvm.struct<(ptr, i64)>
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
// CHECK-DAG: [[ADDR:%.+]] = llvm.mlir.addressof @testAssoc : !llvm.ptr
// CHECK-DAG: [[KEY:%.+]] = hw.constant 42 : i8
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
// CHECK: [[QPTR:%.+]] = llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[ELEMALLOCA:%.+]] = llvm.alloca {{.*}} x i32
// CHECK: llvm.store {{.*}} : i32, !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @__moore_queue_push_back([[QPTR]], [[ELEMALLOCA]], [[SIZE]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
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

// CHECK-LABEL: func @test_queue_insert
// CHECK: [[QPTR:%.+]] = llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: llvm.call @__moore_queue_insert([[QPTR]], {{.*}}, {{.*}}, [[SIZE]]) : (!llvm.ptr, i32, !llvm.ptr, i64) -> ()
func.func @test_queue_insert() {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %index = moore.constant 1 : i32
  %elem = moore.constant 42 : i32
  moore.queue.insert %queue_ref, %index, %elem : !moore.ref<queue<!moore.i32, 0>>, !moore.i32, !moore.i32
  return
}

// CHECK-LABEL: func @test_queue_pop_back
// CHECK: [[QPTR:%.+]] = llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[SIZE:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_queue_pop_back([[QPTR]], [[SIZE]]) : (!llvm.ptr, i64) -> i64
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

// CHECK-LABEL: func @test_array_locator_find_string_eq
// CHECK: llvm.call @__moore_string_cmp
// CHECK: llvm.call @__moore_queue_push_back
func.func @test_array_locator_find_string_eq(%target: !moore.string) -> !moore.queue<!moore.string, 0> {
  %queue_ref = moore.get_global_variable @testStringQueue : !moore.ref<queue<!moore.string, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.string, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.string, 0> -> !moore.queue<!moore.string, 0> {
  ^bb0(%item: !moore.string):
    %cond = moore.string_cmp eq %item, %target : !moore.string -> !moore.i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.string, 0>
}

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
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
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
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
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
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
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
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
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
// CHECK: llvm.call @__moore_array_find_cmp({{.*}}) : (!llvm.ptr, i64, !llvm.ptr, i32, i32, i1) -> !llvm.struct<(ptr, i64)>
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

//===----------------------------------------------------------------------===//
// Queue Indexing Operations (dyn_extract)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_dyn_extract
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[QUEUE:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[PTR:%.+]] = llvm.extractvalue [[QUEUE]][0] : !llvm.struct<(ptr, i64)>
// CHECK: [[IDX:%.+]] = arith.extui {{.*}} : i32 to i64
// CHECK: [[ELEM_PTR:%.+]] = llvm.getelementptr [[PTR]][[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.load [[ELEM_PTR]] : !llvm.ptr -> i32
func.func @test_queue_dyn_extract() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %idx = moore.constant 2 : i32
  %elem = moore.dyn_extract %queue from %idx : !moore.queue<!moore.i32, 0>, !moore.i32 -> !moore.i32
  return %elem : !moore.i32
}

// CHECK-LABEL: func @test_queue_dyn_extract_ref
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[QUEUE:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[PTR:%.+]] = llvm.extractvalue [[QUEUE]][0] : !llvm.struct<(ptr, i64)>
// CHECK: [[IDX:%.+]] = arith.extui {{.*}} : i32 to i64
// CHECK: [[ELEM_PTR:%.+]] = llvm.getelementptr [[PTR]][[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
func.func @test_queue_dyn_extract_ref() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %idx = moore.constant 2 : i32
  %elem_ref = moore.dyn_extract_ref %queue_ref from %idx : !moore.ref<queue<!moore.i32, 0>>, !moore.i32 -> !moore.ref<i32>
  %val = moore.read %elem_ref : <i32>
  return %val : !moore.i32
}

// CHECK-LABEL: func @test_dyn_array_dyn_extract
// CHECK: llvm.mlir.addressof @testDynArray : !llvm.ptr
// CHECK: [[ARR:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[PTR:%.+]] = llvm.extractvalue [[ARR]][0] : !llvm.struct<(ptr, i64)>
// CHECK: [[IDX:%.+]] = arith.extui {{.*}} : i32 to i64
// CHECK: [[ELEM_PTR:%.+]] = llvm.getelementptr [[PTR]][[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.load [[ELEM_PTR]] : !llvm.ptr -> i32
func.func @test_dyn_array_dyn_extract() -> !moore.i32 {
  %arr_ref = moore.get_global_variable @testDynArray : !moore.ref<open_uarray<!moore.i32>>
  %arr = moore.read %arr_ref : <open_uarray<!moore.i32>>
  %idx = moore.constant 3 : i32
  %elem = moore.dyn_extract %arr from %idx : !moore.open_uarray<!moore.i32>, !moore.i32 -> !moore.i32
  return %elem : !moore.i32
}

// CHECK-LABEL: func @test_dyn_array_dyn_extract_ref
// CHECK: llvm.mlir.addressof @testDynArray : !llvm.ptr
// CHECK: [[ARR:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[PTR:%.+]] = llvm.extractvalue [[ARR]][0] : !llvm.struct<(ptr, i64)>
// CHECK: [[IDX:%.+]] = arith.extui {{.*}} : i32 to i64
// CHECK: [[ELEM_PTR:%.+]] = llvm.getelementptr [[PTR]][[[IDX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
func.func @test_dyn_array_dyn_extract_ref() -> !moore.i32 {
  %arr_ref = moore.get_global_variable @testDynArray : !moore.ref<open_uarray<!moore.i32>>
  %idx = moore.constant 3 : i32
  %elem_ref = moore.dyn_extract_ref %arr_ref from %idx : !moore.ref<open_uarray<!moore.i32>>, !moore.i32 -> !moore.ref<i32>
  %val = moore.read %elem_ref : <i32>
  return %val : !moore.i32
}

//===----------------------------------------------------------------------===//
// Array Size Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_queue_size
// CHECK: llvm.mlir.addressof @testQueue : !llvm.ptr
// CHECK: [[QUEUE:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[LEN:%.+]] = llvm.extractvalue [[QUEUE]][1] : !llvm.struct<(ptr, i64)>
// CHECK: [[RESULT:%.+]] = arith.trunci [[LEN]] : i64 to i32
func.func @test_queue_size() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %size = moore.array.size %queue : !moore.queue<!moore.i32, 0>
  return %size : !moore.i32
}

// CHECK-LABEL: func @test_dyn_array_size
// CHECK: llvm.mlir.addressof @testDynArray : !llvm.ptr
// CHECK: [[ARR:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: [[LEN:%.+]] = llvm.extractvalue [[ARR]][1] : !llvm.struct<(ptr, i64)>
// CHECK: [[RESULT:%.+]] = arith.trunci [[LEN]] : i64 to i32
func.func @test_dyn_array_size() -> !moore.i32 {
  %arr_ref = moore.get_global_variable @testDynArray : !moore.ref<open_uarray<!moore.i32>>
  %arr = moore.read %arr_ref : <open_uarray<!moore.i32>>
  %size = moore.array.size %arr : !moore.open_uarray<!moore.i32>
  return %size : !moore.i32
}

// CHECK-LABEL: func @test_assoc_array_size
// CHECK: llvm.mlir.addressof @testAssoc : !llvm.ptr
// CHECK: [[ASSOC:%.+]] = llvm.load {{.*}} : !llvm.ptr -> !llvm.ptr
// CHECK: [[LEN:%.+]] = llvm.call @__moore_assoc_size([[ASSOC]]) : (!llvm.ptr) -> i64
// CHECK: [[RESULT:%.+]] = arith.trunci [[LEN]] : i64 to i32
func.func @test_assoc_array_size() -> !moore.i32 {
  %assoc_ref = moore.get_global_variable @testAssoc : !moore.ref<assoc_array<!moore.i32, !moore.i8>>
  %assoc = moore.read %assoc_ref : <assoc_array<!moore.i32, !moore.i8>>
  %size = moore.array.size %assoc : !moore.assoc_array<!moore.i32, !moore.i8>
  return %size : !moore.i32
}

//===----------------------------------------------------------------------===//
// Array Locator with Complex Predicates (Inline Loop Lowering)
//===----------------------------------------------------------------------===//

// Test AND predicate - uses inline loop approach since simple pattern doesn't
// match compound predicates.
// CHECK-LABEL: func @test_array_locator_and_predicate
// CHECK: scf.for
// CHECK: comb.and
// CHECK: scf.if
// CHECK: llvm.call @__moore_queue_push_back
func.func @test_array_locator_and_predicate() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %five = moore.constant 5 : i32
    %ten = moore.constant 10 : i32
    %cond1 = moore.sgt %item, %five : i32 -> i1
    %cond2 = moore.slt %item, %ten : i32 -> i1
    %cond = moore.and %cond1, %cond2 : i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}

// Test OR predicate - uses inline loop approach.
// CHECK-LABEL: func @test_array_locator_or_predicate
// CHECK: scf.for
// CHECK: comb.or
// CHECK: scf.if
// CHECK: llvm.call @__moore_queue_push_back
func.func @test_array_locator_or_predicate() -> !moore.queue<!moore.i32, 0> {
  %queue_ref = moore.get_global_variable @testQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %result = moore.array.locator all, elements %queue : !moore.queue<!moore.i32, 0> -> !moore.queue<!moore.i32, 0> {
  ^bb0(%item: !moore.i32):
    %zero = moore.constant 0 : i32
    %hundred = moore.constant 100 : i32
    %cond1 = moore.eq %item, %zero : i32 -> i1
    %cond2 = moore.eq %item, %hundred : i32 -> i1
    %cond = moore.or %cond1, %cond2 : i1
    moore.array.locator.yield %cond : i1
  }
  return %result : !moore.queue<!moore.i32, 0>
}
