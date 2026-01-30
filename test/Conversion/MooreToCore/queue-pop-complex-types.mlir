// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_queue_pop_back(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_queue_pop_front(!llvm.ptr, i64) -> i64
// CHECK-DAG: llvm.func @__moore_queue_pop_back_ptr(!llvm.ptr, !llvm.ptr, i64)
// CHECK-DAG: llvm.func @__moore_queue_pop_front_ptr(!llvm.ptr, !llvm.ptr, i64)

//===----------------------------------------------------------------------===//
// Test queue pop with class types (pointers)
//===----------------------------------------------------------------------===//

moore.class.classdecl @TestClass {
  moore.class.propertydecl @value : !moore.i32
}

moore.global_variable @classQueue : !moore.queue<!moore.class<@TestClass>, 0>

// CHECK-LABEL: func @test_queue_pop_back_class
// CHECK: [[QPTR:%.+]] = llvm.mlir.addressof @classQueue : !llvm.ptr
// CHECK: [[RESULTALLOCA:%.+]] = llvm.alloca {{.*}} x !llvm.ptr
// CHECK: llvm.call @__moore_queue_pop_back_ptr([[QPTR]], [[RESULTALLOCA]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: llvm.load [[RESULTALLOCA]] : !llvm.ptr -> !llvm.ptr
// CHECK: return
func.func @test_queue_pop_back_class() -> !moore.class<@TestClass> {
  %queue_ref = moore.get_global_variable @classQueue : !moore.ref<queue<!moore.class<@TestClass>, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.class<@TestClass>, 0>>
  %elem = moore.queue.pop_back %queue_ref : !moore.ref<queue<!moore.class<@TestClass>, 0>> -> !moore.class<@TestClass>
  return %elem : !moore.class<@TestClass>
}

// CHECK-LABEL: func @test_queue_pop_front_class
// CHECK: llvm.call @__moore_queue_pop_front_ptr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.ptr
func.func @test_queue_pop_front_class() -> !moore.class<@TestClass> {
  %queue_ref = moore.get_global_variable @classQueue : !moore.ref<queue<!moore.class<@TestClass>, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.class<@TestClass>, 0>>
  %elem = moore.queue.pop_front %queue_ref : !moore.ref<queue<!moore.class<@TestClass>, 0>> -> !moore.class<@TestClass>
  return %elem : !moore.class<@TestClass>
}

//===----------------------------------------------------------------------===//
// Test queue pop with string types (struct types)
//===----------------------------------------------------------------------===//

moore.global_variable @stringQueue : !moore.queue<!moore.string, 0>

// CHECK-LABEL: func @test_queue_pop_back_string
// CHECK: [[QPTR:%.+]] = llvm.mlir.addressof @stringQueue : !llvm.ptr
// CHECK: [[RESULTALLOCA:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, i64)>
// CHECK: llvm.call @__moore_queue_pop_back_ptr([[QPTR]], [[RESULTALLOCA]], {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: llvm.load [[RESULTALLOCA]] : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: return
func.func @test_queue_pop_back_string() -> !moore.string {
  %queue_ref = moore.get_global_variable @stringQueue : !moore.ref<queue<!moore.string, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.string, 0>>
  %elem = moore.queue.pop_back %queue_ref : !moore.ref<queue<!moore.string, 0>> -> !moore.string
  return %elem : !moore.string
}

// CHECK-LABEL: func @test_queue_pop_front_string
// CHECK: llvm.call @__moore_queue_pop_front_ptr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
func.func @test_queue_pop_front_string() -> !moore.string {
  %queue_ref = moore.get_global_variable @stringQueue : !moore.ref<queue<!moore.string, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.string, 0>>
  %elem = moore.queue.pop_front %queue_ref : !moore.ref<queue<!moore.string, 0>> -> !moore.string
  return %elem : !moore.string
}

//===----------------------------------------------------------------------===//
// Test queue pop with integer types (should still use i64 return)
//===----------------------------------------------------------------------===//

moore.global_variable @intQueue : !moore.queue<!moore.i32, 0>

// CHECK-LABEL: func @test_queue_pop_back_int
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_queue_pop_back({{.*}}, {{.*}}) : (!llvm.ptr, i64) -> i64
// CHECK: arith.trunci [[RESULT]] : i64 to i32
func.func @test_queue_pop_back_int() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @intQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.queue.pop_back %queue_ref : !moore.ref<queue<!moore.i32, 0>> -> !moore.i32
  return %elem : !moore.i32
}

// CHECK-LABEL: func @test_queue_pop_front_int
// CHECK: [[RESULT:%.+]] = llvm.call @__moore_queue_pop_front({{.*}}, {{.*}}) : (!llvm.ptr, i64) -> i64
// CHECK: arith.trunci [[RESULT]] : i64 to i32
func.func @test_queue_pop_front_int() -> !moore.i32 {
  %queue_ref = moore.get_global_variable @intQueue : !moore.ref<queue<!moore.i32, 0>>
  %queue = moore.read %queue_ref : <queue<!moore.i32, 0>>
  %elem = moore.queue.pop_front %queue_ref : !moore.ref<queue<!moore.i32, 0>> -> !moore.i32
  return %elem : !moore.i32
}
