// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test queue.sort.with operation lowering

moore.global_variable @intQueue : !moore.queue<!moore.i32, 0>

//===----------------------------------------------------------------------===//
// Queue Sort With Operations
//===----------------------------------------------------------------------===//

// Test sort.with using modulo as key
// CHECK-LABEL: func @test_queue_sort_with_mod
// CHECK: llvm.load {{.*}} : !llvm.ptr -> !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr, i64)>
// CHECK: llvm.alloca
// CHECK: llvm.alloca
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
func.func @test_queue_sort_with_mod() {
  %queue_ref = moore.get_global_variable @intQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %key = moore.mods %item, %ten : i32
    moore.queue.sort.key.yield %key : i32
  }
  return
}

// Test rsort.with using modulo as key (descending)
// CHECK-LABEL: func @test_queue_rsort_with_mod
// CHECK: scf.for
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     arith.cmpi slt
// CHECK:     scf.if
// CHECK: scf.for
// CHECK: scf.for
func.func @test_queue_rsort_with_mod() {
  %queue_ref = moore.get_global_variable @intQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.rsort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    %ten = moore.constant 10 : i32
    %key = moore.mods %item, %ten : i32
    moore.queue.sort.key.yield %key : i32
  }
  return
}

// Test sort.with with simple identity key (effectively same as sort)
// CHECK-LABEL: func @test_queue_sort_with_identity
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
func.func @test_queue_sort_with_identity() {
  %queue_ref = moore.get_global_variable @intQueue : !moore.ref<queue<!moore.i32, 0>>
  moore.queue.sort.with %queue_ref : !moore.ref<queue<!moore.i32, 0>> {
  ^bb0(%item: !moore.i32):
    moore.queue.sort.key.yield %item : i32
  }
  return
}
