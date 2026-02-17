// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test deep inheritance (5+ levels) with queue property initialization.
// This tests the fix for GEP index calculation in deep class hierarchies.

// Root class (like uvm_void)
moore.class.classdecl @Level0 {
}

// Level 1 (like uvm_factory - no properties, just methods)
moore.class.classdecl @Level1 extends @Level0 {
}

// Level 2 (like uvm_default_factory - has queue properties)
moore.class.classdecl @Level2 extends @Level1 {
  moore.class.propertydecl @prop1 : !moore.i32
  moore.class.propertydecl @queue1 : !moore.queue<i32, 0>
  moore.class.propertydecl @prop2 : !moore.i32
  moore.class.propertydecl @queue2 : !moore.queue<i64, 0>
}

// Level 3 - even deeper
moore.class.classdecl @Level3 extends @Level2 {
  moore.class.propertydecl @prop3 : !moore.i32
  moore.class.propertydecl @queue3 : !moore.queue<i32, 0>
}

// Level 4 - deepest test case
moore.class.classdecl @Level4 extends @Level3 {
  moore.class.propertydecl @prop4 : !moore.i32
}

// Test that class.new correctly initializes queue properties at all levels.
// The GEP paths should use the cached field paths from structInfo.

// CHECK-LABEL: func.func @test_new_level2
// CHECK:   [[MALLOC:%.+]] = llvm.call @malloc
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
// CHECK:   llvm.store {{.*}} : i32, !llvm.ptr
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
// CHECK:   llvm.store {{.*}} : !llvm.ptr, !llvm.ptr
// queue1 should be at index 2, queue2 at index 4
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
// CHECK:   llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level2"
// CHECK:   llvm.store {{.*}} : !llvm.struct<(ptr, i64)>, !llvm.ptr
// CHECK:   return
func.func @test_new_level2() -> !moore.class<@Level2> {
  %obj = moore.class.new : <@Level2>
  return %obj : !moore.class<@Level2>
}

// CHECK-LABEL: func.func @test_new_level4
// Level4 inherits queue properties from Level2 and Level3
// queue1 path: [0, 0, 2] (into Level3, into Level2, index 2)
// queue2 path: [0, 0, 4] (into Level3, into Level2, index 4)
// queue3 path: [0, 2] (into Level3, index 2)
// CHECK:   [[MALLOC:%.+]] = llvm.call @malloc
// Verify we have GEP operations with correct indices (not out-of-bounds)
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 0, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// Queue initializations should use correct cached paths
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 4] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   return
func.func @test_new_level4() -> !moore.class<@Level4> {
  %obj = moore.class.new : <@Level4>
  return %obj : !moore.class<@Level4>
}

// Test property access to inherited queues
// CHECK-LABEL: func.func @test_queue_access
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   llvm.getelementptr {{.*}}[{{%.+}}, 0, 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"Level4"
// CHECK:   return
func.func @test_queue_access(%obj: !moore.class<@Level4>) -> (!moore.ref<queue<i32, 0>>, !moore.ref<queue<i32, 0>>) {
  %ref1 = moore.class.property_ref %obj[@queue1] : <@Level4> -> !moore.ref<queue<i32, 0>>
  %ref2 = moore.class.property_ref %obj[@queue3] : <@Level4> -> !moore.ref<queue<i32, 0>>
  return %ref1, %ref2 : !moore.ref<queue<i32, 0>>, !moore.ref<queue<i32, 0>>
}
