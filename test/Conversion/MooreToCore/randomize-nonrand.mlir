// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_randomize_basic(!llvm.ptr, i64) -> i32

// Class with one non-rand and one rand property.
moore.class.classdecl @RandClass {
  moore.class.propertydecl @fixed : !moore.i32
  moore.class.propertydecl @rnd : !moore.i32 rand_mode rand
}

// CHECK-LABEL: func.func @test_randomize_preserve
// CHECK-SAME: (%[[OBJ:.*]]: !llvm.ptr)
func.func @test_randomize_preserve(%obj: !moore.class<@RandClass>) -> i1 {
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: llvm.call @__moore_randomize_basic
  // CHECK: llvm.store
  %success = moore.randomize %obj : !moore.class<@RandClass>
  return %success : i1
}
