// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that pre_randomize and post_randomize callbacks are called directly
// when the class has user-defined implementations.
// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods"

// Class with pre_randomize and post_randomize methods
moore.class.classdecl @RandomizeCallbackClass {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  moore.class.methoddecl @pre_randomize -> @"RandomizeCallbackClass::pre_randomize" : (!moore.class<@RandomizeCallbackClass>) -> ()
  moore.class.methoddecl @post_randomize -> @"RandomizeCallbackClass::post_randomize" : (!moore.class<@RandomizeCallbackClass>) -> ()
}

// Implementation functions
func.func private @"RandomizeCallbackClass::pre_randomize"(%this: !moore.class<@RandomizeCallbackClass>) {
  return
}

func.func private @"RandomizeCallbackClass::post_randomize"(%this: !moore.class<@RandomizeCallbackClass>) {
  return
}

// CHECK-LABEL: func.func @test_pre_randomize
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         call @"RandomizeCallbackClass::pre_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_pre_randomize(%obj: !moore.class<@RandomizeCallbackClass>) {
  moore.call_pre_randomize %obj : !moore.class<@RandomizeCallbackClass>
  return
}

// CHECK-LABEL: func.func @test_post_randomize
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK:         call @"RandomizeCallbackClass::post_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_post_randomize(%obj: !moore.class<@RandomizeCallbackClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@RandomizeCallbackClass>
  return
}

// Test class without pre/post_randomize methods - callbacks should be no-ops
moore.class.classdecl @NoCallbackClass {
  moore.class.propertydecl @y : !moore.i32 rand_mode rand
}

// CHECK-LABEL: func.func @test_no_pre_randomize
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_no_pre_randomize(%obj: !moore.class<@NoCallbackClass>) {
  moore.call_pre_randomize %obj : !moore.class<@NoCallbackClass>
  return
}

// CHECK-LABEL: func.func @test_no_post_randomize
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_no_post_randomize(%obj: !moore.class<@NoCallbackClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@NoCallbackClass>
  return
}

// Test class with only pre_randomize (not post_randomize)
moore.class.classdecl @PreOnlyClass {
  moore.class.propertydecl @z : !moore.i32 rand_mode rand
  moore.class.methoddecl @pre_randomize -> @"PreOnlyClass::pre_randomize" : (!moore.class<@PreOnlyClass>) -> ()
}

func.func private @"PreOnlyClass::pre_randomize"(%this: !moore.class<@PreOnlyClass>) {
  return
}

// CHECK-LABEL: func.func @test_pre_only_pre
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         call @"PreOnlyClass::pre_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_pre_only_pre(%obj: !moore.class<@PreOnlyClass>) {
  moore.call_pre_randomize %obj : !moore.class<@PreOnlyClass>
  return
}

// CHECK-LABEL: func.func @test_pre_only_post
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_pre_only_post(%obj: !moore.class<@PreOnlyClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@PreOnlyClass>
  return
}

// Test class with only post_randomize (not pre_randomize)
moore.class.classdecl @PostOnlyClass {
  moore.class.propertydecl @w : !moore.i32 rand_mode rand
  moore.class.methoddecl @post_randomize -> @"PostOnlyClass::post_randomize" : (!moore.class<@PostOnlyClass>) -> ()
}

func.func private @"PostOnlyClass::post_randomize"(%this: !moore.class<@PostOnlyClass>) {
  return
}

// CHECK-LABEL: func.func @test_post_only_pre
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_post_only_pre(%obj: !moore.class<@PostOnlyClass>) {
  moore.call_pre_randomize %obj : !moore.class<@PostOnlyClass>
  return
}

// CHECK-LABEL: func.func @test_post_only_post
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK:         call @"PostOnlyClass::post_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_post_only_post(%obj: !moore.class<@PostOnlyClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@PostOnlyClass>
  return
}
