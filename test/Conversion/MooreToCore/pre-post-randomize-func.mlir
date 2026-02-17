// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that pre_randomize and post_randomize callbacks are found when they
// exist as func.func but NOT as ClassMethodDeclOp (the common case from
// ImportVerilog, since these methods are not virtual).
// IEEE 1800-2017 Section 18.6.1 "Pre and post randomize methods"

// Class without ClassMethodDeclOp for the callbacks (typical ImportVerilog output)
moore.class.classdecl @CallbackClassFunc {
  moore.class.propertydecl @x : !moore.i32 rand_mode rand
  // Note: no methoddecl for pre_randomize or post_randomize
}

// Implementation functions named with ClassName::methodName convention
func.func private @"CallbackClassFunc::pre_randomize"(%this: !moore.class<@CallbackClassFunc>) {
  return
}

func.func private @"CallbackClassFunc::post_randomize"(%this: !moore.class<@CallbackClassFunc>) {
  return
}

// CHECK-LABEL: func.func @test_pre_randomize_func_lookup
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         call @"CallbackClassFunc::pre_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_pre_randomize_func_lookup(%obj: !moore.class<@CallbackClassFunc>) {
  moore.call_pre_randomize %obj : !moore.class<@CallbackClassFunc>
  return
}

// CHECK-LABEL: func.func @test_post_randomize_func_lookup
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK:         call @"CallbackClassFunc::post_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_post_randomize_func_lookup(%obj: !moore.class<@CallbackClassFunc>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@CallbackClassFunc>
  return
}

// Test class with only pre_randomize as func.func
moore.class.classdecl @PreOnlyFuncClass {
  moore.class.propertydecl @y : !moore.i32 rand_mode rand
}

func.func private @"PreOnlyFuncClass::pre_randomize"(%this: !moore.class<@PreOnlyFuncClass>) {
  return
}

// CHECK-LABEL: func.func @test_pre_only_func_pre
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK:         call @"PreOnlyFuncClass::pre_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_pre_only_func_pre(%obj: !moore.class<@PreOnlyFuncClass>) {
  moore.call_pre_randomize %obj : !moore.class<@PreOnlyFuncClass>
  return
}

// CHECK-LABEL: func.func @test_pre_only_func_post
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_pre_only_func_post(%obj: !moore.class<@PreOnlyFuncClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@PreOnlyFuncClass>
  return
}

// Test class with only post_randomize as func.func
moore.class.classdecl @PostOnlyFuncClass {
  moore.class.propertydecl @z : !moore.i32 rand_mode rand
}

func.func private @"PostOnlyFuncClass::post_randomize"(%this: !moore.class<@PostOnlyFuncClass>) {
  return
}

// CHECK-LABEL: func.func @test_post_only_func_pre
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr)
// CHECK-NOT:     call @
// CHECK:         return
func.func @test_post_only_func_pre(%obj: !moore.class<@PostOnlyFuncClass>) {
  moore.call_pre_randomize %obj : !moore.class<@PostOnlyFuncClass>
  return
}

// CHECK-LABEL: func.func @test_post_only_func_post
// CHECK-SAME:    (%[[OBJ:.*]]: !llvm.ptr, %[[S:.*]]: i1)
// CHECK:         call @"PostOnlyFuncClass::post_randomize"(%[[OBJ]]) : (!llvm.ptr) -> ()
// CHECK:         return
func.func @test_post_only_func_post(%obj: !moore.class<@PostOnlyFuncClass>, %success: i1) {
  moore.call_post_randomize %obj, %success : !moore.class<@PostOnlyFuncClass>
  return
}
