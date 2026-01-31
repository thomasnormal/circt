// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that associative array iterator operations (first/next/etc) on function
// ref parameters use llvm.load/store instead of llhd.prb/drv.
// The simulator cannot track signal references through function call boundaries.

// CHECK-LABEL: func.func @test_assoc_first_func_param
// CHECK-SAME: (%[[ARRAY:.*]]: !llvm.ptr, %[[KEY_REF:.*]]: !llhd.ref<i32>)
func.func @test_assoc_first_func_param(%array: !moore.ref<assoc_array<i32, i32>>, %key_ref: !moore.ref<i32>) -> !moore.i1 {
  // Key is function ref parameter - should use llvm.load to read initial value
  // CHECK: %[[KEY_PTR:.*]] = builtin.unrealized_conversion_cast %[[KEY_REF]] : !llhd.ref<i32> to !llvm.ptr
  // CHECK: %[[INITIAL_KEY:.*]] = llvm.load %[[KEY_PTR]] : !llvm.ptr -> i32
  // CHECK: llvm.store %[[INITIAL_KEY]], %[[ALLOCA:.*]] : i32, !llvm.ptr
  // CHECK: llvm.call @__moore_assoc_first
  // After call, write back updated key with llvm.store (not llhd.drv)
  // CHECK: %[[UPDATED_KEY:.*]] = llvm.load %[[ALLOCA]] : !llvm.ptr -> i32
  // CHECK: %[[KEY_PTR2:.*]] = builtin.unrealized_conversion_cast %[[KEY_REF]] : !llhd.ref<i32> to !llvm.ptr
  // CHECK: llvm.store %[[UPDATED_KEY]], %[[KEY_PTR2]] : i32, !llvm.ptr
  // CHECK-NOT: llhd.drv
  %result = moore.assoc.first %array, %key_ref : !moore.ref<assoc_array<i32, i32>>, !moore.ref<i32>
  return %result : !moore.i1
}

// CHECK-LABEL: func.func @test_assoc_next_func_param
func.func @test_assoc_next_func_param(%array: !moore.ref<assoc_array<i32, i32>>, %key_ref: !moore.ref<i32>) -> !moore.i1 {
  // CHECK: llvm.load %{{.*}} : !llvm.ptr -> i32
  // CHECK: llvm.call @__moore_assoc_next
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
  // CHECK-NOT: llhd.drv
  %result = moore.assoc.next %array, %key_ref : !moore.ref<assoc_array<i32, i32>>, !moore.ref<i32>
  return %result : !moore.i1
}

// Test that signals (not function params) still use llhd.prb/drv
// CHECK-LABEL: hw.module @test_assoc_first_signal
moore.module @test_assoc_first_signal() {
  %array = moore.variable : !moore.ref<assoc_array<i32, i32>>
  %key = moore.variable : !moore.ref<i32>

  moore.procedure initial {
    // For module-level signals, should use llhd.prb to read
    // CHECK: llhd.prb
    // CHECK: llvm.call @__moore_assoc_first
    // After call, write back with llhd.drv (signal semantics)
    // CHECK: llhd.drv
    %result = moore.assoc.first %array, %key : !moore.ref<assoc_array<i32, i32>>, !moore.ref<i32>
    moore.return
  }

  moore.output
}
