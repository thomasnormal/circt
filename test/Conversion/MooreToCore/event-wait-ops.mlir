// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK: llvm.func @__moore_event_triggered(!llvm.ptr) -> i1

//===----------------------------------------------------------------------===//
// Event Triggered Operation (combinational context)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_event_triggered
moore.module @test_event_triggered(in %event: !moore.event, out result: !moore.i1) {
  // CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca %[[ONE]] x i1 : (i64) -> !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %[[ALLOCA]] : i1, !llvm.ptr
  // CHECK: %[[RESULT:.*]] = llvm.call @__moore_event_triggered(%[[ALLOCA]]) : (!llvm.ptr) -> i1
  %triggered = moore.event_triggered %event : !moore.event
  moore.output %triggered : !moore.i1
}

// Note: event_trigger and wait_condition inside procedures with module inputs
// would violate LLHD region isolation constraints after lowering.
// These operations are typically used with locally-defined values in real code.
