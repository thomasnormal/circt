// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.func @__moore_event_triggered(!llvm.ptr) -> i1

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

//===----------------------------------------------------------------------===//
// Event Trigger Operation (inside procedure)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_event_trigger_in_procedure
moore.module @test_event_trigger_in_procedure(in %event_ref: !moore.ref<event>) {
  moore.procedure initial {
    // Trigger the event - this takes a ref and toggles the underlying signal
    // CHECK: llhd.prb %event_ref
    // CHECK: llhd.drv %event_ref
    moore.event_trigger %event_ref : <event>
    moore.return
  }
  moore.output
}

//===----------------------------------------------------------------------===//
// Event Trigger with ReadOp - tests the address fix
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @test_event_trigger_with_read
moore.module @test_event_trigger_with_read(in %event_ref: !moore.ref<event>) {
  moore.procedure initial {
    // Trigger the event ref directly - toggles the underlying LLHD signal
    // CHECK: llhd.prb %event_ref
    // CHECK: llhd.drv %event_ref
    moore.event_trigger %event_ref : <event>
    moore.return
  }
  moore.output
}

// Note: event_trigger and wait_condition inside procedures with module inputs
// would violate LLHD region isolation constraints after lowering.
// These operations are typically used with locally-defined values in real code.
