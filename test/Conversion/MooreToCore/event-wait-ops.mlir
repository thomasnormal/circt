// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Event Triggered Operation
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @__moore_event_triggered(!llvm.ptr) -> i1

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
// Wait Condition Operation
//===----------------------------------------------------------------------===//

// CHECK: llvm.func @__moore_wait_condition(i32)

// CHECK-LABEL: hw.module @test_wait_condition
moore.module @test_wait_condition(in %clk: !moore.i1) {
  // CHECK: llhd.process
  moore.procedure initial {
    // CHECK: [[COND:%.+]] = llvm.zext %{{.*}} : i1 to i32
    // CHECK: llvm.call @__moore_wait_condition([[COND]]) : (i32) -> ()
    moore.wait_condition %clk : !moore.i1
    moore.return
  }
  moore.output
}
