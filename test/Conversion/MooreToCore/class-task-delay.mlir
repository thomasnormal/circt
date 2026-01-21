// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// Test that delay statements (#delay) in class tasks/methods are correctly
// lowered to runtime function calls instead of llhd.wait (which requires
// an llhd.process parent).

// CHECK: llvm.func @__moore_delay(i64)

// Define a simple class for testing
moore.class.classdecl @TestClass {
  moore.class.propertydecl @value : !moore.i32
}

//===----------------------------------------------------------------------===//
// Test 1: Delay in a class method (func.func context)
// This should lower to a runtime call, not llhd.wait
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @"TestClass::run_phase"
// CHECK-SAME: (%arg0: !llvm.ptr)
// CHECK: %[[DELAY:.*]] = llvm.mlir.constant(10000000 : i64) : i64
// CHECK: llvm.call @__moore_delay(%[[DELAY]]) : (i64) -> ()
// CHECK: return
func.func @"TestClass::run_phase"(%this: !moore.class<@TestClass>) {
  %delay = moore.constant_time 10000000 fs
  moore.wait_delay %delay
  return
}

//===----------------------------------------------------------------------===//
// Test 2: Multiple delays in a class method
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @"TestClass::multiple_delays"
// CHECK-DAG: %[[DELAY1:.*]] = llvm.mlir.constant(5000000 : i64) : i64
// CHECK-DAG: %[[DELAY2:.*]] = llvm.mlir.constant(15000000 : i64) : i64
// CHECK: llvm.call @__moore_delay(%[[DELAY1]]) : (i64) -> ()
// CHECK: llvm.call @__moore_delay(%[[DELAY2]]) : (i64) -> ()
// CHECK: return
func.func @"TestClass::multiple_delays"(%this: !moore.class<@TestClass>) {
  %delay1 = moore.constant_time 5000000 fs
  %delay2 = moore.constant_time 15000000 fs
  moore.wait_delay %delay1
  moore.wait_delay %delay2
  return
}

//===----------------------------------------------------------------------===//
// Test 3: Delay with dynamic time value (passed as argument)
// Uses unrealized_conversion_cast for llhd.time -> i64
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @"TestClass::dynamic_delay"
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llhd.time)
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg1 : !llhd.time to i64
// CHECK: llvm.call @__moore_delay(%[[CAST]]) : (i64) -> ()
// CHECK: return
func.func @"TestClass::dynamic_delay"(%this: !moore.class<@TestClass>, %delay: !moore.time) {
  moore.wait_delay %delay
  return
}

//===----------------------------------------------------------------------===//
// Test 4: Delay in module procedure (should still use llhd.wait)
// This verifies we didn't break the normal case
//===----------------------------------------------------------------------===//

// CHECK-LABEL: hw.module @TestDelayInProcess
moore.module @TestDelayInProcess() {
  // CHECK: llhd.process {
  // CHECK:   llhd.wait delay
  moore.procedure initial {
    %delay = moore.constant_time 10000000 fs
    moore.wait_delay %delay
    moore.return
  }
}
