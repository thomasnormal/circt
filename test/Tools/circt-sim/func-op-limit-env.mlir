// RUN: env CIRCT_SIM_MAX_FUNC_OPS=64 circt-sim %s --max-time=100000000 2>&1 | FileCheck %s
// Test that CIRCT_SIM_MAX_FUNC_OPS enforces a per-function operation cap.

// CHECK: circt-sim: Function 'spin' exceeded 64 operations for process

module {
  func.func @spin(%seed: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    cf.br ^loop(%seed : i32)
  ^loop(%value: i32):
    %next = arith.addi %value, %c1 : i32
    cf.br ^loop(%next : i32)
  }

  hw.module @top() {
    %t1 = llhd.constant_time <1ns, 0d, 0e>
    llhd.process {
      cf.br ^bb0
    ^bb0:
      %c0 = arith.constant 0 : i32
      %unused = func.call @spin(%c0) : (i32) -> i32
      llhd.wait delay %t1, ^bb1
    ^bb1:
      llhd.halt
    }
    hw.output
  }
}
