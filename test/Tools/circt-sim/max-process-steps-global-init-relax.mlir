// RUN: circt-sim %s --skip-passes --max-process-steps=50 2>&1 | FileCheck %s
//
// CHECK-NOT: PROCESS_STEP_OVERFLOW
// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  // Force non-trivial module-level execution work during global init.
  %c200 = arith.constant 200 : i32
  %0 = func.call @countdown(%c200) : (i32) -> i32

  // End simulation immediately after init completes.
  llhd.process {
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}

func.func @countdown(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  cf.br ^loop(%n : i32)

^loop(%i: i32):
  %cond = arith.cmpi sgt, %i, %c0 : i32
  cf.cond_br %cond, ^step, ^done

^step:
  %next = arith.subi %i, %c1 : i32
  cf.br ^loop(%next : i32)

^done:
  return %i : i32
}
