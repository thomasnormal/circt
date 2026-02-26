// RUN: circt-sim --max-time=20000000 %s 2>&1 | FileCheck %s
//
// Regression: when a process waits in a function-local blocking fork/join and
// resumes from child completion (non-signal wake), interpreter call-stack
// resume must not treat that wake as spurious and drop the continuation.
//
// CHECK: BEFORE_FORK
// CHECK: CHILD_DONE
// CHECK: AFTER_FORK
// CHECK: Simulation completed

module {
  llvm.func @__moore_delay(i64)

  func.func @child_fn() {
    %c1000000_i64 = hw.constant 1000000 : i64
    llvm.call @__moore_delay(%c1000000_i64) : (i64) -> ()
    %child = sim.fmt.literal "CHILD_DONE\0A"
    sim.proc.print %child
    return
  }

  func.func @parent_fn() {
    %before = sim.fmt.literal "BEFORE_FORK\0A"
    %after = sim.fmt.literal "AFTER_FORK\0A"
    sim.proc.print %before
    %h = sim.fork join_type "join" {
      func.call @child_fn() : () -> ()
      sim.fork.terminator
    }
    sim.proc.print %after
    return
  }

  hw.module @top() {
    llhd.process {
      func.call @parent_fn() : () -> ()
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
