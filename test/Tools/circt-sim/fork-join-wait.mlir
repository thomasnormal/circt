// RUN: circt-sim %s | FileCheck %s

// Test sim.fork with join semantics.
// The parent should wait until all children complete before continuing.

// With join semantics, the expected order is:
// 1. Parent starts
// 2. Children execute (order not guaranteed between children)
// 3. Parent continues after all children complete
//
// CHECK: Parent starts
// CHECK-DAG: Child 1 done
// CHECK-DAG: Child 2 done
// CHECK: Parent continues after join

hw.module @fork_join_wait_test() {
  %fmt_parent_starts = sim.fmt.literal "Parent starts\0A"
  %fmt_parent_continues = sim.fmt.literal "Parent continues after join\0A"
  %fmt_child1_done = sim.fmt.literal "Child 1 done\0A"
  %fmt_child2_done = sim.fmt.literal "Child 2 done\0A"

  llhd.process {
    sim.proc.print %fmt_parent_starts

    // Fork with join - parent waits for all children
    %handle = sim.fork join_type "join" {
      sim.proc.print %fmt_child1_done
      sim.fork.terminator
    }, {
      sim.proc.print %fmt_child2_done
      sim.fork.terminator
    }

    // This executes only after BOTH children complete
    sim.proc.print %fmt_parent_continues

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
