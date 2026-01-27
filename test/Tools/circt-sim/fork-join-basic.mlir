// RUN: circt-sim %s | FileCheck %s

// Test basic sim.fork with join_none semantics.
// The parent should continue immediately after spawning child processes.

// CHECK: Parent starts
// CHECK: Parent continues after fork_none
// CHECK: Child 1 done
// CHECK: Child 2 done

hw.module @fork_join_none_test() {
  %fmt_parent_starts = sim.fmt.literal "Parent starts\0A"
  %fmt_parent_continues = sim.fmt.literal "Parent continues after fork_none\0A"
  %fmt_child1_done = sim.fmt.literal "Child 1 done\0A"
  %fmt_child2_done = sim.fmt.literal "Child 2 done\0A"
  %fmt_done = sim.fmt.literal "Test complete\0A"

  llhd.process {
    // Print that parent is starting
    sim.proc.print %fmt_parent_starts

    // Fork with join_none - parent continues immediately
    %handle = sim.fork join_type "join_none" {
      // Child process 1
      sim.proc.print %fmt_child1_done
      sim.fork.terminator
    }, {
      // Child process 2
      sim.proc.print %fmt_child2_done
      sim.fork.terminator
    }

    // This should execute immediately (join_none semantics)
    sim.proc.print %fmt_parent_continues

    // Terminate simulation
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
