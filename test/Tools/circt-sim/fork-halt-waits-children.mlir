// RUN: circt-sim %s | FileCheck %s

// Test that llhd.halt waits for forked children to complete before halting.
// This tests the UVM phase termination scenario where run_test() spawns
// UVM phases via fork-join_none, but the parent process should not halt
// until all forked children have completed.

// Expected order:
// 1. Parent starts and forks children with join_none
// 2. Parent continues immediately (join_none semantics) and prints messages
// 3. Parent reaches halt, but halt is deferred until children complete
// 4. Children execute and complete
// 5. Simulation terminates successfully

// The key test: children MUST complete before simulation finishes.
// With the fix, llhd.halt waits for active forked children before terminating.

// CHECK: Parent process starting
// CHECK: Parent reached halt point
// CHECK: Simulation success
// CHECK-DAG: Child 1 executing
// CHECK-DAG: Child 2 executing
// CHECK-DAG: Child 1 complete
// CHECK-DAG: Child 2 complete
// CHECK: Simulation completed

hw.module @fork_halt_waits_children_test() {
  %fmt_parent_start = sim.fmt.literal "Parent process starting\0A"
  %fmt_parent_halt = sim.fmt.literal "Parent reached halt point\0A"
  %fmt_child1_exec = sim.fmt.literal "Child 1 executing\0A"
  %fmt_child1_done = sim.fmt.literal "Child 1 complete\0A"
  %fmt_child2_exec = sim.fmt.literal "Child 2 executing\0A"
  %fmt_child2_done = sim.fmt.literal "Child 2 complete\0A"
  %fmt_success = sim.fmt.literal "Simulation success\0A"

  llhd.process {
    sim.proc.print %fmt_parent_start

    // Fork with join_none - parent continues immediately
    // This simulates UVM's run_test() spawning phases
    %handle = sim.fork join_type "join_none" {
      // Child process 1 - simulates a UVM phase
      sim.proc.print %fmt_child1_exec
      sim.proc.print %fmt_child1_done
      sim.fork.terminator
    }, {
      // Child process 2 - simulates another UVM phase
      sim.proc.print %fmt_child2_exec
      sim.proc.print %fmt_child2_done
      sim.fork.terminator
    }

    // With join_none, we reach here immediately, but halt should wait
    // for children to complete before actually terminating
    sim.proc.print %fmt_parent_halt

    // This halt should be deferred until all children complete
    sim.proc.print %fmt_success
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
