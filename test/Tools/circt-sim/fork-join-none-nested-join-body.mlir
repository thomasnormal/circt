// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s

// Regression test for the SEQBDYZMB fix.
//
// This test exercises the pattern where a join_none fork (simulating the
// "master_phase_process" in UVM) spawns a child, and that child creates a
// blocking Join fork whose body modifies a global counter. Without the
// SEQBDYZMB fix, the executePhaseBlockingPhaseMap would propagate from the
// parent to the join_none child, causing the child's inner Join fork to be
// intercepted (skipped). This would prevent the body fork from executing,
// leaving the counter at 0 instead of 1.
//
// The test verifies that:
// 1. The join_none child starts
// 2. The child's inner Join fork body runs (increments counter from 0 to 1)
// 3. After the Join fork completes, the child reads back counter == 1
// 4. The parent terminates the simulation
//
// CHECK: join_none child started
// CHECK: body fork ran
// CHECK: counter=1
// CHECK: Simulation completed

module {
  llvm.mlir.global internal @counter(0 : i32) : i32

  hw.module @top() {
    %fmt_child_started = sim.fmt.literal "join_none child started\0A"
    %fmt_body_ran = sim.fmt.literal "body fork ran\0A"
    %fmt_counter_prefix = sim.fmt.literal "counter="
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      // Outer join_none fork (simulates "master_phase_process" spawn).
      // The parent continues immediately.
      %handle = sim.fork join_type "join_none" {
        sim.proc.print %fmt_child_started

        // Inner blocking Join fork (simulates the body fork in
        // uvm_sequence_base::start()). This MUST execute its body.
        %inner = sim.fork {
          // Body: increment the global counter from 0 to 1.
          %ptr = llvm.mlir.addressof @counter : !llvm.ptr
          %c1 = arith.constant 1 : i32
          %v = llvm.load %ptr : !llvm.ptr -> i32
          %v1 = arith.addi %v, %c1 : i32
          llvm.store %v1, %ptr : i32, !llvm.ptr
          sim.proc.print %fmt_body_ran
          sim.fork.terminator
        }

        // After the Join fork completes, read back the counter.
        // If the body fork was skipped, counter would still be 0.
        %ptr2 = llvm.mlir.addressof @counter : !llvm.ptr
        %final = llvm.load %ptr2 : !llvm.ptr -> i32
        %fmt_val = sim.fmt.dec %final : i32
        %out = sim.fmt.concat (%fmt_counter_prefix, %fmt_val, %fmt_nl)
        sim.proc.print %out

        sim.fork.terminator
      }

      // Parent: wait a small amount then terminate.
      %c10000000 = hw.constant 10000000 : i64
      %delay = llhd.int_to_time %c10000000
      llhd.wait delay %delay, ^done

    ^done:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
