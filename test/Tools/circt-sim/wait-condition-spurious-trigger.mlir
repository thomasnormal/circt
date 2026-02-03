// RUN: circt-sim %s --max-time=100000000 2>&1 | FileCheck %s

// Test that wait(condition) properly handles spurious triggers.
// This is a regression test for the infinite scheduling loop bug where:
// 1. Fork creates a branch that calls wait(condition) with false condition
// 2. wait(condition) sets waiting=true and schedules poll callback
// 3. Before poll fires, signal change triggers process again
// 4. Previously: Process would restart, re-evaluate false condition, loop infinitely
// 5. Now: Process ignores spurious trigger and only resumes via poll callback

// The test creates a scenario where:
// - Parent forks a child that waits on a memory-based condition
// - Parent immediately drives a signal that would trigger the child
// - The child should NOT loop infinitely but wait for the poll callback

// CHECK: Parent started
// CHECK-DAG: Child waiting on condition
// CHECK-DAG: Parent driving signal
// CHECK: Condition now true
// CHECK: Child condition met
// CHECK: Test PASSED
// CHECK: Simulation finished successfully
// CHECK-NOT: ERROR

module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @__moore_wait_condition(i32)

  hw.module @top() {
    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %c16_i64 = hw.constant 16 : i64

    %fmt_parent_start = sim.fmt.literal "Parent started\0A"
    %fmt_child_wait = sim.fmt.literal "Child waiting on condition\0A"
    %fmt_parent_drive = sim.fmt.literal "Parent driving signal\0A"
    %fmt_cond_true = sim.fmt.literal "Condition now true\0A"
    %fmt_child_met = sim.fmt.literal "Child condition met\0A"
    %fmt_pass = sim.fmt.literal "Test PASSED\0A"

    %time_epsilon = llhd.constant_time <0ns, 0d, 1e>
    %time_10ns = llhd.constant_time <10ns, 0d, 0e>
    %null = llvm.mlir.zero : !llvm.ptr

    // Signal that will be driven after fork
    %trigger_sig = llhd.sig %c0_i32 : i32

    // Signal to hold condition variable pointer
    %cond_ptr_sig = llhd.sig %null : !llvm.ptr

    llhd.process {
      sim.proc.print %fmt_parent_start

      // Allocate condition variable (simulating class member)
      %ptr = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      llvm.store %c0_i32, %ptr : i32, !llvm.ptr

      // Store pointer in signal
      llhd.drv %cond_ptr_sig, %ptr after %time_epsilon : !llvm.ptr

      // Fork child that will wait on condition
      %handle = sim.fork join_type "join_none" {
        sim.proc.print %fmt_child_wait

        // Get the condition pointer
        %cond_ptr = llhd.prb %cond_ptr_sig : !llvm.ptr

        // Wait for condition (value == 1) - uses polling
        %cond_val = llvm.load %cond_ptr : !llvm.ptr -> i32
        %cond = comb.icmp eq %cond_val, %c1_i32 : i32
        %cond_i32 = llvm.zext %cond : i1 to i32
        llvm.call @__moore_wait_condition(%cond_i32) : (i32) -> ()

        sim.proc.print %fmt_child_met
        sim.proc.print %fmt_pass
        sim.terminate success, quiet
        sim.fork.terminator
      }

      // Parent immediately drives a signal - this could spuriously trigger
      // the child process before its poll callback fires
      sim.proc.print %fmt_parent_drive
      llhd.drv %trigger_sig, %c1_i32 after %time_epsilon : i32

      // Wait and then update condition to true
      llhd.wait delay %time_10ns, ^bb1(%ptr : !llvm.ptr)
    ^bb1(%ptr_arg: !llvm.ptr):
      sim.proc.print %fmt_cond_true
      llvm.store %c1_i32, %ptr_arg : i32, !llvm.ptr

      llhd.halt
    }

    hw.output
  }
}
