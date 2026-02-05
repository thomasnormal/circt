// RUN: circt-sim %s | FileCheck %s
// RUN: circt-sim --skip-passes %s | FileCheck %s

// Test that disable fork marks children complete so llhd.halt can proceed.
// This mirrors the UVM timeout watchdog pattern (fork/join_any + disable fork).

// CHECK: Parent process starting
// CHECK: Child A done
// CHECK: Parent after disable
// CHECK-NOT: Child B done
// CHECK: Simulation finished successfully

!i64 = i64

llvm.func @__moore_delay(i64)

hw.module @disable_fork_halt_test() {
  %fmt_parent_start = sim.fmt.literal "Parent process starting\0A"
  %fmt_child_a_done = sim.fmt.literal "Child A done\0A"
  %fmt_child_b_done = sim.fmt.literal "Child B done\0A"
  %fmt_parent_after = sim.fmt.literal "Parent after disable\0A"
  %fmt_success = sim.fmt.literal "Simulation finished successfully\0A"

  llhd.process {
    sim.proc.print %fmt_parent_start

    %handle = sim.fork join_type "join_any" {
      // Child A finishes immediately.
      sim.proc.print %fmt_child_a_done
      sim.fork.terminator
    }, {
      // Child B waits; should be killed by disable_fork.
      %c100ns = llvm.mlir.constant(100000000 : i64) : !i64
      llvm.call @__moore_delay(%c100ns) : (i64) -> ()
      sim.proc.print %fmt_child_b_done
      sim.fork.terminator
    }

    sim.disable_fork
    sim.proc.print %fmt_parent_after
    sim.proc.print %fmt_success
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
