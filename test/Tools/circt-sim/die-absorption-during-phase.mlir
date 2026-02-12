// RUN: circt-sim %s 2>&1 | FileCheck %s

// Test that llvm.unreachable in a function called from an llhd.process
// gracefully halts the process without crashing circt-sim.
//
// This mirrors the die() pattern in UVM where $finish (sim.terminate) is
// followed by llvm.unreachable because the compiler assumes $finish never
// returns. When llvm.unreachable is hit, the process should be finalized
// (halted) and the simulation should complete normally.
//
// The test verifies:
// 1. A print before the die-like function call is visible in output.
// 2. llvm.unreachable does not crash the simulator.
// 3. The simulation completes successfully (exit 0).

// CHECK: before die function
// CHECK: [circt-sim] Simulation completed

module {
  // A function that mimics the die() pattern: sim.terminate then
  // llvm.unreachable. Outside of a UVM phase context, sim.terminate fires
  // normally and llvm.unreachable finalizes the process.
  func.func @die_like_function() {
    sim.terminate success, quiet
    llvm.unreachable
  }

  hw.module @test() {
    %fmt_before = sim.fmt.literal "before die function\0A"

    llhd.process {
      // This print must appear in the output.
      sim.proc.print %fmt_before

      // Call the function containing llvm.unreachable.
      // The process will be finalized (halted) when unreachable is hit.
      func.call @die_like_function() : () -> ()

      // This code is unreachable — the process was already halted above.
      llhd.halt
    }
    hw.output
  }
}
