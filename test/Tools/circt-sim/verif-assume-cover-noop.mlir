// RUN: circt-sim %s 2>&1 | FileCheck %s

// Test: verif.assume and verif.cover in processes are no-ops in simulation.
// Deferred immediate assertions (assume #0, cover #0, assume final, cover final)
// generate processes with infinite loops. The per-activation step limit catches
// these loops, and the simulation completes cleanly.

// CHECK: Simulation completed

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false

  // Mimics 'assume #0 (expr)' - infinite loop with verif.assume
  llhd.process {
    cf.br ^bb1
  ^bb1:
    verif.assume %true label "assume_test" : i1
    cf.br ^bb1
  }

  // Mimics 'cover #0 (expr)' - infinite loop with verif.cover
  llhd.process {
    cf.br ^bb1
  ^bb1:
    verif.cover %true label "cover_test" : i1
    cf.br ^bb1
  }

  // Mimics 'assert #0 (expr)' - infinite loop with verif.assert
  llhd.process {
    cf.br ^bb1
  ^bb1:
    verif.assert %true label "assert_test" : i1
    cf.br ^bb1
  }

  hw.output
}
