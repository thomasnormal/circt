// RUN: circt-sim %s --max-cycles=100 2>&1 | FileCheck %s
//
// Test that transitive self-driven signal dependencies are properly filtered.
// When a process drives signal Z through a module-level drive that reads
// signal X, both Z and X should be filtered from the sensitivity list to
// prevent zero-delta feedback loops.
//
// The scenario:
// - Process P yields a result that depends on probing signal X
// - Module-level drive uses P's result to drive signal Z
// - P's wait observes both X and Z (via yield operands)
// - Without transitive filtering: P wakes on X, computes, drives Z, but
//   still sensitive to X -> infinite delta loop
// - With transitive filtering: X is recognized as transitively self-driven
//   (because the module-level drive value depends on X), so both X and Z
//   are filtered from sensitivity when non-self signals exist

hw.module @test(in %clk : i1, in %external_input : i1) {
  %eps = llhd.constant_time <0ns, 1d, 0e>
  %true = hw.constant 1 : i1
  %false = hw.constant 0 : i1

  // Signal X - will be transitively self-driven
  %x = llhd.sig %false : i1
  // Signal Z - directly self-driven by module-level drive
  %z = llhd.sig %false : i1
  // External signal - should NOT be filtered
  %ext = llhd.sig %external_input : i1

  // Process that reads X and external, and outputs a computed value
  %proc_result = llhd.process -> i1 {
  ^entry:
    llhd.wait yield (%false : i1), (), ^loop
  ^loop:
    %x_val = llhd.prb %x : i1
    %ext_val = llhd.prb %ext : i1
    %computed = comb.xor %x_val, %ext_val : i1
    // Wait on X, Z, and external. X and Z should be filtered (transitive self-driven).
    // Only external should remain, preventing zero-delta loops.
    llhd.wait yield (%computed : i1), (%x_val, %ext_val : i1, i1), ^loop
  }

  // Module-level drive: Z = proc_result (which depends on X)
  // This creates: Z <- proc_result <- X (transitive dependency)
  llhd.drv %z, %proc_result after %eps : i1

  // Also drive X from the process to test direct module-level drive filtering
  llhd.drv %x, %proc_result after %eps : i1
}

// CHECK-NOT: delta cycle overflow
// CHECK: simulation finished
