// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $get_initial_random_seed returns a meaningful non-zero seed.
// Bug: Always returns 0 (hardcoded constant).
// IEEE 1800-2017 Section 20.15.2: Returns the initial seed value.
//
// This test verifies the seed is non-zero AND is a positive integer.
module top;
  integer seed;

  initial begin
    seed = $get_initial_random_seed;

    // Seed should be positive (non-zero, non-negative)
    // CHECK: seed_positive=1
    $display("seed_positive=%0d", seed > 0);

    // Print the value — must match a positive integer pattern
    // CHECK: seed_val={{[1-9][0-9]*}}
    $display("seed_val=%0d", seed);

    $finish;
  end
endmodule
