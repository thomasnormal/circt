// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $get_initial_random_seed returns the same non-zero value on each call.
// Bug: $get_initial_random_seed always returns 0.
// IEEE 1800-2017 Section 20.15.2: Returns the initial seed value
// for the simulation. Must be consistent across calls.
module top;
  integer seed1, seed2;

  initial begin
    seed1 = $get_initial_random_seed;
    seed2 = $get_initial_random_seed;

    // Both calls should return the same value
    // CHECK: seeds_match=1
    $display("seeds_match=%0d", seed1 == seed2);

    // The seed must be non-zero
    // CHECK: seed1_nonzero=1
    $display("seed1_nonzero=%0d", seed1 != 0);

    // Print actual value for debugging
    // CHECK: seed1={{[1-9][0-9]*}}
    $display("seed1=%0d", seed1);

    $finish;
  end
endmodule
