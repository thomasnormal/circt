// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $get_initial_random_seed returns a non-zero value.
// Bug: $get_initial_random_seed always returns 0.
// IEEE 1800-2017 Section 20.15.2: Returns the initial seed value
// used for the simulation. This should be non-zero (either from a
// default seed or from a command-line specified seed).
module top;
  integer seed;

  initial begin
    seed = $get_initial_random_seed;
    // The seed should be non-zero (every simulation should have a seed)
    // CHECK: seed_nonzero=1
    $display("seed_nonzero=%0d", seed != 0);
    $finish;
  end
endmodule
