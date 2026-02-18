// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $random, $urandom, $urandom_range
module top;
  integer r1, r2;
  integer seed;
  integer ur1, ur2;

  initial begin
    // $random with same seed should produce deterministic results
    seed = 42;
    r1 = $random(seed);
    // Verify seed was modified (not left unchanged as a no-op would)
    // CHECK: seed_changed=1
    $display("seed_changed=%0d", seed != 42);

    // Two consecutive calls produce different results
    r2 = $random(seed);
    // CHECK: random_diff=1
    $display("random_diff=%0d", r1 !== r2);

    // $urandom — two calls should differ (vanishingly small chance of collision)
    ur1 = $urandom;
    ur2 = $urandom;
    // CHECK: urandom_diff=1
    $display("urandom_diff=%0d", ur1 !== ur2);

    // $urandom_range with min and max — result must be in [5, 10]
    ur1 = $urandom_range(10, 5);
    // CHECK: urange_in_bounds=1
    $display("urange_in_bounds=%0d", (ur1 >= 5) && (ur1 <= 10));

    // $urandom_range with just max (min defaults to 0) — result in [0, 3]
    ur1 = $urandom_range(3);
    // CHECK: urange_max_in_bounds=1
    $display("urange_max_in_bounds=%0d", (ur1 >= 0) && (ur1 <= 3));

    // Verify reproducibility: same seed → same sequence
    seed = 100;
    r1 = $random(seed);
    seed = 100;
    r2 = $random(seed);
    // CHECK: reproducible=1
    $display("reproducible=%0d", r1 == r2);

    $finish;
  end
endmodule
