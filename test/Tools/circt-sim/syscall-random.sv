// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: xrun-compatible seeded random semantics.
// Test $random, $urandom, $urandom_range
module top;
  integer r1, r2;
  integer seed;
  integer ur1, ur2;

  initial begin
    // xrun reference for seed=32'h1234_abcd:
    // first call:  r=1823735769, seed=-323748822
    // second call: r=739840344,  seed=-1407643741
    seed = 32'h1234_abcd;
    r1 = $random(seed);
    // CHECK: random0=1823735769 seed0=-323748822
    $display("random0=%0d seed0=%0d", r1, seed);
    r2 = $random(seed);
    // CHECK: random1=739840344 seed1=-1407643741
    $display("random1=%0d seed1=%0d", r2, seed);

    // xrun seed==0 bootstrap behavior.
    seed = 0;
    r1 = $random(seed);
    // CHECK: random_seed0=303379748 seed0_boot=-1844104698
    $display("random_seed0=%0d seed0_boot=%0d", r1, seed);

    // $random with same seed should produce deterministic results.
    seed = 42;
    r1 = $random(seed);
    // Verify seed was modified (not left unchanged as a no-op would)
    // CHECK: seed_changed=1
    $display("seed_changed=%0d", seed != 42);

    // Two consecutive calls produce different results
    r2 = $random(seed);
    // CHECK: random_diff=1
    $display("random_diff=%0d", r1 !== r2);

    // Seeded $urandom(seed) is pure in seed in xrun and does not mutate it.
    seed = 32'h1234_abcd;
    ur1 = $urandom(seed);
    ur2 = $urandom(seed);
    // CHECK: ur_seed_unchanged=1
    $display("ur_seed_unchanged=%0d", seed == 32'h1234_abcd);
    // CHECK: ur_seed_pure=1
    $display("ur_seed_pure=%0d", ur1 == ur2);
    // CHECK: ur_seed_value=-323769319
    $display("ur_seed_value=%0d", ur1);

    // xrun seeded $urandom zero-seed behavior.
    seed = 0;
    ur1 = $urandom(seed);
    // CHECK: ur_seed0_value=144547288
    $display("ur_seed0_value=%0d", ur1);
    // CHECK: ur_seed0_unchanged=1
    $display("ur_seed0_unchanged=%0d", seed == 0);

    // Unseeded $urandom — two calls should differ (vanishingly small collision chance)
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
