// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $dist_uniform, $dist_normal, $dist_exponential, $dist_poisson,
// $dist_chi_square, $dist_t, $dist_erlang
module top;
  integer seed, r;
  integer seed2, r2;

  initial begin
    seed = 1;

    // $dist_uniform(seed, start, end) — uniform distribution in [0, 100]
    r = $dist_uniform(seed, 0, 100);
    // CHECK: uniform_in_range=1
    $display("uniform_in_range=%0d", (r >= 0) && (r <= 100));

    // Call again — seed was modified, result should differ
    seed2 = 1;
    r2 = $dist_uniform(seed2, 0, 100);
    // Verify seed was actually modified (not just returning 0)
    // CHECK: uniform_seed_changed=1
    $display("uniform_seed_changed=%0d", seed2 != 1);

    // $dist_normal(seed, mean, std_dev) — normal distribution
    // Result should be in a reasonable range around mean=1000
    seed = 42;
    r = $dist_normal(seed, 1000, 100);
    // CHECK: normal_reasonable=1
    $display("normal_reasonable=%0d", (r > 0) && (r < 5000));
    // Verify seed was modified
    // CHECK: normal_seed_changed=1
    $display("normal_seed_changed=%0d", seed != 42);

    // $dist_exponential(seed, mean) — exponential distribution, result >= 0
    seed = 7;
    r = $dist_exponential(seed, 100);
    // CHECK: exp_nonneg=1
    $display("exp_nonneg=%0d", r >= 0);
    // CHECK: exp_seed_changed=1
    $display("exp_seed_changed=%0d", seed != 7);

    // $dist_poisson(seed, mean) — poisson distribution, result >= 0
    seed = 13;
    r = $dist_poisson(seed, 50);
    // CHECK: poisson_nonneg=1
    $display("poisson_nonneg=%0d", r >= 0);
    // CHECK: poisson_seed_changed=1
    $display("poisson_seed_changed=%0d", seed != 13);

    // $dist_chi_square(seed, deg_of_freedom) — chi-square, result >= 0
    seed = 19;
    r = $dist_chi_square(seed, 3);
    // CHECK: chi_nonneg=1
    $display("chi_nonneg=%0d", r >= 0);
    // CHECK: chi_seed_changed=1
    $display("chi_seed_changed=%0d", seed != 19);

    // $dist_t(seed, deg_of_freedom) — Student's t distribution
    seed = 23;
    r = $dist_t(seed, 5);
    // Verify seed was modified (result can be negative)
    // CHECK: t_seed_changed=1
    $display("t_seed_changed=%0d", seed != 23);

    // $dist_erlang(seed, k, mean) — Erlang distribution, result >= 0
    seed = 29;
    r = $dist_erlang(seed, 2, 100);
    // CHECK: erlang_nonneg=1
    $display("erlang_nonneg=%0d", r >= 0);
    // CHECK: erlang_seed_changed=1
    $display("erlang_seed_changed=%0d", seed != 29);

    $finish;
  end
endmodule
