// RUN: circt-verilog %s --no-uvm-auto-include

// Test $dist_* distribution functions (IEEE 1800-2017 Section 20.15)
// These are stubbed to return 0 for now.

module dist_functions_test;
  int seed;
  int result;

  initial begin
    seed = 42;

    // 2-argument distribution functions
    result = $dist_chi_square(seed, 5);
    result = $dist_exponential(seed, 100);
    result = $dist_t(seed, 10);
    result = $dist_poisson(seed, 50);

    // 3-argument distribution functions
    result = $dist_uniform(seed, 0, 100);
    result = $dist_normal(seed, 50, 10);
    result = $dist_erlang(seed, 3, 100);

    $display("Distribution functions test passed");
  end
endmodule
