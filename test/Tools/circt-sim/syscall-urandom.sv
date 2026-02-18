// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $urandom and $urandom_range — unsigned random number generation
module top;
  int unsigned r1, r2;
  int in_range_count;

  initial begin
    // $urandom should return a 32-bit unsigned value.
    // Call twice — with overwhelming probability they differ.
    r1 = $urandom;
    r2 = $urandom;
    // At least one should be non-zero (probability of both zero is ~2^-64)
    // CHECK: urandom_nonzero=1
    $display("urandom_nonzero=%0d", (r1 != 0) || (r2 != 0));

    // CHECK: urandom_differ=1
    $display("urandom_differ=%0d", r1 != r2);

    // $urandom_range(max, min) should return a value in [min, max]
    // Test with range [10, 20]
    in_range_count = 0;
    for (int i = 0; i < 100; i++) begin
      r1 = $urandom_range(20, 10);
      if (r1 >= 10 && r1 <= 20)
        in_range_count++;
    end
    // All 100 values must be in range [10,20]
    // CHECK: range_all_in=100
    $display("range_all_in=%0d", in_range_count);

    // $urandom_range with single argument: range [0, max]
    in_range_count = 0;
    for (int i = 0; i < 100; i++) begin
      r1 = $urandom_range(5);
      if (r1 <= 5)
        in_range_count++;
    end
    // CHECK: range0_all_in=100
    $display("range0_all_in=%0d", in_range_count);

    $finish;
  end
endmodule
