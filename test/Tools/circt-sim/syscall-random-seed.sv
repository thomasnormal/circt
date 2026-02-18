// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $random — signed random number generation
// $random returns a signed 32-bit integer (IEEE 1800-2017 18.13.1)
module top;
  integer r1, r2, r3, r4;

  initial begin
    // $random without argument should produce different values
    r1 = $random;
    r2 = $random;
    r3 = $random;

    // At least two of three should differ (probability ~1)
    // CHECK: random_vary=1
    $display("random_vary=%0d", (r1 != r2) || (r2 != r3) || (r1 != r3));

    // At least one should be non-zero
    // CHECK: random_nonzero=1
    $display("random_nonzero=%0d", (r1 != 0) || (r2 != 0) || (r3 != 0));

    // Verify $random produces 32-bit values (can be negative since signed)
    // At least one of 3 random values should have bit 31 set (with high probability)
    // We just check that the values are not all identical (which a no-op returning 0 would be)
    r4 = r1 + r2 + r3;
    // CHECK: random_sum_nonzero=1
    $display("random_sum_nonzero=%0d", r4 != 0);

    $finish;
  end
endmodule
