// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test process::srandom — seed a process's RNG and verify deterministic output
module top;
  int r1, r2, r3, r4;

  initial begin
    // Seed the process RNG
    process::self().srandom(12345);

    // Generate two values
    r1 = $urandom;
    r2 = $urandom;

    // Re-seed with the SAME seed
    process::self().srandom(12345);

    // Generate two more values — must match the first pair
    r3 = $urandom;
    r4 = $urandom;

    // CHECK: seed_deterministic_1=1
    $display("seed_deterministic_1=%0d", r1 == r3);
    // CHECK: seed_deterministic_2=1
    $display("seed_deterministic_2=%0d", r2 == r4);

    // Sanity: the two values from the same seed should differ from each other
    // (unless the RNG is trivially broken)
    // CHECK: seed_differ=1
    $display("seed_differ=%0d", r1 != r2);

    $finish;
  end
endmodule
