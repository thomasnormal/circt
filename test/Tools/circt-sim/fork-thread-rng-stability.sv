// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

module test_fork_rng;
  int val1, val2;
  initial begin
    // Two fork blocks with same srandom seed should produce same $urandom values
    fork
      begin
        process p = process::self();
        p.srandom(100);
        val1 = $urandom;
      end
      begin
        process p = process::self();
        p.srandom(100);
        val2 = $urandom;
      end
    join
    // CHECK: FORK_RNG val1==val2: 1
    $display("FORK_RNG val1==val2: %0d", (val1 == val2) ? 1 : 0);
    $finish;
  end
endmodule
