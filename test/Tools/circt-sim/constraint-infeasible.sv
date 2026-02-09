// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test infeasible constraint detection.
// IEEE 1800-2017 Section 18.6.3: randomize() returns 0 when constraints
// are infeasible, random variables retain previous values, and
// post_randomize() is not called.

class a;
  rand int b = 42;
  int d = 1;

  // Infeasible constraint: b == 0 AND b > 0
  constraint c { b == 0 && b > 0; }

  function void post_randomize();
    d = 99;
  endfunction
endclass

module top;
  initial begin
    a obj = new;
    int status;
    int prev_b;

    prev_b = obj.b;
    status = obj.randomize();

    // Test 1: randomize() should return 0 (failure)
    // CHECK: INFEASIBLE_STATUS: 0
    $display("INFEASIBLE_STATUS: %0d", status);

    // Test 2: b should retain its previous value
    // CHECK: INFEASIBLE_RETAIN: 1
    $display("INFEASIBLE_RETAIN: %0d", (obj.b == prev_b) ? 1 : 0);

    // Test 3: post_randomize should NOT have been called
    // CHECK: POST_RAND_SKIPPED: 1
    $display("POST_RAND_SKIPPED: %0d", (obj.d == 1) ? 1 : 0);

    $finish;
  end
endmodule
