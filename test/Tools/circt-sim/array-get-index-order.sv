// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that hw.array_get uses LSB-first indexing (element 0 = least significant).
// This catches a bug where array_get(0) returned the MSB element instead of LSB,
// causing infinite loops in enum iteration (e.g., UVM severity_count reset).

module top;
  initial begin
    static int lookup[4] = '{10, 20, 30, 40};
    static int next_key[4] = '{1, 2, 3, 0};
    int idx;
    int count;

    // lookup[0] should be 10, lookup[3] should be 40
    // CHECK: lookup[0] = 10
    $display("lookup[0] = %0d", lookup[0]);
    // CHECK: lookup[1] = 20
    $display("lookup[1] = %0d", lookup[1]);
    // CHECK: lookup[2] = 30
    $display("lookup[2] = %0d", lookup[2]);
    // CHECK: lookup[3] = 40
    $display("lookup[3] = %0d", lookup[3]);

    // Test enum-style iteration (like UVM severity loop)
    idx = 0;
    count = 0;
    while (1) begin
      count = count + 1;
      if (idx == 3) break;
      idx = next_key[idx];
    end
    // CHECK: count = 4
    $display("count = %0d", count);
    // CHECK: PASS
    $display("PASS");
  end
endmodule
