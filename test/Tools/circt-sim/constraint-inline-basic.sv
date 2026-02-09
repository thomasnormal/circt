// RUN: circt-verilog %s -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test inline constraints with randomize() with { ... } syntax
// IEEE 1800-2017 Section 18.7 "In-line constraints"

class packet;
  rand bit [7:0] x;
  rand bit [7:0] y;
endclass

module top;
  initial begin
    packet p = new;
    int ok;

    // Test 1: Single inline constraint with constant upper bound
    // x < 10 means x in [0, 9]
    ok = 1;
    repeat (20) begin
      void'(p.randomize() with { x < 10; });
      if (p.x >= 10) ok = 0;
    end
    // CHECK: PASS1
    if (ok) $display("PASS1");
    else $display("FAIL1: x was not constrained below 10");

    // Test 2: Single inline constraint with constant lower bound
    // x > 200 means x in [201, 255]
    ok = 1;
    repeat (20) begin
      void'(p.randomize() with { x > 200; });
      if (p.x <= 200) ok = 0;
    end
    // CHECK: PASS2
    if (ok) $display("PASS2");
    else $display("FAIL2: x was not constrained above 200");

    // Test 3: Two inline constraints forming a range
    // x >= 50 and x <= 60 means x in [50, 60]
    ok = 1;
    repeat (20) begin
      void'(p.randomize() with { x >= 50; x <= 60; });
      if (p.x < 50 || p.x > 60) ok = 0;
    end
    // CHECK: PASS3
    if (ok) $display("PASS3");
    else $display("FAIL3: x was not constrained to [50, 60]");

    // Test 4: Inline constraint on different property (y)
    ok = 1;
    repeat (20) begin
      void'(p.randomize() with { y < 5; });
      if (p.y >= 5) ok = 0;
    end
    // CHECK: PASS4
    if (ok) $display("PASS4");
    else $display("FAIL4: y was not constrained below 5");

    // Test 5: Multiple properties constrained inline
    ok = 1;
    repeat (20) begin
      void'(p.randomize() with { x >= 100; x <= 110; y >= 200; y <= 210; });
      if (p.x < 100 || p.x > 110 || p.y < 200 || p.y > 210) ok = 0;
    end
    // CHECK: PASS5
    if (ok) $display("PASS5");
    else $display("FAIL5: x or y out of range");

    $finish;
  end
endmodule
