// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test randomize() with inline constraints
module top;
  class packet;
    rand int data;
  endclass

  initial begin
    packet p = new();
    int ok;

    // Randomize with inline constraint
    ok = p.randomize() with { data > 100 && data < 200; };
    // CHECK: randomize_with_ok=1
    $display("randomize_with_ok=%0d", ok);
    // CHECK: data_constrained=1
    $display("data_constrained=%0d", (p.data > 100) && (p.data < 200));

    // Randomize(null) — check-only mode
    p.data = 150;
    ok = p.randomize(null);
    // CHECK: randomize_null=1
    $display("randomize_null=%0d", ok);
    // Value should be unchanged
    // CHECK: data_unchanged=150
    $display("data_unchanged=%0d", p.data);

    $finish;
  end
endmodule
