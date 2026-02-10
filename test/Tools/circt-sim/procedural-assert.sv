// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test that procedural (immediate) assertions do not halt the process.
// A failing procedural assert should report the failure but continue execution.
// Only clocked/concurrent assertions should halt.

module top;
  int x;

  initial begin
    x = 5;
    // This assertion passes - execution continues
    assert(x == 5);
    // CHECK: PASS: after passing assert
    $display("PASS: after passing assert");

    // This assertion fails - execution should still continue
    // CHECK: Assertion failed
    assert(x == 99);
    // CHECK: PASS: after failing assert
    $display("PASS: after failing assert");

    $finish;
  end
endmodule
