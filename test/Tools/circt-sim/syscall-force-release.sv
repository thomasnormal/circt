// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test force/release — release should revert to the driven value.
// Bug: force is converted to a blocking assignment, release is a no-op.
// After release, the signal should revert to the last procedurally
// driven value, not retain the forced value.
module top;
  reg [7:0] a;
  initial begin
    // Drive a to 42 (not 0, not the forced value)
    a = 42;
    force a = 99;
    #1;
    // CHECK: forced=99
    $display("forced=%0d", a);
    release a;
    #1;
    // After release, a should revert to its pre-force driven value of 42
    // (not 99, and not 0 — this distinguishes real release from "a = 0")
    // CHECK: released=42
    $display("released=%0d", a);
    $finish;
  end
endmodule
