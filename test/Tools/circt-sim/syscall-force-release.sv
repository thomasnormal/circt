// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test force/release — release should revert to the driven value.
// Bug: force is converted to a blocking assignment, release is a no-op.
// After release, the signal should revert to the last procedurally
// driven value (0), not retain the forced value (1).
module top;
  reg a;
  initial begin
    a = 0;
    force a = 1;
    #1;
    // CHECK: forced=1
    $display("forced=%0d", a);
    release a;
    #1;
    // After release, a should revert to its pre-force value of 0
    // CHECK: released=0
    $display("released=%0d", a);
    $finish;
  end
endmodule
