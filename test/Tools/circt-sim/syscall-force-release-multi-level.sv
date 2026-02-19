// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test force/release: after release, the signal reverts to its LAST
// procedurally driven value (not the forced value and not the initial value).
// Bug: force is a blocking assignment, release is a no-op.
// After release, signal should revert to the value it was driven to
// BEFORE the force was applied.
module top;
  reg [7:0] sig;

  initial begin
    // Drive to 10
    sig = 10;
    // CHECK: driven=10
    $display("driven=%0d", sig);

    // Force to 200
    force sig = 200;
    #1;
    // CHECK: forced=200
    $display("forced=%0d", sig);

    // Drive a NEW value while force is active — this should NOT take effect
    // because force overrides all drivers. But the "driven value" is updated.
    sig = 77;
    #1;
    // CHECK: still_forced=200
    $display("still_forced=%0d", sig);

    // Release — should revert to the last driven value (77)
    release sig;
    #1;
    // CHECK: released=77
    $display("released=%0d", sig);

    $finish;
  end
endmodule
