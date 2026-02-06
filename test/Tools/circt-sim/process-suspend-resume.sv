// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top process_suspend_resume_tb 2>&1 | FileCheck %s

// Test: process::suspend() and process::resume() methods.
// Verifies that suspend/resume lowering works end-to-end.

// CHECK: child_status_before={{.*}}
// CHECK: suspended
// CHECK: resumed
// CHECK: [circt-sim] Simulation completed
module process_suspend_resume_tb();
  initial begin
    process p;

    fork
      begin
        p = process::self();
        #100;
      end
    join_none

    // Wait for child to register
    #1;

    $display("child_status_before=%0d", p.status());
    p.suspend();
    $display("suspended");
    #10;
    p.resume();
    $display("resumed");

    #200;
    $finish;
  end
endmodule
