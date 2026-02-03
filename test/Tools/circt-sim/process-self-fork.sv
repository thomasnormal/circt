// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

// Test process::self() returns non-null inside fork branches.
// IEEE 1800-2017 Section 9.7 "Process control"
// Each fork branch creates a new process, and process::self() should return
// a handle to that specific process.

module test_process_self_fork;
  initial begin
    process p_main, p_fork1, p_fork2;

    // Get main process handle
    p_main = process::self();

    // CHECK: Main process handle: non-null
    if (p_main != null)
      $display("Main process handle: non-null");
    else
      $display("Main process handle: null");

    fork
      begin
        // Fork branch 1 - should get its own process handle
        p_fork1 = process::self();
        // CHECK: Fork1 process handle: non-null
        if (p_fork1 != null)
          $display("Fork1 process handle: non-null");
        else
          $display("Fork1 process handle: null");
      end
      begin
        // Fork branch 2 - should get its own process handle
        p_fork2 = process::self();
        // CHECK: Fork2 process handle: non-null
        if (p_fork2 != null)
          $display("Fork2 process handle: non-null");
        else
          $display("Fork2 process handle: null");
      end
    join

    // CHECK: TEST PASSED
    $display("TEST PASSED");
    $finish;
  end
endmodule
