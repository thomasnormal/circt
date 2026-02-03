// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000 2>&1 | FileCheck %s

// Test process::self() returns non-null inside a process context.
// IEEE 1800-2017 Section 9.7 "Process control"
// The process::self() static method returns a handle to the currently
// executing process when called from within a process context.

module test_process_self;
  initial begin
    process p;

    // Get current process handle
    p = process::self();

    // CHECK: process::self() returned: non-null
    if (p != null)
      $display("process::self() returned: non-null");
    else
      $display("process::self() returned: null");

    // CHECK: TEST PASSED
    $display("TEST PASSED");
    $finish;
  end
endmodule
