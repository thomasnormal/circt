// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// XFAIL: *
// Reason: Stochastic queue operations ($q_*) are deprecated legacy from Verilog-1364 — not planned.
// Test $q_full to check if stochastic queue is full
module top;
  integer q_id, status, value;
  integer is_full;

  initial begin
    // Initialize FIFO queue with max size 2
    $q_initialize(q_id, 1, 2, status);

    // Check full status when empty
    is_full = $q_full(q_id, status);
    // CHECK: full_empty=0
    $display("full_empty=%0d", is_full);

    // Add 2 items to fill it
    $q_add(q_id, 0, 10, status);
    $q_add(q_id, 0, 20, status);

    // Check full status when full
    is_full = $q_full(q_id, status);
    // CHECK: full_full=1
    $display("full_full=%0d", is_full);

    // Remove one and check again
    $q_remove(q_id, 0, value, status);
    is_full = $q_full(q_id, status);
    // CHECK: full_after_remove=0
    $display("full_after_remove=%0d", is_full);

    $finish;
  end
endmodule
