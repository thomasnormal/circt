// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test stochastic analysis queues: $q_initialize, $q_add, $q_remove, $q_exam, $q_full
module top;
  integer q_id, status, value;

  initial begin
    // Initialize a FIFO queue (type 1) with max size 4
    $q_initialize(q_id, 1, 4, status);
    // CHECK: init_status=0
    $display("init_status=%0d", status);

    // Add items
    $q_add(q_id, 0, 10, status);
    // CHECK: add1_status=0
    $display("add1_status=%0d", status);

    $q_add(q_id, 0, 20, status);
    $q_add(q_id, 0, 30, status);

    // Examine front of queue without removing
    $q_exam(q_id, 0, value, status);
    // CHECK: exam_value=10
    $display("exam_value=%0d", value);

    // Remove from front (FIFO order)
    $q_remove(q_id, 0, value, status);
    // CHECK: remove1=10
    $display("remove1=%0d", value);

    $q_remove(q_id, 0, value, status);
    // CHECK: remove2=20
    $display("remove2=%0d", value);

    $q_remove(q_id, 0, value, status);
    // CHECK: remove3=30
    $display("remove3=%0d", value);

    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
