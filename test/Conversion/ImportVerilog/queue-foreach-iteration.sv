// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

//===----------------------------------------------------------------------===//
// Queue Foreach Iteration Test - Iteration 121 Track D
//===----------------------------------------------------------------------===//
//
// This test validates that foreach loops over queues use size-based iteration
// (for i=0 to size-1) rather than associative array iteration (first/next).
//
// This is important for TLM analysis ports which use queues internally.
//
//===----------------------------------------------------------------------===//

module queue_foreach_test;
  // Test queue of integers
  int queue_int [$];

  // Test queue of class handles
  class Transaction;
    int data;
    function new(int d);
      data = d;
    endfunction
  endclass

  Transaction txn_queue [$];

  initial begin
    int sum;

    // Populate integer queue
    queue_int.push_back(10);
    queue_int.push_back(20);
    queue_int.push_back(30);

    // Foreach over integer queue - should use size-based iteration
    sum = 0;
    foreach (queue_int[i]) begin
      sum = sum + queue_int[i];
      $display("queue_int[%0d] = %0d", i, queue_int[i]);
    end
    $display("Sum = %0d", sum);

    // Populate transaction queue
    begin
      Transaction t1 = new(100);
      Transaction t2 = new(200);
      txn_queue.push_back(t1);
      txn_queue.push_back(t2);
    end

    // Foreach over class queue - should use size-based iteration
    foreach (txn_queue[j]) begin
      $display("txn_queue[%0d].data = %0d", j, txn_queue[j].data);
    end
  end
endmodule

// Queue foreach should use size-based iteration, not associative array iteration
// CHECK: moore.module @queue_foreach_test
// CHECK: %queue_int = moore.variable : <queue<i32, 0>>
// CHECK: moore.procedure initial
// CHECK: moore.read %queue_int : <queue<i32, 0>>
// CHECK: moore.array.size {{.*}} : queue<i32, 0>
// CHECK: moore.slt {{.*}} : i32 -> i1
