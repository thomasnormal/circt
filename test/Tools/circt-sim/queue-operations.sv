// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that queue.unique(), queue.sort(), and queue.reverse() work together.
// This exercises the __moore_queue_unique, __moore_queue_sort, and
// __moore_queue_reverse interceptors in the interpreter.

module top;
  int q[$];
  int unique_q[$];

  initial begin
    // Push values with duplicates: 42, 7, 42, 99, 7, 3, 99, 55, 3
    q.push_back(42);
    q.push_back(7);
    q.push_back(42);
    q.push_back(99);
    q.push_back(7);
    q.push_back(3);
    q.push_back(99);
    q.push_back(55);
    q.push_back(3);

    // CHECK: initial size = 9
    $display("initial size = %0d", q.size());

    // Test unique() - returns new queue with only unique values
    unique_q = q.unique();

    // CHECK: after unique size = 5
    $display("after unique size = %0d", unique_q.size());

    // Verify unique elements (in order of first occurrence):
    // Original order of first occurrences: 42, 7, 99, 3, 55
    // CHECK: unique_q[0] = 42
    $display("unique_q[0] = %0d", unique_q[0]);
    // CHECK: unique_q[1] = 7
    $display("unique_q[1] = %0d", unique_q[1]);
    // CHECK: unique_q[2] = 99
    $display("unique_q[2] = %0d", unique_q[2]);
    // CHECK: unique_q[3] = 3
    $display("unique_q[3] = %0d", unique_q[3]);
    // CHECK: unique_q[4] = 55
    $display("unique_q[4] = %0d", unique_q[4]);

    // Sort the unique result (ascending order)
    unique_q.sort();

    // CHECK: after sort size = 5
    $display("after sort size = %0d", unique_q.size());

    // Verify sorted order: 3, 7, 42, 55, 99
    // CHECK: sorted[0] = 3
    $display("sorted[0] = %0d", unique_q[0]);
    // CHECK: sorted[1] = 7
    $display("sorted[1] = %0d", unique_q[1]);
    // CHECK: sorted[2] = 42
    $display("sorted[2] = %0d", unique_q[2]);
    // CHECK: sorted[3] = 55
    $display("sorted[3] = %0d", unique_q[3]);
    // CHECK: sorted[4] = 99
    $display("sorted[4] = %0d", unique_q[4]);

    // Reverse the sorted queue
    unique_q.reverse();

    // CHECK: after reverse size = 5
    $display("after reverse size = %0d", unique_q.size());

    // Verify reversed order: 99, 55, 42, 7, 3
    // CHECK: reversed[0] = 99
    $display("reversed[0] = %0d", unique_q[0]);
    // CHECK: reversed[1] = 55
    $display("reversed[1] = %0d", unique_q[1]);
    // CHECK: reversed[2] = 42
    $display("reversed[2] = %0d", unique_q[2]);
    // CHECK: reversed[3] = 7
    $display("reversed[3] = %0d", unique_q[3]);
    // CHECK: reversed[4] = 3
    $display("reversed[4] = %0d", unique_q[4]);

    // Test original queue is still intact (size should still be 9)
    // CHECK: original q size = 9
    $display("original q size = %0d", q.size());

    // Push more values to original queue
    q.push_back(100);
    q.push_back(50);

    // CHECK: q size after push = 11
    $display("q size after push = %0d", q.size());

    // Verify last two elements
    // CHECK: q[9] = 100
    $display("q[9] = %0d", q[9]);
    // CHECK: q[10] = 50
    $display("q[10] = %0d", q[10]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
