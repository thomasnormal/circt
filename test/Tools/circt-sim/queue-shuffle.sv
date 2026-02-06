// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that queue.shuffle() randomizes element order while preserving all elements.
// This exercises the __moore_queue_shuffle interceptor in the interpreter.
// Since shuffle is random, we verify by checking size preservation and that
// all original elements are still present (by sorting after shuffle).

module top;
  int q[$];

  initial begin
    // Push values
    q.push_back(42);
    q.push_back(7);
    q.push_back(99);
    q.push_back(3);
    q.push_back(55);

    // CHECK: size before = 5
    $display("size before = %0d", q.size());

    // Shuffle the queue
    q.shuffle();

    // CHECK: size after = 5
    $display("size after = %0d", q.size());

    // Sort the shuffled queue to verify all elements are preserved
    q.sort();

    // Verify all original elements are still present in sorted order
    // CHECK: q[0] = 3
    $display("q[0] = %0d", q[0]);
    // CHECK: q[1] = 7
    $display("q[1] = %0d", q[1]);
    // CHECK: q[2] = 42
    $display("q[2] = %0d", q[2]);
    // CHECK: q[3] = 55
    $display("q[3] = %0d", q[3]);
    // CHECK: q[4] = 99
    $display("q[4] = %0d", q[4]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
