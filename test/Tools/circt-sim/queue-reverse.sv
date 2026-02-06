// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --top top 2>&1 | FileCheck %s

// Test that queue.reverse() reverses element order.
// This exercises the __moore_queue_reverse interceptor in the interpreter.

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

    // Reverse the queue
    q.reverse();

    // CHECK: size after = 5
    $display("size after = %0d", q.size());

    // Verify elements are in reversed order
    // CHECK: q[0] = 55
    $display("q[0] = %0d", q[0]);
    // CHECK: q[1] = 3
    $display("q[1] = %0d", q[1]);
    // CHECK: q[2] = 99
    $display("q[2] = %0d", q[2]);
    // CHECK: q[3] = 7
    $display("q[3] = %0d", q[3]);
    // CHECK: q[4] = 42
    $display("q[4] = %0d", q[4]);

    // CHECK: PASS
    $display("PASS");
    $finish;
  end
endmodule
