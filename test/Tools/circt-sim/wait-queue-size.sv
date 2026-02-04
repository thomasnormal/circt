// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=10000000000 2>&1 | FileCheck %s

// Test that wait(queue.size() != 0) properly wakes up when queue is modified.
// This tests the wait_condition polling mechanism with queue-based conditions.

module test;
  int q[$];
  
  initial begin
    $display("T=%0t: Starting wait queue test", $time);
    fork
      begin
        // CHECK: Fork branch 1 - waiting
        $display("T=%0t: Fork branch 1 - waiting for q.size() != 0", $time);
        wait (q.size() != 0);
        // CHECK-DAG: Fork branch 1 - done
        $display("T=%0t: Fork branch 1 - done, q.size()=%0d", $time, q.size());
      end
      begin
        #2;
        // CHECK-DAG: Fork branch 2 - pushing
        $display("T=%0t: Fork branch 2 - pushing to queue", $time);
        q.push_back(42);
        $display("T=%0t: Fork branch 2 - pushed, q.size()=%0d", $time, q.size());
      end
    join
    // CHECK-DAG: Test complete
    $display("T=%0t: Test complete", $time);
    $finish;
  end
endmodule
