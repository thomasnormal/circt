// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top semaphore_tb 2>&1 | FileCheck %s

// Test: basic semaphore operations - create, try_get, put, get.
// Regression test for semaphore interceptor support.

// CHECK: try_get with 1 key: 1
// CHECK: try_get with 1 key (empty): 0
// CHECK: put then get ok
// CHECK: [circt-sim] Simulation completed
module semaphore_tb();
  semaphore sem;

  initial begin
    sem = new(1);

    // try_get should succeed (1 key available)
    if (sem.try_get(1))
      $display("try_get with 1 key: 1");
    else
      $display("try_get with 1 key: 0");

    // try_get should fail (0 keys available)
    if (sem.try_get(1))
      $display("try_get with 1 key (empty): 1");
    else
      $display("try_get with 1 key (empty): 0");

    // put a key back and get it
    sem.put(1);
    sem.get(1);
    $display("put then get ok");
  end
endmodule
