// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test semaphore: new, put, get, try_get
module top;
  semaphore sem;
  int ok;

  initial begin
    sem = new(2);  // 2 keys initially

    // try_get should succeed
    ok = sem.try_get(1);
    // CHECK: try_get1=1
    $display("try_get1=%0d", ok);

    ok = sem.try_get(1);
    // CHECK: try_get2=1
    $display("try_get2=%0d", ok);

    // No keys left — try_get should fail
    ok = sem.try_get(1);
    // CHECK: try_get3=0
    $display("try_get3=%0d", ok);

    // Put keys back
    sem.put(2);

    // Now try_get should succeed again
    ok = sem.try_get(1);
    // CHECK: try_get_after_put=1
    $display("try_get_after_put=%0d", ok);

    $finish;
  end
endmodule
