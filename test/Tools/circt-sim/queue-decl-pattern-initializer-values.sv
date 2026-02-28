// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=10000000 2>&1 | FileCheck %s

// Regression for issue #30: declaration-time queue pattern initializer must
// populate queue contents.
module tb;
  int q[$] = '{1, 2, 3, 4, 5};

  initial begin
    if (q.size() == 5 && q[0] == 1 && q[1] == 2 && q[2] == 3 && q[3] == 4 &&
        q[4] == 5)
      $display("PASS");
    else
      $display("FAIL size=%0d q0=%0d q1=%0d q2=%0d q3=%0d q4=%0d", q.size(),
               q[0], q[1], q[2], q[3], q[4]);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
