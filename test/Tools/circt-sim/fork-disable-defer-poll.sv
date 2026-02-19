// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: env CIRCT_SIM_TRACE_DISABLE_FORK=1 circt-sim %t.mlir --top fork_disable_defer_poll_tb 2>&1 | FileCheck %s

// Regression: bounded deferred disable_fork polling for suspended waiters.
// This should terminate without letting the child body run.

// CHECK: [DISABLE-FORK]
// CHECK: [DISABLE-FORK-DEFER]
// CHECK: [DISABLE-FORK-DEFER-FIRE]
// CHECK: mode=deferred
// CHECK-NOT: CHILD_RAN
// CHECK: PASS
// CHECK-NOT: FAIL

module fork_disable_defer_poll_tb;
  bit gate = 0;
  integer saw = 0;

  initial begin
    fork
      begin
        wait (gate);
        saw = 1;
        $display("CHILD_RAN");
      end
    join_none

    #1;
    disable fork;

    #1;
    if (saw == 0)
      $display("PASS");
    else
      $display("FAIL");
    $finish;
  end
endmodule
