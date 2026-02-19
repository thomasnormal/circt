// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: env CIRCT_SIM_TRACE_I3C_FORK_RUNTIME=1 circt-sim %t.mlir --top i3c_samplewrite_disable_fork_ordering_tb 2>&1 | FileCheck %s

// Regression: model the I3C monitor sampleWriteDataAndACK ordering shape:
// one join_any branch completes, sibling waits on edge, parent immediately
// does disable fork after waking the sibling. Wake must be consumed first.

// CHECK-DAG: [I3C-FORK-RUNTIME] tag=disable_fork_enter{{.*}}mode=deferred
// CHECK-DAG: [I3C-FORK-RUNTIME] tag=disable_fork_defer
// CHECK-DAG: [I3C-FORK-RUNTIME] tag=disable_fork_resume_parent
// CHECK: MONITOR_CHILD_WOKE iter=0
// CHECK: MONITOR_CHILD_WOKE iter=1
// CHECK: MONITOR_CHILD_WOKE iter=2
// CHECK: PASS
// CHECK-NOT: FAIL

module i3c_samplewrite_disable_fork_ordering_tb;
  reg clk = 1'b0;
  integer iter = 0;

  task automatic sampleWriteDataAndACK(input integer iter);
    fork
      begin
        $display("MONITOR_JOIN_ANY_DONE iter=%0d", iter);
      end
      begin
        @(posedge clk);
        $display("MONITOR_CHILD_WOKE iter=%0d", iter);
      end
    join_any

    // Wake waiting sibling, then disable fork in same parent turn.
    clk = 1'b1;
    disable fork;
    #1;
    clk = 1'b0;
    #1;
  endtask

  initial begin
    for (iter = 0; iter < 3; iter = iter + 1)
      sampleWriteDataAndACK(iter);
    #2;
    $display("PASS");
    $finish;
  end
endmodule
