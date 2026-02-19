// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: env CIRCT_SIM_TRACE_DISABLE_FORK=1 circt-sim %t.mlir --top fork_disable_ready_wakeup_tb 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_TRACE_I3C_FORK_RUNTIME=1 circt-sim %t.mlir --top fork_disable_ready_wakeup_tb 2>&1 | FileCheck %s --check-prefix=I3CTRACE

// Regression: if a fork child is already queued Ready from a wakeup but still
// has interpreter waiting=true, immediate disable fork can kill it before it
// consumes that wakeup. Ensure the wakeup is observed before disable kills.

// CHECK: [DISABLE-FORK-DEFER]
// CHECK: child B woke
// CHECK: PASS
// CHECK-NOT: FAIL

// I3CTRACE: [I3C-FORK-RUNTIME] tag=disable_fork_enter
// I3CTRACE-SAME: parent_call_stack={{[0-9]+}}
// I3CTRACE-SAME: parent_current_block_ops={{[0-9]+}}

module fork_disable_ready_wakeup_tb;
  reg clk = 0;
  integer saw_b = 0;

  initial begin
    fork
      begin
        $display("child A done");
      end
      begin
        @(posedge clk);
        saw_b = 1;
        $display("child B woke");
      end
    join_any

    clk = 1'b1;
    disable fork;

    #1;
    if (!saw_b)
      $display("FAIL");
    else
      $display("PASS");

    $finish;
  end
endmodule
