// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top i3c_samplewrite_joinnone_disable_fork_ordering_tb 2>&1 | FileCheck %s

// Regression: model the I3C monitor shape where a join_none child and
// wrDetect_stop wake on the same edge, then the parent executes disable fork.
// The child wake must be consumed before disable kills remaining children.

// CHECK: CHILD_SAMPLE count=1
// CHECK: STOP_WOKE
// CHECK: PASS
// CHECK-NOT: FAIL

module i3c_samplewrite_joinnone_disable_fork_ordering_tb;
  reg clk = 1'b0;
  integer sample_count = 0;

  task automatic sample_write_data();
    @(posedge clk);
    sample_count = sample_count + 1;
    $display("CHILD_SAMPLE count=%0d", sample_count);
    @(posedge clk); // Remaining work should be killed by disable fork.
    sample_count = sample_count + 100;
    $display("CHILD_SAMPLE_LATE count=%0d", sample_count);
  endtask

  task automatic wrDetect_stop();
    @(posedge clk);
    $display("STOP_WOKE");
  endtask

  task automatic sampleWriteDataAndACK();
    fork
      begin
        sample_write_data();
      end
    join_none

    wrDetect_stop();
    disable fork;
  endtask

  initial begin
    fork
      begin
        #1 clk = 1'b1;
        #1 clk = 1'b0;
        #1 clk = 1'b1;
        #1 clk = 1'b0;
      end
      begin
        sampleWriteDataAndACK();
      end
    join

    #1;
    if (sample_count == 1)
      $display("PASS");
    else
      $display("FAIL count=%0d", sample_count);

    $finish;
  end
endmodule
