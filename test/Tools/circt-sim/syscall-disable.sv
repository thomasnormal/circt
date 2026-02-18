// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test disable statement for named blocks and tasks
module top;
  integer i;

  task automatic count_task;
    for (i = 0; i < 10; i = i + 1) begin
      if (i == 3) return;
    end
  endtask

  initial begin
    count_task();
    // CHECK: after_disable=3
    $display("after_disable=%0d", i);

    // Named block disable
    begin : my_block
      for (i = 0; i < 10; i = i + 1) begin
        if (i == 5) disable my_block;
      end
    end
    // CHECK: after_block_disable=5
    $display("after_block_disable=%0d", i);

    $finish;
  end
endmodule
