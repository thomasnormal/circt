// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $stacktrace — print call stack
module top;

  task inner_task;
    $stacktrace;
  endtask

  task outer_task;
    inner_task();
  endtask

  initial begin
    outer_task();
    // $stacktrace should show the function call hierarchy
    // CHECK-DAG: inner_task
    // CHECK-DAG: outer_task
    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
