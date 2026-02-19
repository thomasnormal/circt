// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $stacktrace — print the lexical scope hierarchy at the call site.
// In compiled mode, this walks the compile-time scope chain (not the dynamic
// call stack), so it shows the task/module names in the lexical enclosure.
module top;

  task inner_task;
    $stacktrace;
  endtask

  task outer_task;
    inner_task();
  endtask

  initial begin
    outer_task();
    // CHECK: inner_task
    // CHECK: top
    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
