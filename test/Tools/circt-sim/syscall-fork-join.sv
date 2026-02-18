// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test fork/join, fork/join_any, fork/join_none, wait fork, disable fork
module top;
  integer a = 0, b = 0, c = 0;

  initial begin
    // fork/join — waits for all
    fork
      begin #1; a = 1; end
      begin #2; b = 2; end
    join
    // CHECK: fork_join: a=1 b=2
    $display("fork_join: a=%0d b=%0d", a, b);

    // fork/join_any — waits for first
    a = 0; b = 0;
    fork
      begin #1; a = 10; end
      begin #5; b = 20; end
    join_any
    // CHECK: fork_join_any: a=10
    $display("fork_join_any: a=%0d", a);

    // fork/join_none — doesn't wait
    c = 0;
    fork
      begin #1; c = 100; end
    join_none
    // CHECK: fork_join_none: c=0
    $display("fork_join_none: c=%0d", c);

    // wait fork
    wait fork;
    // CHECK: after_wait_fork: c=100
    $display("after_wait_fork: c=%0d", c);

    $finish;
  end
endmodule
