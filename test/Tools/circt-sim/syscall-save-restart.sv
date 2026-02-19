// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $save, $restart, $reset_count — these are no-ops but should not crash
module top;
  int counter;

  initial begin
    counter = 42;
    $save("checkpoint_test.dat");
    counter = 99;
    // CHECK: counter=99
    $display("counter=%0d", counter);
    // $reset_count returns 0 since $reset is never called
    // CHECK: reset_count=0
    $display("reset_count=%0d", $reset_count);
    $finish;
  end
endmodule
