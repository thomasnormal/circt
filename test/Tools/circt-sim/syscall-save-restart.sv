// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test $save and $restart — checkpoint/restore simulation state
module top;
  int counter;

  initial begin
    counter = 42;
    $save("checkpoint_test.dat");
    counter = 99;
    // CHECK: before_restart=99
    $display("before_restart=%0d", counter);
    $restart("checkpoint_test.dat");
    // After restart, counter should be restored to 42
    // CHECK: after_restart=42
    $display("after_restart=%0d", counter);
    // CHECK: reset_count=1
    $display("reset_count=%0d", $reset_count);
    $finish;
  end
endmodule
