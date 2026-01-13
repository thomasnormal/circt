// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

// CHECK-LABEL: moore.module @WaitForkTest()
module WaitForkTest;
  int x;

  // CHECK: moore.procedure initial {
  initial begin
    // CHECK: moore.wait_fork
    wait fork;
  end

  // Test wait fork after variable assignment (common UVM pattern)
  // CHECK: moore.procedure initial {
  initial begin
    x = 1;
    // CHECK: moore.wait_fork
    wait fork;
    x = 2;
  end
endmodule
