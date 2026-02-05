// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Test that automatic variables in fork loops capture the loop variable's
// current value at fork creation time, not when the fork branch executes.

// CHECK-LABEL: moore.module @ForkLoopCapture
module ForkLoopCapture;
  initial begin
    // Each fork iteration should capture a different value of i.
    // The initializer for local_i must be evaluated BEFORE the fork is created.
    for (int i = 1; i <= 3; i++) begin
      fork
        automatic int local_i = i;
        begin
          // The value of local_i should be 1, 2, or 3 depending on the iteration.
          // CHECK: moore.fork join_none {
          // CHECK:   %local_i = moore.variable
          // CHECK:   moore.fork.terminator
          // CHECK: }
          #(local_i * 10);
        end
      join_none
    end
  end
endmodule

// CHECK-LABEL: moore.module @ForkLoopCaptureMultiVar
module ForkLoopCaptureMultiVar;
  int arr[3];
  initial begin
    // Test multiple automatic variables capturing loop values.
    for (int i = 0; i < 3; i++) begin
      fork
        automatic int idx = i;
        automatic int val = i * 10;
        begin
          // CHECK: moore.fork join_none {
          // CHECK:   %idx = moore.variable
          // CHECK:   %val = moore.variable
          // CHECK: }
          arr[idx] = val;
        end
      join_none
    end
  end
endmodule
