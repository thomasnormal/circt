// RUN: circt-verilog %s --ir-llhd --no-uvm-auto-include -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=200000000000 2>&1 | FileCheck %s

// Test that wait(condition) works with signal-based conditions.
// This tests the fix for llhd.prb invalidation in wait condition polling.
// Previously, the wait would never wake up because the cached llhd.prb
// result was not invalidated when the signal changed.

// CHECK: Initial counter = 0 at time 0
// CHECK: Waiting for counter to reach 5 at 0
// CHECK: Incremented counter to 1 at 10
// CHECK: Incremented counter to 2 at 20
// CHECK: Incremented counter to 3 at 30
// CHECK: Incremented counter to 4 at 40
// CHECK: Incremented counter to 5 at 50
// CHECK: Wait completed! counter = 5 at 50
// CHECK: Test completed at 100

module wait_condition_signal;
  int counter;

  initial begin
    counter = 0;
    $display("Initial counter = %0d at time %0t", counter, $time);

    // Fork a process that waits on counter and one that increments it
    fork
      begin
        // Wait for counter to reach 5
        $display("Waiting for counter to reach 5 at %0t", $time);
        wait(counter == 5);
        $display("Wait completed! counter = %0d at %0t", counter, $time);
      end
      begin
        // Increment counter with delays
        repeat(10) begin
          #10ns;
          counter = counter + 1;
          $display("Incremented counter to %0d at %0t", counter, $time);
        end
      end
    join

    $display("Test completed at %0t", $time);
    $finish;
  end
endmodule
